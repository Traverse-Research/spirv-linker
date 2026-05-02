//! End-to-end check: link two SPIR-V modules produced by DXC's
//! `-fspv-allow-import` Import-linkage path against an Export library.
//!
//! The fixtures under `tests/data/` are pre-built by DXC so this test does
//! not depend on a local DXC checkout. To regenerate them:
//!
//! ```text
//! dxc -spirv -T lib_6_x -fspv-target-env=universal1.5 \
//!     tests/data/src/lib.hlsl -Fo tests/data/dxc_lib.spv
//! dxc -spirv -T lib_6_x -E main -fspv-target-env=universal1.5 \
//!     -fspv-allow-import \
//!     tests/data/src/consumer.hlsl -Fo tests/data/dxc_consumer.spv
//! ```

use rspirv::dr::{Module, Operand};
use rspirv::spirv::{Capability, Decoration, LinkageType, Op};

const LIB: &[u8] = include_bytes!("data/dxc_lib.spv");
const CONSUMER: &[u8] = include_bytes!("data/dxc_consumer.spv");

fn words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn name_of(module: &Module, id: u32) -> Option<&str> {
    for inst in &module.debug_names {
        if inst.class.opcode != Op::Name {
            continue;
        }
        let Some(Operand::IdRef(target)) = inst.operands.first() else {
            continue;
        };
        if *target != id {
            continue;
        }
        if let Some(Operand::LiteralString(s)) = inst.operands.get(1) {
            return Some(s);
        }
    }
    None
}

#[test]
fn link_dxc_export_and_import() {
    let lib = words(LIB);
    let consumer = words(CONSUMER);

    // Sanity-check the inputs match what we expect: the library exports the
    // two callables, the consumer imports them with body-less prototypes.
    let lib_mod = spirv_linker::load_words(&lib);
    let consumer_mod = spirv_linker::load_words(&consumer);

    let exports: Vec<String> = collect_linkage_names(&lib_mod, LinkageType::Export);
    let imports: Vec<String> = collect_linkage_names(&consumer_mod, LinkageType::Import);
    assert_eq!(
        exports,
        vec!["scale_by".to_string(), "add_offset".to_string()]
    );
    assert_eq!(
        imports,
        vec!["scale_by".to_string(), "add_offset".to_string()]
    );
    assert_eq!(
        consumer_mod
            .functions
            .iter()
            .filter(|f| f.blocks.is_empty())
            .count(),
        2,
        "consumer should have exactly 2 body-less Import prototypes"
    );

    // The actual link.
    let opts = spirv_linker::Options {
        lib: false,
        partial: false,
    };
    let linked_words =
        spirv_linker::link_bytes(&[&consumer, &lib], &opts).expect("link should succeed");
    let linked = spirv_linker::load_words(&linked_words);

    // No `Linkage` capability or `LinkageAttributes` decorations remain.
    assert!(
        !linked.capabilities.iter().any(|c| matches!(
            c.operands.first(),
            Some(Operand::Capability(Capability::Linkage))
        )),
        "linked module still declares OpCapability Linkage"
    );
    assert!(
        !linked.annotations.iter().any(|i| matches!(
            i.operands.get(1),
            Some(Operand::Decoration(Decoration::LinkageAttributes))
        )),
        "linked module still has LinkageAttributes decorations"
    );

    // Every reachable function must now have a body.
    for f in &linked.functions {
        assert!(
            !f.blocks.is_empty(),
            "linked module still has body-less function (id %{})",
            f.def.as_ref().and_then(|d| d.result_id).unwrap_or(0)
        );
    }

    // Entry point survives.
    assert!(
        !linked.entry_points.is_empty(),
        "linked module has no OpEntryPoint"
    );

    // The previously-imported names resolve to functions with bodies.
    for &expected in &["scale_by", "add_offset"] {
        let resolved = linked.functions.iter().any(|f| {
            let Some(id) = f.def.as_ref().and_then(|d| d.result_id) else {
                return false;
            };
            name_of(&linked, id) == Some(expected) && !f.blocks.is_empty()
        });
        assert!(
            resolved,
            "expected callable `{expected}` to be linked in with a body"
        );
    }
}

fn collect_linkage_names(module: &Module, want: LinkageType) -> Vec<String> {
    let mut out = Vec::new();
    for inst in &module.annotations {
        if inst.class.opcode != Op::Decorate {
            continue;
        }
        if !matches!(
            inst.operands.get(1),
            Some(Operand::Decoration(Decoration::LinkageAttributes))
        ) {
            continue;
        }
        let name = match inst.operands.get(2) {
            Some(Operand::LiteralString(s)) => s.clone(),
            _ => continue,
        };
        let kind = match inst.operands.get(3) {
            Some(Operand::LinkageType(lt)) => *lt,
            _ => continue,
        };
        if kind == want {
            out.push(name);
        }
    }
    out
}
