use rspirv::binary::Consumer;
use rspirv::binary::Disassemble;
use rspirv::spirv;
use std::collections::{HashMap, HashSet};
use thiserror::Error;
use topological_sort::TopologicalSort;

#[derive(Error, Debug, PartialEq)]
pub enum LinkerError {
    #[error("Unresolved symbol {:?}", .0)]
    UnresolvedSymbol(String),
    #[error("Multiple exports found for {:?}", .0)]
    MultipleExports(String),
    #[error("Types mismatch for {:?}, imported with type {:?}, exported with type {:?}", .name, .import_type, .export_type)]
    TypeMismatch {
        name: String,
        import_type: String,
        export_type: String,
    },
    #[error("unknown data store error")]
    Unknown,
}

pub type Result<T> = std::result::Result<T, LinkerError>;

pub fn load(bytes: &[u8]) -> rspirv::dr::Module {
    let mut loader = rspirv::dr::Loader::new();
    rspirv::binary::parse_bytes(&bytes, &mut loader).unwrap();
    let module = loader.module();
    module
}

/// Convenience over [`load`] that parses a slice of SPIR-V words (the
/// natural form of DXC's `-spirv` output).
pub fn load_words(words: &[u32]) -> rspirv::dr::Module {
    let mut loader = rspirv::dr::Loader::new();
    rspirv::binary::parse_words(words, &mut loader).unwrap();
    loader.module()
}

fn shift_ids(module: &mut rspirv::dr::Module, add: u32) {
    module.all_inst_iter_mut().for_each(|inst| {
        if let Some(ref mut result_id) = &mut inst.result_id {
            *result_id += add;
        }

        if let Some(ref mut result_type) = &mut inst.result_type {
            *result_type += add;
        }

        inst.operands.iter_mut().for_each(|op| match op {
            rspirv::dr::Operand::IdMemorySemantics(w)
            | rspirv::dr::Operand::IdScope(w)
            | rspirv::dr::Operand::IdRef(w) => *w += add,
            _ => {}
        })
    });
}

fn replace_all_uses_with(module: &mut rspirv::dr::Module, before: u32, after: u32) {
    module.all_inst_iter_mut().for_each(|inst| {
        if let Some(ref mut result_type) = &mut inst.result_type {
            if *result_type == before {
                *result_type = after;
            }
        }

        inst.operands.iter_mut().for_each(|op| match op {
            rspirv::dr::Operand::IdMemorySemantics(w)
            | rspirv::dr::Operand::IdScope(w)
            | rspirv::dr::Operand::IdRef(w) => {
                if *w == before {
                    *w = after
                }
            }
            _ => {}
        })
    });
}

fn remove_duplicate_capablities(module: &mut rspirv::dr::Module) {
    let mut set = HashSet::new();
    let mut caps = vec![];

    for c in &module.capabilities {
        let keep = match c.operands[0] {
            rspirv::dr::Operand::Capability(cap) => set.insert(cap),
            _ => true,
        };

        if keep {
            caps.push(c.clone());
        }
    }

    module.capabilities = caps;
}

fn remove_duplicate_ext_inst_imports(module: &mut rspirv::dr::Module) {
    use std::collections::hash_map::Entry;

    // Each `OpExtInstImport` defines a result-id that subsequent
    // `OpExtInst` instructions reference as their "set" operand. When
    // we merge multiple input modules that all import the same
    // extended instruction set (e.g. "GLSL.std.450"), id-shifting gave
    // each import its own result-id but they're semantically the same
    // import. Dedup them by name *and* rewrite every subsequent
    // reference to the dropped id to point at the kept one — otherwise
    // the linked module ends up with `OpExtInst` instructions whose
    // set-id no longer references any `OpExtInstImport`, producing a
    // spirv-val error of the form "OpExtInst set Id N does not
    // reference an OpExtInstImport result Id" and (typically) a
    // driver-side access violation when the module is loaded.
    let mut kept_by_name: HashMap<String, u32> = HashMap::new();
    let mut new_imports: Vec<rspirv::dr::Instruction> = Vec::new();
    let mut id_remap: Vec<(u32, u32)> = Vec::new();

    for inst in &module.ext_inst_imports {
        let name = match inst.operands.first() {
            Some(rspirv::dr::Operand::LiteralString(s)) => s.clone(),
            _ => {
                new_imports.push(inst.clone());
                continue;
            }
        };
        // `OpExtInstImport` always carries a result-id; if it doesn't,
        // this module is malformed and there's nothing to dedup against.
        let Some(this_id) = inst.result_id else {
            new_imports.push(inst.clone());
            continue;
        };
        match kept_by_name.entry(name) {
            Entry::Occupied(e) => {
                id_remap.push((this_id, *e.get()));
            }
            Entry::Vacant(e) => {
                e.insert(this_id);
                new_imports.push(inst.clone());
            }
        }
    }

    module.ext_inst_imports = new_imports;
    for (old_id, new_id) in id_remap {
        replace_all_uses_with(module, old_id, new_id);
    }
}

#[cfg(test)]
mod remove_duplicate_types_tests {
    use super::*;
    use rspirv::dr::{Instruction, Module, Operand};
    use rspirv::spirv::Op;

    fn op_type_int(result_id: u32, width: u32, signed: u32) -> Instruction {
        Instruction::new(
            Op::TypeInt,
            None,
            Some(result_id),
            vec![Operand::LiteralBit32(width), Operand::LiteralBit32(signed)],
        )
    }

    /// Three `OpTypeInt 32 0` declarations with different result-ids
    /// should all collapse to one. The previous implementation only
    /// caught two of them: it advanced `continue_from_idx` past the
    /// kept instance after each duplicate, so by the third pass the
    /// dedup HashMap was empty and the third copy was inserted as a
    /// fresh entry rather than recognised as a duplicate of the first.
    /// Vulkan rejects the resulting binary with
    /// "Duplicate non-aggregate type declarations are not allowed".
    #[test]
    fn collapses_more_than_two_duplicate_types() {
        let mut module = Module::new();
        // Three `OpTypeInt 32 0` with different ids.
        module.types_global_values.push(op_type_int(10, 32, 0));
        module.types_global_values.push(op_type_int(20, 32, 0));
        module.types_global_values.push(op_type_int(30, 32, 0));

        let linked = remove_duplicate_types(module);

        let int_count = linked
            .types_global_values
            .iter()
            .filter(|i| i.class.opcode == Op::TypeInt)
            .count();
        assert_eq!(
            int_count, 1,
            "expected the three duplicate OpTypeInt to collapse to one, got {int_count}",
        );
    }

    /// `OpConstant %float 0` and `OpConstant %uint 0` have identical
    /// operands but different result-types, so they must NOT be
    /// collapsed. The previous dedup key was `(opcode, operands)` only,
    /// which silently merged them — and downstream uses (like an
    /// `OpConstantComposite %v3float ...`) ended up referencing a
    /// `uint` constant where a `float` was expected, producing
    /// "OpConstantComposite Constituent <id>'s type does not match
    /// Result Type's vector element type" at SPIR-V validation.
    #[test]
    fn keeps_constants_with_same_value_but_different_types() {
        let float_type_id = 1_u32;
        let uint_type_id = 2_u32;
        let float_zero_id = 10_u32;
        let uint_zero_id = 20_u32;

        let mut module = Module::new();
        module.types_global_values.push(Instruction::new(
            Op::TypeFloat,
            None,
            Some(float_type_id),
            vec![Operand::LiteralBit32(32)],
        ));
        module.types_global_values.push(Instruction::new(
            Op::TypeInt,
            None,
            Some(uint_type_id),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        ));
        module.types_global_values.push(Instruction::new(
            Op::Constant,
            Some(float_type_id),
            Some(float_zero_id),
            vec![Operand::LiteralBit32(0)],
        ));
        module.types_global_values.push(Instruction::new(
            Op::Constant,
            Some(uint_type_id),
            Some(uint_zero_id),
            vec![Operand::LiteralBit32(0)],
        ));

        let linked = remove_duplicate_types(module);

        let constants: Vec<&Instruction> = linked
            .types_global_values
            .iter()
            .filter(|i| i.class.opcode == Op::Constant)
            .collect();
        assert_eq!(
            constants.len(),
            2,
            "constants with same operands but different result-types must not merge"
        );
        // Both type-id-distinct constants survive.
        assert!(
            constants
                .iter()
                .any(|c| c.result_type == Some(float_type_id)),
            "float zero constant got dropped"
        );
        assert!(
            constants
                .iter()
                .any(|c| c.result_type == Some(uint_type_id)),
            "uint zero constant got dropped"
        );
    }
}

#[cfg(test)]
mod dedup_entry_point_interfaces_tests {
    use super::*;
    use rspirv::dr::{Instruction, Module, Operand};
    use rspirv::spirv::{ExecutionModel, Op};

    /// `OpEntryPoint Compute %main "main" %5 %7 %5` (with a duplicate
    /// in the interface list, mimicking the post-dedup state where
    /// two original variables collapsed to the same id) should
    /// collapse to `... %5 %7`. Vulkan rejects the binary otherwise
    /// with "Non-unique OpEntryPoint interface".
    #[test]
    fn drops_duplicate_interface_ids() {
        let mut module = Module::new();
        module.entry_points.push(Instruction::new(
            Op::EntryPoint,
            None,
            None,
            vec![
                Operand::ExecutionModel(ExecutionModel::GLCompute),
                Operand::IdRef(1),
                Operand::LiteralString("main".to_owned()),
                Operand::IdRef(5),
                Operand::IdRef(7),
                Operand::IdRef(5),
            ],
        ));

        dedup_entry_point_interfaces(&mut module);

        let interface: Vec<u32> = module.entry_points[0]
            .operands
            .iter()
            .skip(3)
            .filter_map(|op| {
                if let Operand::IdRef(id) = op {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(interface, vec![5, 7]);
    }
}

#[cfg(test)]
mod remove_duplicate_annotations_tests {
    use super::*;
    use rspirv::dr::{Instruction, Module, Operand};
    use rspirv::spirv::{Decoration, Op};

    /// After ID dedup collapses two `OpVariable`s onto the same id, every
    /// `OpDecorate` that previously decorated either of them now decorates
    /// the kept id — producing two identical `OpDecorate %5 DescriptorSet 0`
    /// (and similar) entries. spirv-val rejects this with "ID 'N' decorated
    /// with DescriptorSet multiple times is not allowed". Dedup must collapse
    /// fully-identical annotation instructions.
    #[test]
    fn collapses_identical_op_decorate_pairs() {
        let mut module = Module::new();
        let target = 5_u32;
        let make_decorate_descriptor_set = || {
            Instruction::new(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(target),
                    Operand::Decoration(Decoration::DescriptorSet),
                    Operand::LiteralBit32(0),
                ],
            )
        };
        let make_decorate_binding = || {
            Instruction::new(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(target),
                    Operand::Decoration(Decoration::Binding),
                    Operand::LiteralBit32(7),
                ],
            )
        };
        // Two duplicate DescriptorSet decorations and two duplicate Binding
        // decorations — exactly what ID dedup leaves behind.
        module.annotations.push(make_decorate_descriptor_set());
        module.annotations.push(make_decorate_binding());
        module.annotations.push(make_decorate_descriptor_set());
        module.annotations.push(make_decorate_binding());

        // A non-duplicate decoration on a different id should be preserved.
        module.annotations.push(Instruction::new(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(9),
                Operand::Decoration(Decoration::DescriptorSet),
                Operand::LiteralBit32(0),
            ],
        ));

        remove_duplicate_annotations(&mut module);

        assert_eq!(
            module.annotations.len(),
            3,
            "expected 4 duplicates to collapse to 2, plus the unrelated one to remain"
        );
        let descriptor_set_count = module
            .annotations
            .iter()
            .filter(|i| {
                i.class.opcode == Op::Decorate
                    && i.operands.first() == Some(&Operand::IdRef(target))
                    && i.operands.get(1) == Some(&Operand::Decoration(Decoration::DescriptorSet))
            })
            .count();
        assert_eq!(descriptor_set_count, 1);
        let binding_count = module
            .annotations
            .iter()
            .filter(|i| {
                i.class.opcode == Op::Decorate
                    && i.operands.first() == Some(&Operand::IdRef(target))
                    && i.operands.get(1) == Some(&Operand::Decoration(Decoration::Binding))
            })
            .count();
        assert_eq!(binding_count, 1);
    }
}

#[cfg(test)]
mod ext_inst_imports_dedup_tests {
    use super::*;
    use rspirv::dr::{Instruction, Module, Operand};
    use rspirv::spirv::Op;

    /// Construct a tiny module with two `OpExtInstImport "GLSL.std.450"`
    /// declarations (mimicking the post-id-shift state of two merged
    /// inputs) plus a single `OpExtInst` that references the second
    /// (later) import. The dedup pass must keep the first import and
    /// rewrite the `OpExtInst` set-id to point at it.
    #[test]
    fn rewrites_op_ext_inst_set_id_when_dropping_duplicate() {
        let kept_id = 5_u32;
        let dropped_id = 47_u32;

        let import_glsl_first = Instruction::new(
            Op::ExtInstImport,
            None,
            Some(kept_id),
            vec![Operand::LiteralString("GLSL.std.450".to_owned())],
        );
        let import_glsl_second = Instruction::new(
            Op::ExtInstImport,
            None,
            Some(dropped_id),
            vec![Operand::LiteralString("GLSL.std.450".to_owned())],
        );

        // OpExtInst is permitted inside a function body. Build a minimal
        // function with one block holding the call we want to verify.
        let result_type = 100_u32;
        let result_id = 101_u32;
        let arg_id = 102_u32;
        // GLSL.std.450 Sqrt = 31
        let glsl_sqrt = 31_u32;

        let ext_inst_call = Instruction::new(
            Op::ExtInst,
            Some(result_type),
            Some(result_id),
            vec![
                Operand::IdRef(dropped_id),
                Operand::LiteralBit32(glsl_sqrt),
                Operand::IdRef(arg_id),
            ],
        );

        let block = rspirv::dr::Block {
            label: Some(Instruction::new(Op::Label, None, Some(200), vec![])),
            instructions: vec![ext_inst_call],
        };
        let func_def = Instruction::new(
            Op::Function,
            Some(result_type),
            Some(300),
            vec![
                Operand::FunctionControl(rspirv::spirv::FunctionControl::NONE),
                Operand::IdRef(result_type),
            ],
        );
        let func_end = Instruction::new(Op::FunctionEnd, None, None, vec![]);

        let func = rspirv::dr::Function {
            def: Some(func_def),
            end: Some(func_end),
            parameters: vec![],
            blocks: vec![block],
        };

        let mut module = Module::new();
        module.ext_inst_imports.push(import_glsl_first);
        module.ext_inst_imports.push(import_glsl_second);
        module.functions.push(func);

        remove_duplicate_ext_inst_imports(&mut module);

        // Only the first import survives.
        assert_eq!(module.ext_inst_imports.len(), 1);
        assert_eq!(module.ext_inst_imports[0].result_id, Some(kept_id));

        // The OpExtInst's set-id must have been rewritten to point at
        // the surviving import — that's what the bug fix is for.
        let call = &module.functions[0].blocks[0].instructions[0];
        assert_eq!(call.class.opcode, Op::ExtInst);
        match &call.operands[0] {
            Operand::IdRef(id) => assert_eq!(
                *id, kept_id,
                "OpExtInst set-id should have been rewritten from \
                 dropped {dropped_id} to kept {kept_id}, got {id}"
            ),
            other => panic!("expected IdRef, got {other:?}"),
        }
    }
}

fn kill_with_id(insts: &mut Vec<rspirv::dr::Instruction>, id: u32) {
    kill_with(insts, |inst| {
        if inst.operands.is_empty() {
            return false;
        }

        match inst.operands[0] {
            rspirv::dr::Operand::IdMemorySemantics(w)
            | rspirv::dr::Operand::IdScope(w)
            | rspirv::dr::Operand::IdRef(w)
                if w == id =>
            {
                true
            }
            _ => false,
        }
    })
}

fn kill_with<F>(insts: &mut Vec<rspirv::dr::Instruction>, f: F)
where
    F: Fn(&rspirv::dr::Instruction) -> bool,
{
    if insts.is_empty() {
        return;
    }

    let mut idx = insts.len() - 1;
    // odd backwards loop so we can swap_remove
    loop {
        if f(&insts[idx]) {
            insts.swap_remove(idx);
        }

        if idx == 0 || insts.is_empty() {
            break;
        }

        idx -= 1;
    }
}

fn kill_annotations_and_debug(module: &mut rspirv::dr::Module, id: u32) {
    kill_with_id(&mut module.annotations, id);

    // need to remove OpGroupDecorate members that mention this id
    module.annotations.iter_mut().for_each(|inst| {
        if inst.class.opcode == spirv::Op::GroupDecorate {
            inst.operands.retain(|op| match op {
                rspirv::dr::Operand::IdRef(w) if *w != id => return true,
                _ => return false,
            });
        }
    });

    kill_with_id(&mut module.debug_string_source, id);
    kill_with_id(&mut module.debug_names, id);
    kill_with_id(&mut module.debug_module_processed, id);
}

fn remove_duplicate_types(module: rspirv::dr::Module) -> rspirv::dr::Module {
    use rspirv::binary::Assemble;

    // jb-todo: spirv-tools's linker has special case handling for SpvOpTypeForwardPointer,
    // not sure if we need that; see https://github.com/KhronosGroup/SPIRV-Tools/blob/e7866de4b1dc2a7e8672867caeb0bdca49f458d3/source/opt/remove_duplicates_pass.cpp for reference

    let mut instructions = module
        .all_inst_iter()
        .cloned()
        .collect::<Vec<_>>()
        .into_boxed_slice(); // force boxed slice so we don't accidentally grow or shrink it later

    let mut def_use_analyzer = DefUseAnalyzer::new(&mut instructions);

    let mut kill_annotations = vec![];

    // Iterative because types can reference each other: after merging
    // `OpTypeInt %A`/`%B`, two `OpTypePointer`s that were pointing at
    // each become identical and themselves become duplicates. We
    // restart from index 0 with a fresh dedup map every pass — an
    // earlier optimisation tried to resume from the latest backtrack
    // point but ended up advancing past kept instances and silently
    // missing duplicate groups in the tail of the array.
    loop {
        let mut dedup = std::collections::HashMap::new();
        let mut duplicate = None;

        for module_inst in module.types_global_values.iter() {
            // Some `types_global_values` entries don't carry a
            // `result_id` — most relevantly `OpTypeForwardPointer`,
            // which DXC can emit when targeting `universal1.5`. We
            // can't dedup these by-name; just skip them.
            let Some(result_id) = module_inst.result_id else {
                continue;
            };
            let (inst_idx, inst) = def_use_analyzer.def(result_id);

            if inst.class.opcode == spirv::Op::Nop {
                continue;
            }

            // Partial assembly used as a dedup key: opcode, result-type
            // (if any), and operands. The result-type is essential —
            // `OpConstant %float 0` and `OpConstant %uint 0` have
            // identical operands but are different constants, and
            // collapsing them produced an `OpConstantComposite %v3float
            // %uint_0 %uint_0 %uint_0` type-mismatch downstream.
            let data = {
                let mut data = vec![];

                data.push(inst.class.opcode as u32);
                // Sentinel-prefixed result-type so a None / Some(0) on
                // typed instructions can't be confused for a missing
                // type on type-defining ones (which never have one).
                match inst.result_type {
                    Some(t) => {
                        data.push(0xffff_ffff);
                        data.push(t);
                    }
                    None => {
                        data.push(0xffff_fffe);
                    }
                }
                for op in &inst.operands {
                    op.assemble_into(&mut data);
                }

                data
            };

            dedup
                .entry(data)
                .and_modify(|identical_idx| {
                    duplicate = Some((inst_idx, *identical_idx));
                })
                .or_insert(inst_idx);

            if duplicate.is_some() {
                break;
            }
        }

        if let Some((before_idx, after_idx)) = duplicate {
            let before_id = def_use_analyzer.instructions[before_idx].result_id.unwrap();
            let after_id = def_use_analyzer.instructions[after_idx].result_id.unwrap();

            // remove annotations later
            kill_annotations.push(before_id);

            def_use_analyzer.for_each_use(before_id, |inst| {
                if inst.result_type == Some(before_id) {
                    inst.result_type = Some(after_id);
                }

                for op in inst.operands.iter_mut() {
                    match op {
                        rspirv::dr::Operand::IdMemorySemantics(w)
                        | rspirv::dr::Operand::IdScope(w)
                        | rspirv::dr::Operand::IdRef(w) => {
                            if *w == before_id {
                                *w = after_id
                            }
                        }
                        _ => {}
                    }
                }
            });

            // this loop / system works on the assumption that all indices remain valid,
            // so instead of removing the instruction we just nop it out - `consume_instruction` will then
            // skip all OpNops and they won't appear in the newly constructed module
            def_use_analyzer.instructions[before_idx] =
                rspirv::dr::Instruction::new(spirv::Op::Nop, None, None, vec![]);
        } else {
            break;
        }
    }

    let mut loader = rspirv::dr::Loader::new();

    for inst in def_use_analyzer.instructions.iter() {
        loader.consume_instruction(inst.clone());
    }

    let mut module = loader.module();

    for remove in kill_annotations {
        kill_annotations_and_debug(&mut module, remove);
    }

    module
}

#[derive(Clone, Debug)]
struct LinkSymbol {
    name: String,
    id: u32,
    type_id: u32,
    parameters: Vec<rspirv::dr::Instruction>,
}

#[derive(Debug)]
struct ImportExportPair {
    import: LinkSymbol,
    export: LinkSymbol,
}

#[derive(Debug)]
struct LinkInfo {
    imports: Vec<LinkSymbol>,
    exports: HashMap<String, Vec<LinkSymbol>>,
    potential_pairs: Vec<ImportExportPair>,
}

fn inst_fully_eq(a: &rspirv::dr::Instruction, b: &rspirv::dr::Instruction) -> bool {
    // both function instructions need to be 100% identical so check all members
    // jb-todo: derive(PartialEq) on Instruction?
    a.result_id == b.result_id
        && a.class == b.class
        && a.result_type == b.result_type
        && a.operands == b.operands
}

fn find_import_export_pairs(module: &rspirv::dr::Module, defs: &DefAnalyzer) -> Result<LinkInfo> {
    let mut imports = vec![];
    let mut exports: HashMap<String, Vec<LinkSymbol>> = HashMap::new();

    for annotation in &module.annotations {
        if annotation.class.opcode == spirv::Op::Decorate
            && annotation.operands[1]
                == rspirv::dr::Operand::Decoration(spirv::Decoration::LinkageAttributes)
        {
            let id = match annotation.operands[0] {
                rspirv::dr::Operand::IdRef(i) => i,
                _ => panic!("Expected IdRef"),
            };

            let name = match &annotation.operands[2] {
                rspirv::dr::Operand::LiteralString(s) => s,
                _ => panic!("Expected LiteralString"),
            };

            let ty = &annotation.operands[3];

            let def_inst = defs
                .def(id)
                .expect(&format!("Need a matching op for ID {}", id));

            let (type_id, parameters) = match def_inst.class.opcode {
                spirv::Op::Variable => (def_inst.result_type.unwrap(), vec![]),
                spirv::Op::Function => {
                    let type_id = if let rspirv::dr::Operand::IdRef(id) = &def_inst.operands[1] {
                        *id
                    } else {
                        panic!("Expected IdRef");
                    };

                    let def_fn = module
                        .functions
                        .iter()
                        .find(|f| inst_fully_eq(f.def.as_ref().unwrap(), def_inst))
                        .unwrap();

                    (type_id, def_fn.parameters.clone())
                }
                _ => panic!("Unexpected op"),
            };

            let symbol = LinkSymbol {
                name: name.to_string(),
                id,
                type_id,
                parameters,
            };

            if ty == &rspirv::dr::Operand::LinkageType(spirv::LinkageType::Import) {
                imports.push(symbol);
            } else {
                exports
                    .entry(symbol.name.clone())
                    .and_modify(|v| v.push(symbol.clone()))
                    .or_insert_with(|| vec![symbol.clone()]);
            }
        }
    }

    LinkInfo {
        imports,
        exports,
        potential_pairs: vec![],
    }
    .find_potential_pairs()
}

fn cleanup_type(mut ty: rspirv::dr::Instruction) -> String {
    ty.result_id = None;
    ty.disassemble()
}

impl LinkInfo {
    fn find_potential_pairs(mut self) -> Result<Self> {
        for import in &self.imports {
            let potential_matching_exports = self.exports.get(&import.name);
            if let Some(potential_matching_exports) = potential_matching_exports {
                if potential_matching_exports.len() > 1 {
                    return Err(LinkerError::MultipleExports(import.name.clone()));
                }

                self.potential_pairs.push(ImportExportPair {
                    import: import.clone(),
                    export: potential_matching_exports.first().unwrap().clone(),
                });
            } else {
                return Err(LinkerError::UnresolvedSymbol(import.name.clone()));
            }
        }

        Ok(self)
    }

    /// returns the list of matching import / export pairs after validation the list of potential pairs
    fn ensure_matching_import_export_pairs(
        &self,
        defs: &DefAnalyzer,
    ) -> Result<&Vec<ImportExportPair>> {
        for pair in &self.potential_pairs {
            let import_result_type = defs.def(pair.import.type_id).unwrap();
            let export_result_type = defs.def(pair.export.type_id).unwrap();

            let imp = trans_aggregate_type(defs, import_result_type);
            let exp = trans_aggregate_type(defs, export_result_type);

            if imp != exp {
                return Err(LinkerError::TypeMismatch {
                    name: pair.import.name.clone(),
                    import_type: cleanup_type(import_result_type.clone()),
                    export_type: cleanup_type(export_result_type.clone()),
                });
            }

            for (import_param, export_param) in pair
                .import
                .parameters
                .iter()
                .zip(pair.export.parameters.iter())
            {
                if !import_param.is_type_identical(export_param) {
                    panic!("Type error in signatures")
                }

                // jb-todo: validate that OpDecoration is identical too
            }
        }

        Ok(&self.potential_pairs)
    }
}

struct DefAnalyzer {
    def_ids: HashMap<u32, rspirv::dr::Instruction>,
}

impl DefAnalyzer {
    fn new(module: &rspirv::dr::Module) -> Self {
        let mut def_ids = HashMap::new();

        module.all_inst_iter().for_each(|inst| {
            if let Some(def_id) = inst.result_id {
                def_ids
                    .entry(def_id)
                    .and_modify(|stored_inst| {
                        *stored_inst = inst.clone();
                    })
                    .or_insert(inst.clone());
            }
        });

        Self { def_ids }
    }

    fn def(&self, id: u32) -> Option<&rspirv::dr::Instruction> {
        self.def_ids.get(&id)
    }
}

struct DefUseAnalyzer<'a> {
    def_ids: HashMap<u32, usize>,
    use_ids: HashMap<u32, Vec<usize>>,
    use_result_type_ids: HashMap<u32, Vec<usize>>,
    instructions: &'a mut [rspirv::dr::Instruction],
}

impl<'a> DefUseAnalyzer<'a> {
    fn new(instructions: &'a mut [rspirv::dr::Instruction]) -> Self {
        let mut def_ids = HashMap::new();
        let mut use_ids: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut use_result_type_ids: HashMap<u32, Vec<usize>> = HashMap::new();

        instructions
            .iter()
            .enumerate()
            .for_each(|(inst_idx, inst)| {
                if let Some(def_id) = inst.result_id {
                    def_ids
                        .entry(def_id)
                        .and_modify(|stored_inst| {
                            *stored_inst = inst_idx;
                        })
                        .or_insert(inst_idx);
                }

                if let Some(result_type) = inst.result_type {
                    use_result_type_ids
                        .entry(result_type)
                        .and_modify(|v| v.push(inst_idx))
                        .or_insert(vec![inst_idx]);
                }

                for op in inst.operands.iter() {
                    match op {
                        rspirv::dr::Operand::IdMemorySemantics(w)
                        | rspirv::dr::Operand::IdScope(w)
                        | rspirv::dr::Operand::IdRef(w) => {
                            use_ids
                                .entry(*w)
                                .and_modify(|v| v.push(inst_idx))
                                .or_insert(vec![inst_idx]);
                        }
                        _ => {}
                    }
                }
            });

        Self {
            def_ids,
            use_ids,
            use_result_type_ids,
            instructions,
        }
    }

    fn def_idx(&self, id: u32) -> usize {
        self.def_ids[&id]
    }

    fn def(&self, id: u32) -> (usize, &rspirv::dr::Instruction) {
        let idx = self.def_idx(id);
        (idx, &self.instructions[idx])
    }

    fn for_each_use<F>(&mut self, id: u32, mut f: F)
    where
        F: FnMut(&mut rspirv::dr::Instruction),
    {
        // find by `result_type`
        if let Some(use_result_type_id) = self.use_result_type_ids.get(&id) {
            for inst_idx in use_result_type_id {
                f(&mut self.instructions[*inst_idx])
            }
        }

        // find by operand
        if let Some(use_id) = self.use_ids.get(&id) {
            for inst_idx in use_id {
                f(&mut self.instructions[*inst_idx]);
            }
        }
    }
}

fn import_kill_annotations_and_debug(module: &mut rspirv::dr::Module, info: &LinkInfo) {
    for import in &info.imports {
        kill_annotations_and_debug(module, import.id);
        for param in &import.parameters {
            kill_annotations_and_debug(module, param.result_id.unwrap())
        }
    }
}

pub struct Options {
    /// `true` if we're creating a library
    pub lib: bool,

    /// `true` if partial linking is allowed
    pub partial: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            lib: false,
            partial: false,
        }
    }
}

fn kill_linkage_instructions(
    pairs: &Vec<ImportExportPair>,
    module: &mut rspirv::dr::Module,
    opts: &Options,
) {
    // drop imported functions
    for pair in pairs.iter() {
        module
            .functions
            .retain(|f| pair.import.id != f.def.as_ref().unwrap().result_id.unwrap());
    }

    // drop imported variables
    for pair in pairs.iter() {
        module
            .types_global_values
            .retain(|v| pair.import.id != v.result_id.unwrap());
    }

    // drop linkage attributes (both import and export)
    kill_with(&mut module.annotations, |inst| {
        let eq = pairs
            .iter()
            .find(|p| {
                if inst.operands.is_empty() {
                    return false;
                }

                if let rspirv::dr::Operand::IdRef(id) = inst.operands[0] {
                    id == p.import.id || id == p.export.id
                } else {
                    false
                }
            })
            .is_some();

        eq && inst.class.opcode == spirv::Op::Decorate
            && inst.operands[1]
                == rspirv::dr::Operand::Decoration(spirv::Decoration::LinkageAttributes)
    });

    if !opts.lib {
        kill_with(&mut module.annotations, |inst| {
            inst.class.opcode == spirv::Op::Decorate
                && inst.operands[1]
                    == rspirv::dr::Operand::Decoration(spirv::Decoration::LinkageAttributes)
                && inst.operands[3] == rspirv::dr::Operand::LinkageType(spirv::LinkageType::Export)
        });
    }

    // drop OpCapability Linkage
    kill_with(&mut module.capabilities, |inst| {
        inst.class.opcode == spirv::Op::Capability
            && inst.operands[0] == rspirv::dr::Operand::Capability(spirv::Capability::Linkage)
    })
}

fn compact_ids(module: &mut rspirv::dr::Module) -> u32 {
    let mut remap = HashMap::new();

    let mut insert = |current_id: u32| -> u32 {
        if remap.contains_key(&current_id) {
            remap[&current_id]
        } else {
            let new_id = remap.len() as u32 + 1;
            remap.insert(current_id, new_id);
            new_id
        }
    };

    module.all_inst_iter_mut().for_each(|inst| {
        if let Some(ref mut result_id) = &mut inst.result_id {
            *result_id = insert(*result_id);
        }

        if let Some(ref mut result_type) = &mut inst.result_type {
            *result_type = insert(*result_type);
        }

        inst.operands.iter_mut().for_each(|op| match op {
            rspirv::dr::Operand::IdMemorySemantics(w)
            | rspirv::dr::Operand::IdScope(w)
            | rspirv::dr::Operand::IdRef(w) => {
                *w = insert(*w);
            }
            _ => {}
        })
    });

    remap.len() as u32 + 1
}

/// Remove fully-identical duplicate instructions from
/// `module.annotations`. After merging input modules, dedup of types /
/// variables collapses ids — and the `OpDecorate` / `OpMemberDecorate`
/// instructions that previously decorated the dropped ids get rewritten
/// to point at the kept ones, leaving the kept ids decorated multiple
/// times with the same decoration. spirv-val rejects e.g. "ID 'N'
/// decorated with DescriptorSet multiple times is not allowed".
fn remove_duplicate_annotations(module: &mut rspirv::dr::Module) {
    use rspirv::binary::Assemble;
    let mut seen = HashSet::new();
    let original = std::mem::take(&mut module.annotations);
    module.annotations.reserve(original.len());
    for inst in original {
        let mut key = vec![inst.class.opcode as u32];
        for op in &inst.operands {
            op.assemble_into(&mut key);
        }
        if seen.insert(key) {
            module.annotations.push(inst);
        }
    }
}

/// `OpEntryPoint`'s interface list (the IdRef operands after the
/// execution model, function id, and name) names every `Input` /
/// `Output` / `StorageBuffer` / `Workgroup` / `Private` global
/// variable the entry point uses. After we merge multiple input
/// modules, dedup of `OpVariable`s collapses two names that pointed at
/// what's now the same id — but the entry point's interface list
/// still mentions both of them, so the merged id appears more than
/// once. spirv-val rejects the binary with "Non-unique OpEntryPoint
/// interface 'N[%N]' is disallowed".
///
/// Walk every `OpEntryPoint`, drop duplicate IdRefs in the trailing
/// interface section while preserving order.
fn dedup_entry_point_interfaces(module: &mut rspirv::dr::Module) {
    for inst in module.entry_points.iter_mut() {
        if inst.class.opcode != spirv::Op::EntryPoint {
            continue;
        }
        // Operands 0..=2 are the ExecutionModel, function id, and
        // name; everything from index 3 on is the interface list.
        const INTERFACE_START: usize = 3;
        if inst.operands.len() <= INTERFACE_START {
            continue;
        }

        let head: Vec<_> = inst.operands.drain(..INTERFACE_START).collect();
        let mut seen = HashSet::new();
        let mut deduped: Vec<rspirv::dr::Operand> = Vec::new();
        for op in inst.operands.drain(..) {
            match op {
                rspirv::dr::Operand::IdRef(id) => {
                    if seen.insert(id) {
                        deduped.push(rspirv::dr::Operand::IdRef(id));
                    }
                }
                other => deduped.push(other),
            }
        }
        inst.operands = head;
        inst.operands.extend(deduped);
    }
}

fn sort_globals(module: &mut rspirv::dr::Module) {
    let mut ts = TopologicalSort::<u32>::new();

    // Instructions with no `result_id` (e.g. `OpTypeForwardPointer`) can't
    // participate in the by-id topological sort. Preserve them verbatim
    // up-front; per SPIR-V spec `OpTypeForwardPointer` precedes the
    // corresponding `OpTypePointer` declaration anyway.
    let mut new_types_global_values: Vec<_> = module
        .types_global_values
        .iter()
        .filter(|t| t.result_id.is_none())
        .cloned()
        .collect();

    for t in module.types_global_values.iter() {
        if let Some(result_id) = t.result_id {
            if let Some(result_type) = t.result_type {
                ts.add_dependency(result_type, result_id);
            }

            for op in &t.operands {
                match op {
                    rspirv::dr::Operand::IdMemorySemantics(w)
                    | rspirv::dr::Operand::IdScope(w)
                    | rspirv::dr::Operand::IdRef(w) => {
                        ts.add_dependency(*w, result_id); // the op defining the IdRef should come before our op / result_id
                    }
                    _ => {}
                }
            }
        }
    }

    let defs = DefAnalyzer::new(&module);

    loop {
        if ts.is_empty() {
            break;
        }

        let mut v = ts.pop_all();
        v.sort();

        for result_id in v {
            new_types_global_values.push(defs.def(result_id).unwrap().clone());
        }
    }

    if module.types_global_values.len() != new_types_global_values.len() {
        eprintln!(
            "spirv-linker::sort_globals: input had {} types/globals, output has {} \
             (likely lost a few stragglers; falling back to original order)",
            module.types_global_values.len(),
            new_types_global_values.len()
        );
    } else {
        module.types_global_values = new_types_global_values;
    }
}

#[derive(PartialEq, Debug)]
enum ScalarType {
    Void,
    Bool,
    Int { width: u32, signed: bool },
    Float { width: u32 },
    Opaque { name: String },
    Event,
    DeviceEvent,
    ReserveId,
    Queue,
    Pipe,
    ForwardPointer { storage_class: spirv::StorageClass },
    PipeStorage,
    NamedBarrier,
    Sampler,
}

fn trans_scalar_type(inst: &rspirv::dr::Instruction) -> Option<ScalarType> {
    Some(match inst.class.opcode {
        spirv::Op::TypeVoid => ScalarType::Void,
        spirv::Op::TypeBool => ScalarType::Bool,
        spirv::Op::TypeEvent => ScalarType::Event,
        spirv::Op::TypeDeviceEvent => ScalarType::DeviceEvent,
        spirv::Op::TypeReserveId => ScalarType::ReserveId,
        spirv::Op::TypeQueue => ScalarType::Queue,
        spirv::Op::TypePipe => ScalarType::Pipe,
        spirv::Op::TypePipeStorage => ScalarType::PipeStorage,
        spirv::Op::TypeNamedBarrier => ScalarType::NamedBarrier,
        spirv::Op::TypeSampler => ScalarType::Sampler,
        spirv::Op::TypeForwardPointer => ScalarType::ForwardPointer {
            storage_class: match inst.operands[0] {
                rspirv::dr::Operand::StorageClass(s) => s,
                _ => panic!("Unexpected operand while parsing type"),
            },
        },
        spirv::Op::TypeInt => ScalarType::Int {
            width: match inst.operands[0] {
                rspirv::dr::Operand::LiteralBit32(w) => w,
                _ => panic!("Unexpected operand while parsing type"),
            },
            signed: match inst.operands[1] {
                rspirv::dr::Operand::LiteralBit32(s) => {
                    if s == 0 {
                        false
                    } else {
                        true
                    }
                }
                _ => panic!("Unexpected operand while parsing type"),
            },
        },
        spirv::Op::TypeFloat => ScalarType::Float {
            width: match inst.operands[0] {
                rspirv::dr::Operand::LiteralBit32(w) => w,
                _ => panic!("Unexpected operand while parsing type"),
            },
        },
        spirv::Op::TypeOpaque => ScalarType::Opaque {
            name: match &inst.operands[0] {
                rspirv::dr::Operand::LiteralString(s) => s.clone(),
                _ => panic!("Unexpected operand while parsing type"),
            },
        },
        _ => return None,
    })
}

#[derive(PartialEq, Debug)]
enum AggregateType {
    Scalar(ScalarType),
    Array {
        ty: Box<AggregateType>,
        len: u64,
    },
    Pointer {
        ty: Box<AggregateType>,
        storage_class: spirv::StorageClass,
    },
    Image {
        ty: Box<AggregateType>,
        dim: spirv::Dim,
        depth: u32,
        arrayed: u32,
        multi_sampled: u32,
        sampled: u32,
        format: spirv::ImageFormat,
        access: Option<spirv::AccessQualifier>,
    },
    SampledImage {
        ty: Box<AggregateType>,
    },
    Aggregate(Vec<AggregateType>),
}

fn op_def(def: &DefAnalyzer, operand: &rspirv::dr::Operand) -> rspirv::dr::Instruction {
    def.def(match operand {
        rspirv::dr::Operand::IdMemorySemantics(w)
        | rspirv::dr::Operand::IdScope(w)
        | rspirv::dr::Operand::IdRef(w) => *w,
        _ => panic!("Expected ID"),
    })
    .unwrap()
    .clone()
}

fn extract_literal_int_as_u64(op: &rspirv::dr::Operand) -> u64 {
    match op {
        rspirv::dr::Operand::LiteralBit32(v) => (*v).into(),
        rspirv::dr::Operand::LiteralBit64(v) => *v,
        _ => panic!("Unexpected literal int"),
    }
}

fn extract_literal_u32(op: &rspirv::dr::Operand) -> u32 {
    match op {
        rspirv::dr::Operand::LiteralBit32(v) => *v,
        _ => panic!("Unexpected literal u32"),
    }
}

fn trans_aggregate_type(
    def: &DefAnalyzer,
    inst: &rspirv::dr::Instruction,
) -> Option<AggregateType> {
    Some(match inst.class.opcode {
        spirv::Op::TypeArray => {
            let len_def = op_def(def, &inst.operands[1]);
            assert!(len_def.class.opcode == spirv::Op::Constant); // don't support spec constants yet

            let len_value = extract_literal_int_as_u64(&len_def.operands[1]);

            AggregateType::Array {
                ty: Box::new(
                    trans_aggregate_type(def, &op_def(def, &inst.operands[0]))
                        .expect("Expect base type for OpTypeArray"),
                ),
                len: len_value,
            }
        }
        spirv::Op::TypePointer => AggregateType::Pointer {
            storage_class: match inst.operands[0] {
                rspirv::dr::Operand::StorageClass(s) => s,
                _ => panic!("Unexpected operand while parsing type"),
            },
            ty: Box::new(
                trans_aggregate_type(def, &op_def(def, &inst.operands[1]))
                    .expect("Expect base type for OpTypePointer"),
            ),
        },
        spirv::Op::TypeRuntimeArray
        | spirv::Op::TypeVector
        | spirv::Op::TypeMatrix
        | spirv::Op::TypeSampledImage => AggregateType::Aggregate(
            trans_aggregate_type(def, &op_def(def, &inst.operands[0]))
                .map_or_else(|| vec![], |v| vec![v]),
        ),
        spirv::Op::TypeStruct | spirv::Op::TypeFunction => {
            let mut types = vec![];
            for operand in inst.operands.iter() {
                let op_def = op_def(def, operand);

                match trans_aggregate_type(def, &op_def) {
                    Some(ty) => types.push(ty),
                    None => panic!("Expected type"),
                }
            }

            AggregateType::Aggregate(types)
        }
        spirv::Op::TypeImage => AggregateType::Image {
            ty: Box::new(
                trans_aggregate_type(def, &op_def(def, &inst.operands[0]))
                    .expect("Expect base type for OpTypeImage"),
            ),
            dim: match inst.operands[1] {
                rspirv::dr::Operand::Dim(d) => d,
                _ => panic!("Invalid dim"),
            },
            depth: extract_literal_u32(&inst.operands[2]),
            arrayed: extract_literal_u32(&inst.operands[3]),
            multi_sampled: extract_literal_u32(&inst.operands[4]),
            sampled: extract_literal_u32(&inst.operands[5]),
            format: match inst.operands[6] {
                rspirv::dr::Operand::ImageFormat(f) => f,
                _ => panic!("Invalid image format"),
            },
            access: inst
                .operands
                .get(7)
                .map(|op| match op {
                    rspirv::dr::Operand::AccessQualifier(a) => Some(a.clone()),
                    _ => None,
                })
                .flatten(),
        },
        _ => {
            if let Some(ty) = trans_scalar_type(inst) {
                AggregateType::Scalar(ty)
            } else {
                return None;
            }
        }
    })
}

pub fn link(inputs: &mut [&mut rspirv::dr::Module], opts: &Options) -> Result<rspirv::dr::Module> {
    // shift all the ids
    let mut bound = inputs[0].header.as_ref().unwrap().bound - 1;

    for mut module in inputs.iter_mut().skip(1) {
        shift_ids(&mut module, bound);
        bound += module.header.as_ref().unwrap().bound - 1;
    }

    // merge the binaries
    let mut loader = rspirv::dr::Loader::new();

    for module in inputs.iter() {
        module.all_inst_iter().for_each(|inst| {
            loader.consume_instruction(inst.clone());
        });
    }

    let mut output = loader.module();

    // find import / export pairs
    let defs = DefAnalyzer::new(&output);
    let info = find_import_export_pairs(&output, &defs)?;

    // ensure import / export pairs have matching types and defintions
    let matching_pairs = info.ensure_matching_import_export_pairs(&defs)?;

    // remove duplicates (https://github.com/KhronosGroup/SPIRV-Tools/blob/e7866de4b1dc2a7e8672867caeb0bdca49f458d3/source/opt/remove_duplicates_pass.cpp)
    remove_duplicate_capablities(&mut output);
    remove_duplicate_ext_inst_imports(&mut output);
    let mut output = remove_duplicate_types(output);
    // jb-todo: strip identical OpDecoration / OpDecorationGroups

    // remove names and decorations of import variables / functions https://github.com/KhronosGroup/SPIRV-Tools/blob/8a0ebd40f86d1f18ad42ea96c6ac53915076c3c7/source/opt/ir_context.cpp#L404
    import_kill_annotations_and_debug(&mut output, &info);

    // rematch import variables and functions to export variables / functions https://github.com/KhronosGroup/SPIRV-Tools/blob/8a0ebd40f86d1f18ad42ea96c6ac53915076c3c7/source/opt/ir_context.cpp#L255
    for pair in matching_pairs {
        replace_all_uses_with(&mut output, pair.import.id, pair.export.id);
    }

    // remove linkage specific instructions
    kill_linkage_instructions(&matching_pairs, &mut output, &opts);

    dedup_entry_point_interfaces(&mut output);
    remove_duplicate_annotations(&mut output);

    sort_globals(&mut output);

    // compact the ids https://github.com/KhronosGroup/SPIRV-Tools/blob/e02f178a716b0c3c803ce31b9df4088596537872/source/opt/compact_ids_pass.cpp#L43
    let bound = compact_ids(&mut output);
    output.header = Some(rspirv::dr::ModuleHeader::new(bound));

    output
        .debug_module_processed
        .push(rspirv::dr::Instruction::new(
            spirv::Op::ModuleProcessed,
            None,
            None,
            vec![rspirv::dr::Operand::LiteralString(
                "Linked by rspirv-linker".to_string(),
            )],
        ));

    // output the module
    Ok(output)
}

/// Convenience wrapper: take raw SPIR-V word slices, link them, return the
/// linked binary as a fresh `Vec<u32>`. The natural shape for callers that
/// already hold the words from a compiler invocation (e.g. DXC `-spirv`).
pub fn link_bytes(modules: &[&[u32]], opts: &Options) -> Result<Vec<u32>> {
    use rspirv::binary::Assemble;

    let mut parsed: Vec<rspirv::dr::Module> =
        modules.iter().map(|words| load_words(words)).collect();
    let mut refs: Vec<&mut rspirv::dr::Module> = parsed.iter_mut().collect();
    let linked = link(&mut refs, opts)?;
    Ok(linked.assemble())
}
