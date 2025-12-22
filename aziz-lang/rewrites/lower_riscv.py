from xdsl.context import Context
from xdsl.dialects import arith, func, llvm, printf, riscv, riscv_func, scf
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, UnrealizedConversionCastOp
from xdsl.ir import Attribute
from xdsl.irdl import attr_def, base, irdl_op_definition
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint

#
# assembly ops
#


@irdl_op_definition
class RISCVDirectiveOp(riscv.RISCVAsmOperation):
    name = "riscv.directive"
    directive = attr_def(StringAttr)
    value = attr_def(StringAttr)

    def __init__(self, directive: str, value: str = ""):
        super().__init__(
            attributes={
                "directive": StringAttr(directive),
                "value": StringAttr(value),
            }
        )

    def assembly_line(self) -> str | None:
        if self.value.data:
            return f"{self.directive.data} {self.value.data}"
        return self.directive.data


#
# branching lowering
#


class SelectOpLowering(RewritePattern):
    # lower arith.select by replacing branches with bitwise operations
    # mask = 0b1111 if cond=1 else 0b0000
    # result = (true_val & mask) | (false_val & ~mask)
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SelectOp, rewriter: PatternRewriter):
        cond = op.cond
        true_val = op.lhs
        false_val = op.rhs
        reg_type = riscv.IntRegisterType.unallocated()

        # cast condition to riscv register
        cond_cast = UnrealizedConversionCastOp.create(operands=[cond], result_types=[reg_type])
        rewriter.insert_op(cond_cast, InsertPoint.before(op))
        cond_reg = cond_cast.results[0]

        # cast true_val and false_val to riscv registers for bitwise ops
        true_cast = UnrealizedConversionCastOp.create(operands=[true_val], result_types=[reg_type])
        rewriter.insert_op(true_cast, InsertPoint.before(op))
        true_reg = true_cast.results[0]

        false_cast = UnrealizedConversionCastOp.create(operands=[false_val], result_types=[reg_type])
        rewriter.insert_op(false_cast, InsertPoint.before(op))
        false_reg = false_cast.results[0]

        # mask = 0 - cond  (creates 0b0000 or 0b1111)
        zero = riscv.GetRegisterOp(riscv.Registers.ZERO)
        rewriter.insert_op(zero, InsertPoint.before(op))
        mask = riscv.SubOp(zero.res, cond_reg, rd=reg_type)
        rewriter.insert_op(mask, InsertPoint.before(op))

        # t1 = true_val & mask
        t1 = riscv.AndOp(true_reg, mask.rd, rd=reg_type)
        rewriter.insert_op(t1, InsertPoint.before(op))

        # not_mask = mask XOR -1
        not_mask = riscv.XoriOp(mask.rd, -1, rd=reg_type)
        rewriter.insert_op(not_mask, InsertPoint.before(op))

        # t2 = false_val & not_mask
        t2 = riscv.AndOp(false_reg, not_mask.rd, rd=reg_type)
        rewriter.insert_op(t2, InsertPoint.before(op))

        # result = t1 | t2
        result = riscv.OrOp(t1.rd, t2.rd, rd=reg_type)
        rewriter.insert_op(result, InsertPoint.before(op))

        # cast result back to original type
        result_cast = UnrealizedConversionCastOp.create(operands=[result.rd], result_types=[op.result.type])
        rewriter.insert_op(result_cast, InsertPoint.before(op))

        # replace the select op
        rewriter.replace_op(op, [], [result_cast.results[0]])


class LowerSelectPass(ModulePass):
    name = "lower-select"

    def apply(self, _: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(SelectOpLowering()).rewrite_module(op)


#
# global data lowering
#


@irdl_op_definition
class RISCVLabelOp(riscv.RISCVAsmOperation):
    name = "riscv.label"
    label = attr_def(riscv.LabelAttr)

    def __init__(self, label: str | riscv.LabelAttr):
        if isinstance(label, str):
            label = riscv.LabelAttr(label)
        super().__init__(attributes={"label": label})

    def assembly_line(self) -> str | None:
        return f"{self.label.data}:"


@irdl_op_definition
class RISCVAddSpOp(riscv.RISCVAsmOperation):
    name = "riscv.add_sp"
    immediate = attr_def(IntegerAttr)

    def __init__(self, immediate: int):
        super().__init__(attributes={"immediate": IntegerAttr(immediate, 32)})

    def assembly_line(self) -> str | None:
        return f"addi sp, sp, {self.immediate.value.data}"


@irdl_op_definition
class RISCVSaveRaOp(riscv.RISCVAsmOperation):
    name = "riscv.save_ra"

    def assembly_line(self) -> str | None:
        return "sd ra, 0(sp)"


@irdl_op_definition
class RISCVRestoreRaOp(riscv.RISCVAsmOperation):
    name = "riscv.restore_ra"

    def assembly_line(self) -> str | None:
        return "ld ra, 0(sp)"


class AddRecursionSupportPass(ModulePass):
    name = "add-recursion-support"

    def apply(self, _: Context, op: ModuleOp) -> None:
        for func_op in op.walk():
            if not isinstance(func_op, riscv_func.FuncOp):
                continue
            if func_op.sym_name.data in ["main", "_start"]:
                continue

            # Prologue
            block = func_op.body.blocks[0]
            if block.ops:
                first_op = list(block.ops)[0]
                add_sp = RISCVAddSpOp(-8)
                save_ra = RISCVSaveRaOp()

                block.insert_op_before(add_sp, first_op)
                block.insert_op_before(save_ra, first_op)  # Insert save_ra before first_op (so after add_sp?)
                # Wait:
                # Start: [First]
                # Insert add_sp before First -> [add_sp, First]
                # Insert save_ra before First -> [add_sp, save_ra, First]
                # Correct order: add_sp (-8), save_ra (sd).

            # Epilogue
            for b in func_op.body.blocks:
                for ret in list(b.ops):
                    if isinstance(ret, riscv_func.ReturnOp):
                        # INSERT BEFORE ret
                        restore_ra = RISCVRestoreRaOp()
                        rest_sp = RISCVAddSpOp(8)

                        b.insert_op_before(restore_ra, ret)
                        b.insert_op_before(rest_sp, ret)
                        # Order: restore_ra, rest_sp, ret. Correct.


@irdl_op_definition
class RISCVGlobalOp(riscv.RISCVAsmOperation):
    """
    Represents a global data symbol in the .data section.
    This operation holds global data that will be emitted in assembly.
    """

    name = "riscv.global"

    sym_name = attr_def(StringAttr)
    value = attr_def(base(Attribute))  # Store the llvm dense array
    is_constant = attr_def(base(Attribute))  # Store a boolean indicator

    def __init__(self, sym_name: str | StringAttr, value: Attribute, is_constant: bool = True):
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)

        constant_attr = IntegerAttr(1 if is_constant else 0, 1)

        super().__init__(
            attributes={
                "sym_name": sym_name,
                "value": value,
                "is_constant": constant_attr,
            }
        )

    def assembly_line(self) -> str | None:
        # This will be handled by emit_data_section walking the IR
        return None


class LLVMGlobalToRISCVGlobalLowering(RewritePattern):
    """Convert llvm.GlobalOp to riscv.global"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.GlobalOp, rewriter: PatternRewriter):
        sym_name = op.sym_name.data
        value = op.value
        is_constant = op.constant is not None

        # Create a RISCV global operation
        global_op = RISCVGlobalOp(sym_name, value, is_constant)
        rewriter.insert_op(global_op, InsertPoint.at_start(op.parent_block()))

        # Remove the LLVM global
        rewriter.erase_op(op, safe_erase=False)


class LLVMAddressOfToRISCVLowering(RewritePattern):
    """Convert llvm.AddressOfOp to riscv.li with a label"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.AddressOfOp, rewriter: PatternRewriter):
        global_name = op.global_name.root_reference.data
        reg_type = riscv.IntRegisterType.unallocated()
        label = riscv.LabelAttr(global_name)
        li_op = riscv.LiOp(label, rd=reg_type)
        rewriter.replace_op(op, [li_op], [li_op.rd])


class RemovePrintfOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: printf.PrintFormatOp, rewriter: PatternRewriter):
        rewriter.erase_op(op, safe_erase=False)


# bring back printf, write some util function in assembly
class RemoveUnprintableOpsPass(ModulePass):
    name = "remove-unprintable-ops"

    def apply(self, _: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(LLVMGlobalToRISCVGlobalLowering()).rewrite_module(op)
        PatternRewriteWalker(LLVMAddressOfToRISCVLowering()).rewrite_module(op)
        PatternRewriteWalker(RemovePrintfOpLowering()).rewrite_module(op)


class CustomScfIfToRiscvLowering(RewritePattern):
    def __init__(self):
        super().__init__()
        self.label_counter = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.IfOp, rewriter: PatternRewriter):
        self.label_counter += 1
        suffix = f"{self.label_counter}"
        else_label = f"else_{suffix}"
        cont_label = f"cont_{suffix}"

        reg_type = riscv.IntRegisterType.unallocated()
        cond_cast = UnrealizedConversionCastOp.create(operands=[op.cond], result_types=[reg_type])
        rewriter.insert_op(cond_cast, InsertPoint.before(op))

        zero = riscv.GetRegisterOp(riscv.Registers.ZERO)
        rewriter.insert_op(zero, InsertPoint.before(op))

        # BeqOp args: rs1, rs2, offset
        beq = riscv.BeqOp(cond_cast.results[0], zero.res, offset=riscv.LabelAttr(else_label))
        rewriter.insert_op(beq, InsertPoint.before(op))

        res_ptrs = []
        if op.results:
            for res in op.results:
                add_sp = RISCVAddSpOp(-8)
                rewriter.insert_op(add_sp, InsertPoint.before(op))

                sp = riscv.GetRegisterOp(riscv.Registers.SP)
                rewriter.insert_op(sp, InsertPoint.before(op))

                res_ptrs.append(sp.res)

        for bop in list(op.true_region.block.ops):
            if isinstance(bop, scf.YieldOp):
                for i, val in enumerate(bop.operands):
                    val_cast = UnrealizedConversionCastOp.create(operands=[val], result_types=[reg_type])
                    rewriter.insert_op(val_cast, InsertPoint.before(op))
                    store = riscv.SwOp(res_ptrs[i], val_cast.results[0], 0)
                    rewriter.insert_op(store, InsertPoint.before(op))
            else:
                bop.detach()
                rewriter.insert_op(bop, InsertPoint.before(op))

        j = riscv.JOp(riscv.LabelAttr(cont_label))
        rewriter.insert_op(j, InsertPoint.before(op))

        rewriter.insert_op(RISCVLabelOp(else_label), InsertPoint.before(op))

        if op.false_region.block:
            for bop in list(op.false_region.block.ops):
                if isinstance(bop, scf.YieldOp):
                    for i, val in enumerate(bop.operands):
                        val_cast = UnrealizedConversionCastOp.create(operands=[val], result_types=[reg_type])
                        rewriter.insert_op(val_cast, InsertPoint.before(op))
                        store = riscv.SwOp(res_ptrs[i], val_cast.results[0], 0)
                        rewriter.insert_op(store, InsertPoint.before(op))
                else:
                    bop.detach()
                    rewriter.insert_op(bop, InsertPoint.before(op))

        rewriter.insert_op(RISCVLabelOp(cont_label), InsertPoint.before(op))

        final_results = []
        for i, m in enumerate(res_ptrs):
            load = riscv.LwOp(m, 0, rd=reg_type)
            rewriter.insert_op(load, InsertPoint.before(op))

            res_cast = UnrealizedConversionCastOp.create(operands=[load.rd], result_types=[op.results[i].type])
            rewriter.insert_op(res_cast, InsertPoint.before(op))
            final_results.append(res_cast.results[0])

            # Restore SP
            add_sp = RISCVAddSpOp(8)
            rewriter.insert_op(add_sp, InsertPoint.before(op))

        rewriter.replace_op(op, [], final_results)


class CustomLowerScfToRiscvPass(ModulePass):
    name = "custom-lower-scf-to-riscv"

    def apply(self, _: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(CustomScfIfToRiscvLowering()).rewrite_module(op)


class LowerRISCVGlobalOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: RISCVGlobalOp, rewriter: PatternRewriter):
        sym_name = op.sym_name.data
        value_attr = op.value

        # .data
        data_directive = RISCVDirectiveOp(".data")
        rewriter.insert_op(data_directive, InsertPoint.before(op))

        # .globl sym
        globl_directive = RISCVDirectiveOp(".globl", sym_name)
        rewriter.insert_op(globl_directive, InsertPoint.before(op))

        # sym:
        label_op = RISCVLabelOp(sym_name)
        rewriter.insert_op(label_op, InsertPoint.before(op))

        # .string "..."
        assert hasattr(value_attr, "data") and hasattr(value_attr.data, "data")
        string_bytes = bytes(value_attr.data.data)
        try:
            null_index = string_bytes.index(0)
            string_content = string_bytes[:null_index].decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            string_content = string_bytes.decode("utf-8", errors="replace").rstrip("\x00")

        escaped = string_content.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
        string_directive = RISCVDirectiveOp(".string", f'"{escaped}"')
        rewriter.insert_op(string_directive, InsertPoint.before(op))

        rewriter.erase_op(op)


class EmitDataSectionPass(ModulePass):
    name = "emit-data-section"

    def apply(self, _: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerRISCVGlobalOp()).rewrite_module(op)

        # Ensure .text section for functions
        # We find the first function and insert .text before it
        for oper in op.body.blocks[0].ops:
            if isinstance(oper, (riscv_func.FuncOp, func.FuncOp)):
                text_directive = RISCVDirectiveOp(".text")
                op.body.blocks[0].insert_op_before(text_directive, oper)
                break


class MapToPhysicalRegistersPass(ModulePass):
    name = "map-to-physical-registers"

    def apply(self, _: Context, op: ModuleOp) -> None:
        # manual walk to avoid pattern rewriter issues
        for func_op in op.walk():
            if not isinstance(func_op, riscv_func.FuncOp):
                continue

            # find all virtual registers
            virtual_regs = set()

            def collect_virtual_regs(oper):
                for result in oper.results:
                    if isinstance(result.type, riscv.RegisterType):
                        if hasattr(result.type, "register_name") and result.type.register_name:
                            reg_name = result.type.register_name
                            if isinstance(reg_name, StringAttr):
                                reg_name = reg_name.data
                            if reg_name.startswith("j_"):
                                virtual_regs.add(reg_name)

            # Collect from func op (entry block args are not results, but args)
            # Ops inside func return virtual regs
            for child in func_op.walk():
                collect_virtual_regs(child)

            # Also check block args of func_op??
            # Typically virtual regs are defined by ops (e.g. li, add).
            # But function args might be virtual too?
            # In RISCV, args are usually physical (a0-a7) due to ABI.
            # But let's check just in case.
            for block in func_op.body.blocks:
                for arg in block.args:
                    if isinstance(arg.type, riscv.RegisterType):
                        if hasattr(arg.type, "register_name") and arg.type.register_name:
                            reg_name = arg.type.register_name
                            if isinstance(reg_name, StringAttr):
                                reg_name = reg_name.data
                            if reg_name.startswith("j_"):
                                virtual_regs.add(reg_name)

            if not virtual_regs:
                continue

            # map to physical registers
            physical_regs = [f"t{i}" for i in range(7)] + [f"s{i}" for i in range(12)]
            if len(virtual_regs) > len(physical_regs):
                raise RuntimeError(f"too many virtual registers: {len(virtual_regs)} > {len(physical_regs)}")

            virtual_list = sorted(virtual_regs, key=lambda x: int(x.split("_")[1]))
            reg_map = {v: physical_regs[i] for i, v in enumerate(virtual_list)}

            # apply mapping
            def map_regs(oper):
                if hasattr(oper, "result_types"):
                    new_types = []
                    changed = False
                    for typ in oper.result_types:
                        if isinstance(typ, riscv.RegisterType):
                            if hasattr(typ, "register_name") and typ.register_name:
                                reg_name = typ.register_name
                                if isinstance(reg_name, StringAttr):
                                    reg_name = reg_name.data
                                if reg_name in reg_map:
                                    new_name = reg_map[reg_name]
                                    new_type = riscv.IntRegisterType.from_name(new_name)
                                    new_types.append(new_type)
                                    changed = True
                                    continue
                        new_types.append(typ)

                    if changed:
                        # Construct new operation with updated result types
                        # We use Operation.create to be generic.
                        # Note: We must clone regions if present, but for now assuming simple ops
                        # If regions are present, we might need to detach them from old op?
                        regions = [r for r in oper.regions]
                        for r in regions:
                            r.detach()  # Detach from old op so we can move to new op

                        new_op = oper.__class__.create(operands=oper.operands, result_types=new_types, attributes=oper.attributes, successors=oper.successors, regions=regions)

                        # Replace usages
                        for i, old_res in enumerate(oper.results):
                            old_res.replace_by(new_op.results[i])

                        # Replace in block
                        if oper.parent_block():
                            block = oper.parent_block()
                            block.insert_op_before(new_op, oper)
                            block.detach_op(oper)

                for region in oper.regions:
                    for block in region.blocks:
                        for arg in block.args:
                            if isinstance(arg.type, riscv.RegisterType):
                                if hasattr(arg.type, "register_name") and arg.type.register_name:
                                    reg_name = arg.type.register_name
                                    if isinstance(reg_name, StringAttr):
                                        reg_name = reg_name.data
                                    if reg_name in reg_map:
                                        new_name = reg_map[reg_name]
                                        new_type = riscv.IntRegisterType.from_name(new_name)
                                        arg.type = new_type

            # Apply to func op args
            map_regs(func_op)
            # Apply to all children (ops inside func)
            for child in func_op.walk():
                map_regs(child)
