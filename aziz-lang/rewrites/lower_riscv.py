import re

from xdsl.context import Context
from xdsl.dialects import arith, llvm, printf, riscv
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, UnrealizedConversionCastOp
from xdsl.ir import Attribute
from xdsl.irdl import attr_def, base, irdl_op_definition
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint

#
# branching lowering
#


class SelectOpLowering(RewritePattern):
    # lower arith.select by replacing branches with bitwise operations
    # mask = 0b1111 if cond=1, 0b0000 if cond=0
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


class RemoveUnprintableOpsPass(ModulePass):
    name = "remove-unprintable-ops"

    def apply(self, _: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(LLVMGlobalToRISCVGlobalLowering()).rewrite_module(op)
        PatternRewriteWalker(LLVMAddressOfToRISCVLowering()).rewrite_module(op)
        PatternRewriteWalker(RemovePrintfOpLowering()).rewrite_module(op)


#
# TODO: BRING THESE ALL INTO PASSES
#


def emit_data_section(module_op: ModuleOp) -> str:
    globals = [op for op in module_op.walk() if isinstance(op, RISCVGlobalOp)]

    if not globals:
        return ""

    lines = [".data"]
    for global_op in globals:
        sym_name = global_op.sym_name.data
        value_attr = global_op.value

        lines.extend([f".globl {sym_name}", f"{sym_name}:"])
        assert hasattr(value_attr, "data") and hasattr(value_attr.data, "data")
        string_bytes = bytes(value_attr.data.data)
        try:
            null_index = string_bytes.index(0)
            string_content = string_bytes[:null_index].decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            string_content = string_bytes.decode("utf-8", errors="replace").rstrip("\x00")

        escaped = string_content.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
        lines.append(f'    .string "{escaped}"')

    lines.append("")
    return "\n".join(lines) + "\n"


def map_virtual_to_physical_registers(asm: str) -> str:
    virtual_regs = set(re.findall(r"\bj_\d+\b", asm))
    physical_regs = [f"t{i}" for i in range(7)] + [f"s{i}" for i in range(12)]
    if len(virtual_regs) > len(physical_regs):
        raise RuntimeError(f"too many virtual registers: {len(virtual_regs)} > {len(physical_regs)}")
    virtual_list = sorted(virtual_regs, key=lambda x: int(x.split("_")[1]))
    reg_map = {v: physical_regs[i] for i, v in enumerate(virtual_list)}
    for virt, phys in reg_map.items():
        asm = re.sub(rf"\b{virt}\b", phys, asm)
    return asm
