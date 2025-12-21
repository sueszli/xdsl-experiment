from xdsl.context import Context
from xdsl.dialects import arith, llvm, printf, riscv
from xdsl.dialects.builtin import ModuleOp, UnrealizedConversionCastOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
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
        PatternRewriteWalker(GreedyRewritePatternApplier([SelectOpLowering()])).rewrite_module(op)


#
# printf lowering
#


class LLVMGlobalToDataSectionLowering(RewritePattern):
    # convert llvm.GlobalOp to module attributes that store the global data

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.GlobalOp, rewriter: PatternRewriter):
        # find the module
        module_op = op.parent_op()
        while module_op is not None and not isinstance(module_op, ModuleOp):
            module_op = module_op.parent_op()
        assert module_op is not None, "GlobalOp must be inside a ModuleOp"

        # extract string data from the global
        sym_name = op.sym_name.data
        global_type = op.global_type
        value = op.value

        # store the global info (we'll use this when printing assembly)
        global_info = {
            "type": global_type,
            "value": value,
            "linkage": op.linkage,
            "constant": op.constant is not None,  # UnitAttr presence indicates constant
        }

        # store in the module_op's extra data (not standard MLIR attributes)
        if not hasattr(module_op, "_riscv_globals"):
            module_op._riscv_globals = {}
        module_op._riscv_globals[sym_name] = global_info

        # remove the global op
        rewriter.erase_op(op, safe_erase=False)


class LLVMAddressOfToRISCVLowering(RewritePattern):
    # convert llvm.AddressOfOp to riscv.LiOp with label reference (will be post-processed to 'la')

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.AddressOfOp, rewriter: PatternRewriter):
        # get the symbol name
        global_name = op.global_name.root_reference.data

        # create a riscv.li operation with immediate 0 (placeholder)
        # we'll store the symbol name in an attribute for later processing
        reg_type = riscv.IntRegisterType.unallocated()
        li_op = riscv.LiOp(0, rd=reg_type)

        # store the symbol reference in the op for later conversion to 'la'
        li_op.attributes["symbol_ref"] = op.global_name.root_reference

        # replace the addressof with li
        rewriter.replace_op(op, [li_op], [li_op.rd])


class RemovePrintfOpLowering(RewritePattern):
    # remove printf operations since they can't be represented in riscv assembly without libc

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: printf.PrintFormatOp, rewriter: PatternRewriter):
        rewriter.erase_op(op, safe_erase=False)


class RemoveUnprintableOpsPass(ModulePass):
    name = "remove-unprintable-ops"

    def apply(self, _: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(GreedyRewritePatternApplier([LLVMGlobalToDataSectionLowering()])).rewrite_module(op)
        PatternRewriteWalker(GreedyRewritePatternApplier([LLVMAddressOfToRISCVLowering()])).rewrite_module(op)
        PatternRewriteWalker(GreedyRewritePatternApplier([RemovePrintfOpLowering()])).rewrite_module(op)


def emit_data_section(module_op: ModuleOp) -> str:
    # convert LLVM global strings stored in module attributes to .data section in assembly
    if not hasattr(module_op, "_riscv_globals") or not module_op._riscv_globals:
        return ""

    lines = [".data"]
    for sym_name, global_info in module_op._riscv_globals.items():
        value_attr = global_info["value"]
        lines.extend([f".globl {sym_name}", f"{sym_name}:"])
        assert hasattr(value_attr, "data") and hasattr(value_attr.data, "data"), "unsupported global value type"

        # convert byte array to string
        string_bytes = bytes(value_attr.data.data)
        try:
            string_content = string_bytes[: string_bytes.index(0)].decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            string_content = string_bytes.decode("utf-8", errors="replace").rstrip("\x00")

        # emit as .string directive
        escaped = string_content.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'    .string "{escaped}"')

    lines.append("")
    return "\n".join(lines) + "\n"
