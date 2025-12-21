from xdsl.context import Context
from xdsl.dialects import arith, riscv
from xdsl.dialects.builtin import ModuleOp, UnrealizedConversionCastOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint


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
