from xdsl.context import Context
from xdsl.dialects import llvm, printf
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern


class RemovePrintfOpLowering(RewritePattern):
    # remove printf operations since they can't be represented in riscv assembly without libc

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: printf.PrintFormatOp, rewriter: PatternRewriter):
        rewriter.erase_op(op, safe_erase=False)


class RemoveLLVMAddressOfOpLowering(RewritePattern):
    # remove llvm addressof operations that reference globals

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.AddressOfOp, rewriter: PatternRewriter):
        rewriter.erase_op(op, safe_erase=False)


class RemoveLLVMGlobalOpLowering(RewritePattern):
    # remove llvm global definitions (string constants) that can't be represented in riscv assembly

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.GlobalOp, rewriter: PatternRewriter):
        rewriter.erase_op(op, safe_erase=False)


class RemoveUnprintableOpsPass(ModulePass):
    name = "remove-unprintable-ops"

    def apply(self, _: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(GreedyRewritePatternApplier([RemovePrintfOpLowering()])).rewrite_module(op)
        PatternRewriteWalker(GreedyRewritePatternApplier([RemoveLLVMAddressOfOpLowering()])).rewrite_module(op)
        PatternRewriteWalker(GreedyRewritePatternApplier([RemoveLLVMGlobalOpLowering()])).rewrite_module(op)
