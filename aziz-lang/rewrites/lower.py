from dialects import aziz
from xdsl.context import Context
from xdsl.dialects import arith, func, llvm, printf, scf
from xdsl.dialects.builtin import AnyFloat, DenseIntOrFPElementsAttr, FloatAttr, IntegerAttr, IntegerType, ModuleOp, StringAttr, VectorType, i8
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint


class AddOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.AddOp, rewriter: PatternRewriter):
        if isinstance(op.lhs.type, AnyFloat):
            rewriter.replace_op(op, arith.AddfOp(op.lhs, op.rhs))
        else:
            rewriter.replace_op(op, arith.AddiOp(op.lhs, op.rhs))


class SubOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.SubOp, rewriter: PatternRewriter):
        if isinstance(op.lhs.type, AnyFloat):
            rewriter.replace_op(op, arith.SubfOp(op.lhs, op.rhs))
        else:
            rewriter.replace_op(op, arith.SubiOp(op.lhs, op.rhs))


class MulOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.MulOp, rewriter: PatternRewriter):
        if isinstance(op.lhs.type, AnyFloat):
            rewriter.replace_op(op, arith.MulfOp(op.lhs, op.rhs))
        else:
            rewriter.replace_op(op, arith.MuliOp(op.lhs, op.rhs))


class LessThanEqualOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.LessThanEqualOp, rewriter: PatternRewriter):
        if isinstance(op.lhs.type, AnyFloat):
            rewriter.replace_op(op, arith.CmpfOp(op.lhs, op.rhs, "ole"))  # ordered less equal
        else:
            rewriter.replace_op(op, arith.CmpiOp(op.lhs, op.rhs, "sle"))  # signed less equal


class ConstantOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.ConstantOp, rewriter: PatternRewriter):
        val = op.value
        if isinstance(val, IntegerAttr):
            rewriter.replace_op(op, arith.ConstantOp(val))
        elif isinstance(val, FloatAttr):
            rewriter.replace_op(op, arith.ConstantOp(val))


class ReturnOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.ReturnOp, rewriter: PatternRewriter):
        rewriter.replace_op(op, func.ReturnOp(op.input) if op.input else func.ReturnOp())


class FuncOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.FuncOp, rewriter: PatternRewriter):
        new_op = func.FuncOp(op.sym_name.data, op.function_type, rewriter.move_region_contents_to_new_regions(op.body), visibility=op.sym_visibility)
        rewriter.replace_op(op, new_op)


class CallOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.CallOp, rewriter: PatternRewriter):
        rewriter.replace_op(op, func.CallOp(op.callee, op.arguments, op.res.types))


class IfOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.IfOp, rewriter: PatternRewriter):
        then_region = rewriter.move_region_contents_to_new_regions(op.then_region)
        else_region = rewriter.move_region_contents_to_new_regions(op.else_region)

        # if (integer) -> if (integer != 0)
        # because scf.IfOp condition must be i1
        cond = op.cond
        wider_than_bool = isinstance(cond.type, IntegerType) and cond.type.width.data != 1
        if wider_than_bool:
            zero = arith.ConstantOp(IntegerAttr(0, cond.type))
            rewriter.insert_op(zero, InsertPoint.before(rewriter.current_operation))
            cmp = arith.CmpiOp(cond, zero.result, "ne")  # condition != 0
            rewriter.insert_op(cmp, InsertPoint.before(rewriter.current_operation))
            cond = cmp.result

        new_op = scf.IfOp(cond, [op.res.type], then_region, else_region)
        rewriter.replace_op(op, new_op)


class YieldOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.YieldOp, rewriter: PatternRewriter):
        rewriter.replace_op(op, scf.YieldOp(op.input))


class StringConstantOpLowering(RewritePattern):
    # constant strings -> llvm.mlir.global (compile-time, no allocation/deallocation)
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.StringConstantOp, rewriter: PatternRewriter):
        val = op.value
        assert isinstance(val, StringAttr)
        encoded_val = val.data.encode("utf-8") + b"\0"  # null-terminate

        # llvm global type with string data
        global_name = f"str_const_{id(op)}"
        array_type = llvm.LLVMArrayType.from_size_and_type(len(encoded_val), i8)

        # requires a tensor/vector type, not LLVM array
        vector_type = VectorType(i8, [len(encoded_val)])
        initial_value = DenseIntOrFPElementsAttr.from_list(vector_type, list(encoded_val))

        global_op = llvm.GlobalOp(array_type, global_name, linkage=llvm.LinkageAttr("internal"), constant=True, value=initial_value)

        # navigate to module and insert global at the top
        module = op.parent_op()
        while module and not isinstance(module, ModuleOp):
            module = module.parent_op()
        assert isinstance(module, ModuleOp)
        rewriter.insert_op(global_op, InsertPoint.at_start(module.body.blocks[0]))

        # get pointer to global
        addr = llvm.AddressOfOp(global_name, llvm.LLVMPointerType())
        rewriter.insert_op(addr, InsertPoint.before(op))

        rewriter.replace_op(op, [], [addr.result])


class PrintOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.PrintOp, rewriter: PatternRewriter):
        if isinstance(op.input.type, llvm.LLVMPointerType):  # constant string (llvm global pointer)
            rewriter.replace_op(op, printf.PrintFormatOp("{}", op.input))
        else:  # (integers, floats, etc.)
            rewriter.replace_op(op, printf.PrintFormatOp("{}", op.input))


class LowerAzizPass(ModulePass):
    name = "lower-aziz"

    def apply(self, _: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AddOpLowering(),
                    SubOpLowering(),
                    MulOpLowering(),
                    LessThanEqualOpLowering(),
                    ConstantOpLowering(),
                    ReturnOpLowering(),
                    FuncOpLowering(),
                    CallOpLowering(),
                    IfOpLowering(),
                    YieldOpLowering(),
                    StringConstantOpLowering(),
                    PrintOpLowering(),
                ]
            )
        ).rewrite_module(op)
