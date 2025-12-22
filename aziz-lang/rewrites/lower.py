from dialects import aziz
from xdsl.context import Context
from xdsl.dialects import arith, func, llvm, printf, scf
from xdsl.dialects.builtin import AnyFloat, DenseIntOrFPElementsAttr, FloatAttr, IntegerAttr, IntegerType, ModuleOp, StringAttr, VectorType, i8
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint

#
# arith
#


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


class CastIntToFloatOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.CastIntToFloatOp, rewriter: PatternRewriter):
        rewriter.replace_op(op, arith.SIToFPOp(op.input, op.res.type))


class ConstantOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.ConstantOp, rewriter: PatternRewriter):
        val = op.value
        if isinstance(val, IntegerAttr):
            rewriter.replace_op(op, arith.ConstantOp(val))
        elif isinstance(val, FloatAttr):
            rewriter.replace_op(op, arith.ConstantOp(val))


class IfOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.IfOp, rewriter: PatternRewriter):
        # get the condition - convert to i1 if needed
        cond = op.cond
        wider_than_bool = isinstance(cond.type, IntegerType) and cond.type.width.data != 1
        if wider_than_bool:
            zero = arith.ConstantOp(IntegerAttr(0, cond.type))
            rewriter.insert_op(zero, InsertPoint.before(rewriter.current_operation))
            cmp = arith.CmpiOp(cond, zero.result, "ne")  # condition != 0
            rewriter.insert_op(cmp, InsertPoint.before(rewriter.current_operation))
            cond = cmp.result

        # inline both branches before the if op (they compute the values)
        then_block = op.then_region.block
        else_block = op.else_region.block

        # get the yield values from each branch
        then_yield = then_block.last_op
        else_yield = else_block.last_op
        assert isinstance(then_yield, aziz.YieldOp)
        assert isinstance(else_yield, aziz.YieldOp)

        # inline then branch ops (except yield)
        for bop in list(then_block.ops)[:-1]:
            bop.detach()
            rewriter.insert_op(bop, InsertPoint.before(op))
        then_value = then_yield.input

        # inline else branch ops (except yield)
        for bop in list(else_block.ops)[:-1]:
            bop.detach()
            rewriter.insert_op(bop, InsertPoint.before(op))
        else_value = else_yield.input

        # use arith.select to pick between the two values
        select = arith.SelectOp(cond, then_value, else_value)
        rewriter.replace_op(op, select)


#
# func
#


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


#
# scf
#


class YieldOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.YieldOp, rewriter: PatternRewriter):
        rewriter.replace_op(op, scf.YieldOp(op.input))


#
# printf
#


class PrintOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.PrintOp, rewriter: PatternRewriter):
        if isinstance(op.input.type, llvm.LLVMPointerType):
            # constant string (llvm global pointer)
            rewriter.replace_op(op, printf.PrintFormatOp("{}", op.input))
        else:
            # integers, floats
            rewriter.replace_op(op, printf.PrintFormatOp("{}", op.input))


#
# llvm
#


class StringConstantOpLowering(RewritePattern):
    def __init__(self):
        super().__init__()
        self._string_cache: dict[str, str] = {}  # string content -> global name
        self._counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.StringConstantOp, rewriter: PatternRewriter):
        val = op.value
        assert isinstance(val, StringAttr)
        string_content = val.data
        encoded_val = string_content.encode("utf-8") + b"\0"

        # get or create llvm.global for the string
        if string_content not in self._string_cache:
            module = self._get_module(op)
            self._create_global_for_string(string_content, encoded_val, module, rewriter)

        global_name = self._string_cache[string_content]

        # create addressof and replace
        addr = llvm.AddressOfOp(global_name, llvm.LLVMPointerType())
        rewriter.insert_op(addr, InsertPoint.before(op))
        rewriter.replace_op(op, [], [addr.result])

    def _get_module(self, op: aziz.StringConstantOp) -> ModuleOp:
        module = op.parent_op()
        while module and not isinstance(module, ModuleOp):
            module = module.parent_op()
        assert isinstance(module, ModuleOp)
        return module

    def _create_global_for_string(self, string_content: str, encoded_val: bytes, module: ModuleOp, rewriter: PatternRewriter) -> None:
        global_name = f".aziz.str.{self._counter}"
        self._counter += 1
        self._string_cache[string_content] = global_name

        array_type = llvm.LLVMArrayType.from_size_and_type(len(encoded_val), i8)
        vector_type = VectorType(i8, [len(encoded_val)])
        initial_value = DenseIntOrFPElementsAttr.from_list(vector_type, list(encoded_val))  # requires tensor/vector type, not LLVM array type
        global_op = llvm.GlobalOp(array_type, global_name, linkage=llvm.LinkageAttr("internal"), constant=True, value=initial_value)
        rewriter.insert_op(global_op, InsertPoint.at_start(module.body.blocks[0]))


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
                    CastIntToFloatOpLowering(),
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
