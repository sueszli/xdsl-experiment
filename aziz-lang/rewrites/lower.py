from dialects import aziz
from xdsl.context import Context
from xdsl.dialects import arith, func, llvm, printf, scf
from xdsl.dialects.builtin import AnyFloat, ArrayAttr, FloatAttr, IntegerAttr, IntegerType, ModuleOp, StringAttr, i8
from xdsl.ir import Block
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
            rewriter.insert_op(zero, InsertPoint.before(op))
            cmp = arith.CmpiOp(cond, zero.result, "ne")  # condition != 0
            rewriter.insert_op(cmp, InsertPoint.before(op))
            cond = cmp.result

        # create scf.if which preserves control flow (only executing the taken branch)
        then_region = rewriter.move_region_contents_to_new_regions(op.then_region)
        else_region = rewriter.move_region_contents_to_new_regions(op.else_region)

        new_op = scf.IfOp(cond, [r.type for r in op.results], then_region, else_region)
        rewriter.replace_op(op, new_op)


#
# func
#


class ReturnOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.ReturnOp, rewriter: PatternRewriter):
        rewriter.replace_op(op, func.ReturnOp(op.input) if op.input else func.ReturnOp())


class FuncOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.FuncOp, rewriter: PatternRewriter):  # strings become pointers
        # convert inputs/outputs
        convert_type = lambda t: llvm.LLVMPointerType() if isinstance(t, aziz.StringType) else t
        inputs = [convert_type(t) for t in op.function_type.inputs]
        outputs = [convert_type(t) for t in op.function_type.outputs]
        new_type = func.FunctionType.from_lists(inputs, outputs)

        # move region
        new_body = rewriter.move_region_contents_to_new_regions(op.body)

        # update block arguments in new body
        for block in list(new_body.blocks):
            new_arg_types = [convert_type(arg.type) for arg in block.args]
            if new_arg_types == [arg.type for arg in block.args]:
                continue

            # replace args uses
            new_block = Block(arg_types=new_arg_types)
            for old_arg, new_arg in zip(block.args, new_block.args):
                old_arg.replace_by(new_arg)

            # move ops
            ops = list(block.ops)
            for o in ops:
                o.detach()
                new_block.add_op(o)

            # replace block
            idx = next(i for i, b in enumerate(new_body.blocks) if b is block)
            new_body.detach_block(block)
            new_body.insert_block(new_block, idx)

        new_op = func.FuncOp(op.sym_name.data, new_type, new_body, visibility=op.sym_visibility)
        rewriter.replace_op(op, new_op)


class CallOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.CallOp, rewriter: PatternRewriter):
        convert_type = lambda t: llvm.LLVMPointerType() if isinstance(t, aziz.StringType) else t
        res_types = [convert_type(t) for t in op.res.types]
        rewriter.replace_op(op, func.CallOp(op.callee, op.arguments, res_types))


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

        # Use ArrayAttr for initializer to match llvm.array type in mlir-translate
        array_type = llvm.LLVMArrayType.from_size_and_type(len(encoded_val), i8)
        initial_value = ArrayAttr([IntegerAttr(b, i8) for b in encoded_val])

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
