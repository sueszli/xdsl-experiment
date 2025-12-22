from dialects import aziz
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint
from xdsl.traits import CallableOpInterface, SymbolTable
from xdsl.transforms.dead_code_elimination import dce


class InlineFunctions(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.CallOp, rewriter: PatternRewriter):
        callee = SymbolTable.lookup_symbol(op, op.callee)

        is_unknown = callee is None
        if is_unknown:
            return

        is_self_call = lambda op: isinstance(op, aziz.CallOp) and op.callee.string_value() == callee.sym_name.data
        is_recursive = any(is_self_call(child_op) for child_op in callee.walk())
        if is_recursive:
            return

        callable_interface = callee.get_trait(CallableOpInterface)
        not_callable = callable_interface is None
        if not_callable:
            return

        # just inline one-liner functions for simplicity
        impl_body = callable_interface.get_callable_region(callee)
        is_one_liner = len(impl_body.blocks) == 1  # just inline on
        if not is_one_liner:
            return

        impl_block = impl_body.clone().block
        assert len(op.operands) == len(impl_block.args), f"arity mismatch in inlining {op} and {callee}"

        for operand, arg in zip(op.operands, impl_block.args):
            arg.replace_by(operand)

        while len(impl_block.args):
            rewriter.erase_block_argument(impl_block.args[-1])

        rewriter.inline_block(impl_block, InsertPoint.before(op))

        return_op = op.prev_op
        assert return_op is not None
        assert isinstance(return_op, aziz.ReturnOp)
        if return_op.input:
            rewriter.replace_op(op, [], [return_op.input])
        else:
            rewriter.replace_op(op, [], [])
        rewriter.erase_op(return_op)


class RemoveUnusedPrivateFunctions(RewritePattern):
    _used_funcs: set[str] | None = None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.FuncOp, rewriter: PatternRewriter):
        if self._is_unused(op):
            rewriter.erase_op(op)

    def _is_unused(self, op: aziz.FuncOp) -> bool:
        if op.sym_visibility != StringAttr("private"):
            return False

        if self._used_funcs is None:
            module = op.parent_op()
            assert isinstance(module, ModuleOp)
            self._used_funcs = {op.callee.string_value() for op in module.walk() if isinstance(op, aziz.CallOp)}

        return op.sym_name.data not in self._used_funcs


class OptimizeAzizPass(ModulePass):
    name = "optimize-aziz"

    def apply(self, _: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(InlineFunctions()).rewrite_module(op)
        PatternRewriteWalker(RemoveUnusedPrivateFunctions()).rewrite_module(op)
        dce(op)
