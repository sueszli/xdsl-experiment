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
        """
        For each generic call, find the function that it calls, and inline it.
        """

        callee = SymbolTable.lookup_symbol(op, op.callee)
        if callee is None:
            return  # Cannot inline if function not found

        # Check if callee is recursive (calls itself)
        # If so, do not inline, to avoid infinite expansion
        is_recursive = False
        for child_op in callee.walk():
            if isinstance(child_op, aziz.CallOp) and child_op.callee.string_value() == callee.sym_name.data:
                is_recursive = True
                break

        if is_recursive:
            return

        callable_interface = callee.get_trait(CallableOpInterface)
        if callable_interface is None:
            return

        impl_body = callable_interface.get_callable_region(callee)
        if len(impl_body.blocks) != 1:
            return  # Only inline single-block functions for simplicity

        # Clone called function body
        impl_block = impl_body.clone().block

        # Replace block args with operands
        # In Aziz, we don't have CastOp like Toy, so we direct replace
        if len(op.operands) != len(impl_block.args):
            return  # Mismatch in args

        for operand, arg in zip(op.operands, impl_block.args):
            arg.replace_by(operand)

        # remove block args
        while len(impl_block.args):
            # assert not impl_block.args[-1].uses
            rewriter.erase_block_argument(impl_block.args[-1])

        # Inline function definition before matched op
        rewriter.inline_block(impl_block, InsertPoint.before(op))

        # Get return from function definition
        return_op = op.prev_op
        assert return_op is not None

        if isinstance(return_op, aziz.ReturnOp):
            if return_op.input:
                rewriter.replace_op(op, [], [return_op.input])
            else:
                rewriter.replace_op(op, [], [])
            rewriter.erase_op(return_op)
        else:
            # Should not happen if well-formed
            pass


class RemoveUnusedPrivateFunctions(RewritePattern):
    _used_funcs: set[str] | None = None

    def should_remove_op(self, op: aziz.FuncOp) -> bool:
        if op.sym_visibility != StringAttr("private"):
            return False

        if self._used_funcs is None:
            # Get module
            module = op.parent_op()
            assert isinstance(module, ModuleOp)

            self._used_funcs = {op.callee.string_value() for op in module.walk() if isinstance(op, aziz.CallOp)}

        return op.sym_name.data not in self._used_funcs

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.FuncOp, rewriter: PatternRewriter):
        if self.should_remove_op(op):
            rewriter.erase_op(op)


class OptimizeAzizPass(ModulePass):
    name = "inline-aziz-functions"

    def apply(self, _: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(InlineFunctions()).rewrite_module(op)
        PatternRewriteWalker(RemoveUnusedPrivateFunctions()).rewrite_module(op)
        dce(op)
