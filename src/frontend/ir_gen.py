from __future__ import annotations

from xdsl.builder import Builder, InsertPoint
from xdsl.dialects.builtin import FunctionType, ModuleOp, i32
from xdsl.ir import Block, Region, SSAValue
from xdsl.utils.scoped_dict import ScopedDict

from dialects.aziz import AddOp, CallOp, ConstantOp, FuncOp, IfOp, LessThanEqualOp, MulOp, PrintOp, ReturnOp, StringConstantOp, SubOp, YieldOp

from .ast_nodes import BinaryExprAST, CallExprAST, ExprAST, FunctionAST, IfExprAST, ModuleAST, NumberExprAST, PrintExprAST, PrototypeAST, StringExprAST, VariableExprAST


class IRGenError(Exception):
    pass


class IRGen:
    module: ModuleOp  # generated MLIR ModuleOp from AST
    builder: Builder  # keeps current insertion point for next op
    symbol_table: ScopedDict[str, SSAValue] | None = None  # variable name -> SSAValue for current scope. dropped on scope exit

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))

    def ir_gen_module(self, module_ast: ModuleAST) -> ModuleOp:
        # top level functions
        for op in module_ast.ops:
            if isinstance(op, FunctionAST):
                self._ir_gen_function(op)

        # implicit main function
        main_body = [op for op in module_ast.ops if isinstance(op, ExprAST)]
        if main_body:
            loc = main_body[0].loc
            main_func = FunctionAST(loc, PrototypeAST(loc, "main", []), tuple(main_body))
            self._ir_gen_function(main_func)

        # verify types
        self.module.verify()
        return self.module

    def _ir_gen_function(self, func_ast: FunctionAST) -> FuncOp:
        parent_builder = self.builder
        self.symbol_table = ScopedDict()

        # Default argument types to i32 for now, mirroring toy's unranked tensor assumption -----> this is stupid
        arg_types = [i32] * len(func_ast.proto.args)

        block = Block(arg_types=arg_types)
        self.builder = Builder(InsertPoint.at_end(block))

        for name, value in zip(func_ast.proto.args, block.args):
            self._declare(name, value)

        last_val = None
        for expr in func_ast.body:
            last_val = self._ir_gen_expr(expr)

        return_types = []
        if not block.ops or not isinstance(block.last_op, ReturnOp):
            # Implicit return. If last_val exists, return it, else return 0.
            val = last_val if last_val is not None else self.builder.insert(ConstantOp(0)).res
            self.builder.insert(ReturnOp(val))
            return_types = [i32]
        else:
            # Explicit return exists, infer type from it
            if block.last_op.input:
                return_types = [block.last_op.input.type]

        func_type = FunctionType.from_lists(arg_types, return_types)
        func_op = FuncOp(func_ast.proto.name, func_type, Region(block))

        self.symbol_table = None
        self.builder = parent_builder
        return self.builder.insert(func_op)

    def _declare(self, var: str, value: SSAValue) -> bool:
        # declare a variable in the current scope, return success if not already declared
        assert self.symbol_table is not None
        if var in self.symbol_table:
            return False
        self.symbol_table[var] = value
        return True


    def _ir_gen_expr(self, expr: ExprAST) -> SSAValue:
        if isinstance(expr, BinaryExprAST):
            return self._ir_gen_binary_expr(expr)
        if isinstance(expr, NumberExprAST):
            return self._ir_gen_number_expr(expr)
        if isinstance(expr, VariableExprAST):
            return self._ir_gen_variable_expr(expr)
        if isinstance(expr, CallExprAST):
            return self._ir_gen_call_expr(expr)
        if isinstance(expr, PrintExprAST):
            self._ir_gen_print_expr(expr)
            return None  # Print is void/statement-like in usage often, but expr in AST?
        if isinstance(expr, IfExprAST):
            return self._ir_gen_if_expr(expr)
        if isinstance(expr, StringExprAST):
            return self._ir_gen_string_expr(expr)

        raise IRGenError(f"Unknown expr: {expr}")

    def _ir_gen_binary_expr(self, expr: BinaryExprAST) -> SSAValue:
        lhs = self._ir_gen_expr(expr.lhs)
        rhs = self._ir_gen_expr(expr.rhs)

        if expr.op == "+":
            return self.builder.insert(AddOp(lhs, rhs)).res
        if expr.op == "*":
            return self.builder.insert(MulOp(lhs, rhs)).res
        if expr.op == "-":
            return self.builder.insert(SubOp(lhs, rhs)).res
        if expr.op == "<=":
            return self.builder.insert(LessThanEqualOp(lhs, rhs)).res
        raise IRGenError(f"Unknown op {expr.op}")

    def _ir_gen_number_expr(self, expr: NumberExprAST) -> SSAValue:
        return self.builder.insert(ConstantOp(expr.val)).res

    def _ir_gen_variable_expr(self, expr: VariableExprAST) -> SSAValue:
        if self.symbol_table is None or expr.name not in self.symbol_table:
            raise IRGenError(f"Undefined var {expr.name}")
        return self.symbol_table[expr.name]

    def _ir_gen_call_expr(self, expr: CallExprAST) -> SSAValue:
        args = [self._ir_gen_expr(arg) for arg in expr.args]
        # Default return type to i32, similar to how toy uses generic UnrankedTensorType
        ret_type = i32
        # Note: Toy uses GenericCallOp. Aziz uses CallOp which expects explicit type.
        return self.builder.insert(CallOp(expr.callee, args, [ret_type])).res[0]

    def _ir_gen_print_expr(self, expr: PrintExprAST) -> None:
        self.builder.insert(PrintOp(self._ir_gen_expr(expr.arg)))

    def _ir_gen_if_expr(self, expr: IfExprAST) -> SSAValue:
        cond = self._ir_gen_expr(expr.cond)
        # IfOp expects a result type. We need to infer it.
        # Since we are single pass, we might have to assume i32 or infer from 'then' branch?
        # Toy doesn't have IfExpr in the reference file I read?
        # I'll stick to inferring from 'then' branch recursively, or default i32.
        # Let's try to be smart enough to peek? No, _ir_gen_expr returns SSAValue, which has type.
        # But we need to create the IfOp *before* filling regions?
        # Wait, xDSL IfOp usually allows building regions attached to it.
        # Aziz IfOp: `res = result_def(...)`.
        # We can compile `then` expr into a temporary block to get its type?
        # Or just assume i32 for now to match the "simplification" goal.

        if_op = IfOp(cond, i32)  # Defaulting to i32 result
        self.builder.insert(if_op)

        # Then
        cursor = self.builder
        self.builder = Builder(InsertPoint.at_end(if_op.then_region.blocks[0]))
        then_val = self._ir_gen_expr(expr.then_expr)
        self.builder.insert(YieldOp(then_val))

        # Else
        self.builder = Builder(InsertPoint.at_end(if_op.else_region.blocks[0]))
        else_val = self._ir_gen_expr(expr.else_expr)
        self.builder.insert(YieldOp(else_val))

        self.builder = cursor
        return if_op.res

    def _ir_gen_string_expr(self, expr: StringExprAST) -> SSAValue:
        return self.builder.insert(StringConstantOp(expr.val)).res
