from collections.abc import Iterable
from dataclasses import dataclass

from xdsl.builder import Builder, InsertPoint
from xdsl.dialects.builtin import FunctionType, ModuleOp, i32
from xdsl.ir import Block, Region, SSAValue
from xdsl.utils.scoped_dict import ScopedDict

from dialects.aziz import AddOp, CallOp, ConstantOp, FuncOp, IfOp, LessThanEqualOp, MulOp, PrintOp, ReturnOp, StringConstantOp, SubOp, YieldOp

from .ast_nodes import BinaryExprAST, CallExprAST, ExprAST, FunctionAST, IfExprAST, ModuleAST, NumberExprAST, PrintExprAST, PrototypeAST, StringExprAST, VariableExprAST


class IRGenError(Exception):
    pass


@dataclass(init=False)
class IRGen:
    module: ModuleOp  # the MLIR module being built
    builder: Builder  # position to insert new ops
    symbol_table: ScopedDict[str, SSAValue] | None = None  # var name -> SSAValue within current scope (dropped on exit)

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))

    def ir_gen_module(self, module_ast: ModuleAST) -> ModuleOp:
        functions = [op for op in module_ast.ops if isinstance(op, FunctionAST)]

        # implicit main function logic
        main_body = [op for op in module_ast.ops if isinstance(op, ExprAST)]
        if main_body:
            loc = main_body[0].loc
            main_func = FunctionAST(loc, PrototypeAST(loc, "main", []), tuple(main_body))
            functions.append(main_func)

        # first pass: declare all functions to allow forward references
        for func_ast in functions:
            self._declare_function(func_ast)

        # second pass: generate code for each function
        for func_ast in functions:
            self._ir_gen_function(func_ast)

        # verify types in module
        self.module.verify()
        return self.module

    def _declare_function(self, func_ast: FunctionAST):
        # create FuncOp with correct name and default types (i32), but empty body
        arg_types = [i32] * len(func_ast.proto.args)
        return_types = [i32]

        func_type = FunctionType.from_lists(inputs=arg_types, outputs=return_types)
        region = Region(Block(arg_types=arg_types))

        func_op = FuncOp(func_ast.proto.name, func_type, region)
        self.module.body.blocks[0].add_op(func_op)

    def _ir_gen_function(self, func_ast: FunctionAST):
        func_op = self._get_func_op(func_ast.proto.name)
        if not func_op:
            raise IRGenError(f"function {func_ast.proto.name} not declared")

        block = func_op.body.blocks[0]

        # save current builder and symbol table
        parent_builder = self.builder
        self.symbol_table = ScopedDict()
        self.builder = Builder(InsertPoint.at_end(block))

        # init argument variables in symbol table
        for name, value in zip(func_ast.proto.args, block.args):
            self._declare(name, value)

        # generate body
        last_val = self._ir_gen_expr_list(func_ast.body)

        return_types = []
        if not (block.ops and isinstance(block.last_op, ReturnOp)):
            # implicit return: type of last op or default to 0
            val = last_val if last_val is not None else self.builder.insert(ConstantOp(0)).res
            self.builder.insert(ReturnOp(val))
        if block.last_op.input:
            # explicit return: type of last op
            return_types = [block.last_op.input.type]

        # update function signature if necessary
        current_return_types = func_op.function_type.outputs.data
        if list(current_return_types) != return_types:
            func_type = FunctionType.from_lists(inputs=func_op.function_type.inputs.data, outputs=return_types)
            func_op.function_type = func_type

        # restore state
        self.symbol_table = None
        self.builder = parent_builder

    def _get_func_op(self, name: str) -> FuncOp | None:
        # search for function in module
        for op in self.module.body.blocks[0].ops:
            if isinstance(op, FuncOp) and op.sym_name.data == name:
                return op
        return None

    def _declare(self, var: str, value: SSAValue) -> bool:
        assert self.symbol_table is not None
        if var in self.symbol_table:
            return False
        self.symbol_table[var] = value
        return True

    def _ir_gen_expr_list(self, exprs: Iterable[ExprAST]) -> SSAValue | None:
        last_val = None
        for expr in exprs:
            last_val = self._ir_gen_expr(expr)
        return last_val

    def _ir_gen_expr(self, expr: ExprAST) -> SSAValue | None:
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
            return None
        if isinstance(expr, IfExprAST):
            return self._ir_gen_if_expr(expr)
        if isinstance(expr, StringExprAST):
            return self._ir_gen_string_expr(expr)

        raise IRGenError(f"unknown expr: {expr}")

    def _ir_gen_binary_expr(self, expr: BinaryExprAST) -> SSAValue:
        lhs = self._ir_gen_expr(expr.lhs)
        rhs = self._ir_gen_expr(expr.rhs)

        if expr.op == "+":
            return self.builder.insert(AddOp(lhs, rhs)).res
        if expr.op == "-":
            return self.builder.insert(SubOp(lhs, rhs)).res
        if expr.op == "*":
            return self.builder.insert(MulOp(lhs, rhs)).res
        if expr.op == "<=":
            return self.builder.insert(LessThanEqualOp(lhs, rhs)).res
        raise IRGenError(f"unknown op {expr.op}")

    def _ir_gen_number_expr(self, expr: NumberExprAST) -> SSAValue:
        return self.builder.insert(ConstantOp(expr.val)).res

    def _ir_gen_variable_expr(self, expr: VariableExprAST) -> SSAValue:
        if self.symbol_table is None or expr.name not in self.symbol_table:
            raise IRGenError(f"undefined var {expr.name}")
        return self.symbol_table[expr.name]

    def _ir_gen_call_expr(self, expr: CallExprAST) -> SSAValue:
        args = [self._ir_gen_expr(arg) for arg in expr.args]

        callee_op = self._get_func_op(expr.callee)
        if not callee_op:
            raise IRGenError(f"unknown function called: {expr.callee}")

        if not callee_op.function_type.outputs.data:
            raise IRGenError(f"function {expr.callee} returns void but used as expression")

        ret_type = callee_op.function_type.outputs.data[0]

        return self.builder.insert(CallOp(expr.callee, args, [ret_type])).res[0]

    def _ir_gen_print_expr(self, expr: PrintExprAST) -> None:
        self.builder.insert(PrintOp(self._ir_gen_expr(expr.arg)))

    def _ir_gen_if_expr(self, expr: IfExprAST) -> SSAValue:
        cond = self._ir_gen_expr(expr.cond)

        # then
        then_block = Block()
        then_region = Region(then_block)
        cursor = self.builder
        self.builder = Builder(InsertPoint.at_end(then_block))
        then_val = self._ir_gen_expr(expr.then_expr)
        self.builder.insert(YieldOp(then_val))

        # else
        else_block = Block()
        else_region = Region(else_block)
        self.builder = Builder(InsertPoint.at_end(else_block))
        else_val = self._ir_gen_expr(expr.else_expr)
        self.builder.insert(YieldOp(else_val))

        # restore builder
        self.builder = cursor

        if_op = IfOp(cond, then_val.type, [then_region, else_region])
        self.builder.insert(if_op)
        return if_op.res

    def _ir_gen_string_expr(self, expr: StringExprAST) -> SSAValue:
        return self.builder.insert(StringConstantOp(expr.val)).res
