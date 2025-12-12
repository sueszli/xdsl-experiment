from dataclasses import dataclass

from xdsl.builder import Builder, InsertPoint
from xdsl.dialects.builtin import FunctionType, ModuleOp, i32
from xdsl.ir import Block, Region, SSAValue
from xdsl.utils.scoped_dict import ScopedDict

from ast_nodes import BinaryExprAST, CallExprAST, ExprAST, FunctionAST, ModuleAST, NumberExprAST, PrintExprAST, PrototypeAST, VariableExprAST
from ops import AddOp, CallOp, ConstantOp, FuncOp, MulOp, PrintOp, ReturnOp


@dataclass
class IRGen:
    module: ModuleOp = ModuleOp([])
    builder: Builder = None
    symbol_table: ScopedDict[str, SSAValue] | None = None

    def __post_init__(self):
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))

    def ir_gen_module(self, module_ast: ModuleAST) -> ModuleOp:
        main_body = []
        for op in module_ast.ops:
            if isinstance(op, FunctionAST):
                self.ir_gen_function(op)
            elif isinstance(op, ExprAST):
                main_body.append(op)

        if main_body:
            # Synthesize a main function for top-level expressions
            loc = main_body[0].loc
            main_func = FunctionAST(loc, PrototypeAST(loc, "main", []), tuple(main_body))
            self.ir_gen_function(main_func)

        self.module.verify()
        return self.module

    def ir_gen_function(self, func_ast: FunctionAST) -> FuncOp:
        parent_builder, self.symbol_table = self.builder, ScopedDict()
        block = Block(arg_types=[i32] * len(func_ast.proto.args))
        self.builder = Builder(InsertPoint.at_end(block))

        for name, value in zip(func_ast.proto.args, block.args):
            self.symbol_table[name] = value

        last_val = None
        for expr in func_ast.body:
            last_val = self.ir_gen_expr(expr)

        if not block.ops or not isinstance(block.last_op, ReturnOp):
            val = last_val if last_val else self.builder.insert(ConstantOp(0)).res
            self.builder.insert(ReturnOp(val))

        ret_types = [i32] if isinstance(block.last_op, ReturnOp) and block.last_op.operands else []
        func_op = FuncOp(func_ast.proto.name, FunctionType.from_lists([i32] * len(func_ast.proto.args), ret_types), Region(block))

        self.builder = parent_builder
        return self.builder.insert(func_op)

    def ir_gen_expr(self, expr: ExprAST) -> SSAValue | None:
        if isinstance(expr, BinaryExprAST):
            lhs, rhs = self.ir_gen_expr(expr.lhs), self.ir_gen_expr(expr.rhs)
            if expr.op == "+":
                return self.builder.insert(AddOp(lhs, rhs)).res
            if expr.op == "*":
                return self.builder.insert(MulOp(lhs, rhs)).res
            raise Exception(f"Unknown op {expr.op}")

        if isinstance(expr, NumberExprAST):
            return self.builder.insert(ConstantOp(expr.val)).res

        if isinstance(expr, VariableExprAST):
            if expr.name not in self.symbol_table:
                raise Exception(f"Undefined var {expr.name}")
            return self.symbol_table[expr.name]

        if isinstance(expr, CallExprAST):
            args = [self.ir_gen_expr(arg) for arg in expr.args]
            return self.builder.insert(CallOp(expr.callee, args, [i32])).res[0]

        if isinstance(expr, PrintExprAST):
            self.builder.insert(PrintOp(self.ir_gen_expr(expr.arg)))
            return None

        raise Exception(f"Unknown expr: {expr}")
