from dataclasses import dataclass

from xdsl.builder import Builder, InsertPoint
from xdsl.dialects.builtin import FunctionType, ModuleOp, f64, i32
from xdsl.ir import Attribute, Block, Region, SSAValue
from xdsl.utils.scoped_dict import ScopedDict

from ast_nodes import BinaryExprAST, CallExprAST, ExprAST, FunctionAST, IfExprAST, ModuleAST, NumberExprAST, PrintExprAST, PrototypeAST, StringExprAST, VariableExprAST
from ops import AddOp, CallOp, ConstantOp, FuncOp, IfOp, LessThanEqualOp, MulOp, PrintOp, ReturnOp, StringConstantOp, SubOp, YieldOp, string_type


@dataclass
class IRGen:
    module: ModuleOp = ModuleOp([])
    builder: Builder = None
    symbol_table: ScopedDict[str, SSAValue] | None = None
    function_signatures: dict[str, tuple[list[Attribute], Attribute]] = None  # name -> (arg_types, return_type)

    def __post_init__(self):
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))
        self.function_signatures = {}

    def infer_type_from_expr(self, expr: ExprAST) -> Attribute:
        """Infer the type of an expression without generating IR."""
        if isinstance(expr, NumberExprAST):
            return f64 if isinstance(expr.val, float) else i32
        if isinstance(expr, StringExprAST):
            return string_type
        # For other expressions, default to i32
        return i32

    def collect_function_signatures(self, module_ast: ModuleAST):
        """First pass: collect function signatures from call sites."""
        for op in module_ast.ops:
            if isinstance(op, CallExprAST):
                arg_types = [self.infer_type_from_expr(arg) for arg in op.args]
                # Infer return type - for now, use the type of the first argument
                ret_type = arg_types[0] if arg_types else i32
                self.function_signatures[op.callee] = (arg_types, ret_type)
            elif isinstance(op, PrintExprAST):
                if isinstance(op.arg, CallExprAST):
                    arg_types = [self.infer_type_from_expr(arg) for arg in op.arg.args]
                    ret_type = arg_types[0] if arg_types else i32
                    self.function_signatures[op.arg.callee] = (arg_types, ret_type)

    def ir_gen_module(self, module_ast: ModuleAST) -> ModuleOp:
        # First pass: collect function signatures
        self.collect_function_signatures(module_ast)

        main_body = []
        for op in module_ast.ops:
            if isinstance(op, FunctionAST):
                self.ir_gen_function(op)
            elif isinstance(op, ExprAST):
                main_body.append(op)

        if main_body:
            # create a main function for top-level expressions
            loc = main_body[0].loc
            main_func = FunctionAST(loc, PrototypeAST(loc, "main", []), tuple(main_body))
            self.ir_gen_function(main_func)

        self.module.verify()
        return self.module

    def ir_gen_function(self, func_ast: FunctionAST) -> FuncOp:
        parent_builder, self.symbol_table = self.builder, ScopedDict()

        # Get function signature from first pass, or default to i32
        if func_ast.proto.name in self.function_signatures:
            arg_types, ret_type = self.function_signatures[func_ast.proto.name]
        else:
            arg_types = [i32] * len(func_ast.proto.args)
            ret_type = i32

        block = Block(arg_types=arg_types)
        self.builder = Builder(InsertPoint.at_end(block))

        for name, value in zip(func_ast.proto.args, block.args):
            self.symbol_table[name] = value

        last_val = None
        for expr in func_ast.body:
            last_val = self.ir_gen_expr(expr)

        if not block.ops or not isinstance(block.last_op, ReturnOp):
            val = last_val if last_val else self.builder.insert(ConstantOp(0)).res
            self.builder.insert(ReturnOp(val))

        ret_types = [ret_type] if isinstance(block.last_op, ReturnOp) and block.last_op.operands else []
        func_op = FuncOp(func_ast.proto.name, FunctionType.from_lists(arg_types, ret_types), Region(block))

        self.builder = parent_builder
        return self.builder.insert(func_op)

    def ir_gen_expr(self, expr: ExprAST) -> SSAValue | None:
        if isinstance(expr, BinaryExprAST):
            lhs, rhs = self.ir_gen_expr(expr.lhs), self.ir_gen_expr(expr.rhs)
            if expr.op == "+":
                return self.builder.insert(AddOp(lhs, rhs)).res
            if expr.op == "*":
                return self.builder.insert(MulOp(lhs, rhs)).res
            if expr.op == "-":
                return self.builder.insert(SubOp(lhs, rhs)).res
            if expr.op == "<=":
                return self.builder.insert(LessThanEqualOp(lhs, rhs)).res
            raise Exception(f"Unknown op {expr.op}")

        if isinstance(expr, NumberExprAST):
            return self.builder.insert(ConstantOp(expr.val)).res

        if isinstance(expr, VariableExprAST):
            if expr.name not in self.symbol_table:
                raise Exception(f"Undefined var {expr.name}")
            return self.symbol_table[expr.name]

        if isinstance(expr, CallExprAST):
            args = [self.ir_gen_expr(arg) for arg in expr.args]
            # Get return type from function signature
            ret_type = i32
            if expr.callee in self.function_signatures:
                _, ret_type = self.function_signatures[expr.callee]
            return self.builder.insert(CallOp(expr.callee, args, [ret_type])).res[0]

        if isinstance(expr, PrintExprAST):
            self.builder.insert(PrintOp(self.ir_gen_expr(expr.arg)))
            return None

        if isinstance(expr, IfExprAST):
            cond = self.ir_gen_expr(expr.cond)
            if_op = IfOp(cond)
            self.builder.insert(if_op)

            # Generate Then Block
            cursor = self.builder
            self.builder = Builder(InsertPoint.at_end(if_op.then_region.blocks[0]))
            then_result = self.ir_gen_expr(expr.then_expr)
            self.builder.insert(YieldOp(then_result))

            # Generate Else Block
            self.builder = Builder(InsertPoint.at_end(if_op.else_region.blocks[0]))
            else_result = self.ir_gen_expr(expr.else_expr)
            self.builder.insert(YieldOp(else_result))

            self.builder = cursor
            return if_op.res

        if isinstance(expr, StringExprAST):
            return self.builder.insert(StringConstantOp(expr.val)).res

        raise Exception(f"Unknown expr: {expr}")
