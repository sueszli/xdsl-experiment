from dataclasses import dataclass

from dialects.aziz import AddOp, CallOp, CastIntToFloatOp, ConstantOp, FuncOp, IfOp, LessThanEqualOp, MulOp, PrintOp, ReturnOp, StringConstantOp, StringType, SubOp, YieldOp
from xdsl.builder import Builder, InsertPoint
from xdsl.dialects.builtin import FunctionType, ModuleOp, f64, i32
from xdsl.ir import Attribute, Block, Region, SSAValue
from xdsl.utils.scoped_dict import ScopedDict

from .ast_nodes import BinaryExprAST, CallExprAST, ExprAST, FunctionAST, IfExprAST, ModuleAST, NumberExprAST, PrintExprAST, PrototypeAST, StringExprAST, VariableExprAST


class IRGenError(Exception):
    pass


@dataclass(init=False)
class IRGen:
    module: ModuleOp  # the mlir module being built
    builder: Builder  # position to insert new ops
    symbol_table: ScopedDict[str, SSAValue] | None = None  # var name -> SSAValue within current scope (dropped on exit)
    inferred_arg_types: dict[str, list[Attribute]]  # func name -> inferred arg types

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))
        self.inferred_arg_types = {}

    def ir_gen_module(self, module_ast: ModuleAST) -> ModuleOp:
        functions = [op for op in module_ast.ops if isinstance(op, FunctionAST)]

        # implicit main function logic
        main_body = [op for op in module_ast.ops if isinstance(op, ExprAST)]
        if main_body:
            loc = main_body[0].loc
            main_func = FunctionAST(loc, PrototypeAST(loc, "main", []), tuple(main_body))
            functions.append(main_func)

        # first pass: infer types from calls
        # enforce functions are called with consistent types. promotes i32 -> f64 if needed.
        self._infer_types_in_module(module_ast)

        # second pass: declare all functions to allow forward references
        # allows for forward references (calling a function before it is defined).
        for func_ast in functions:
            self._declare_function(func_ast)

        # third pass: generate code for each function
        for func_ast in functions:
            self._ir_gen_function(func_ast)

        # verify types in module
        self.module.verify()
        return self.module

    def _infer_types_in_module(self, module_ast: ModuleAST) -> None:
        signatures: dict[str, list[list[Attribute]]] = {}

        def get_literal_type(node: ExprAST) -> Attribute | None:
            if isinstance(node, StringExprAST):
                return StringType()
            if isinstance(node, NumberExprAST):
                return f64 if isinstance(node.val, float) else i32
            return None

        def visit(node: object):
            if isinstance(node, CallExprAST):
                arg_types = [get_literal_type(arg) for arg in node.args]
                if all(t is not None for t in arg_types):
                    valid_types = [t for t in arg_types if t is not None]
                    if node.callee not in signatures:
                        signatures[node.callee] = []
                    signatures[node.callee].append(valid_types)

            if hasattr(node, "__dataclass_fields__"):
                for field in node.__dataclass_fields__:
                    visit(getattr(node, field))
            elif isinstance(node, (list, tuple)):
                for item in node:
                    visit(item)

        # traverse the AST to collect call signatures
        visit(module_ast)

        # resolve signatures, check for consistency
        self.inferred_arg_types = {}
        for name, sigs in signatures.items():
            if not sigs:
                continue

            resolved_sig = list(sigs[0])
            for sig in sigs[1:]:
                if len(sig) != len(resolved_sig):
                    raise IRGenError(f"function '{name}' called with inconsistent arity. {len(resolved_sig)} vs {len(sig)}")

                for i, (t1, t2) in enumerate(zip(resolved_sig, sig)):
                    if t1 == t2:
                        continue
                    if {t1, t2} == {i32, f64}:
                        resolved_sig[i] = f64  # promote: i32 -> f64
                    else:
                        raise IRGenError(f"function '{name}' called with inconsistent types at arg {i}. can't reconcile {t1} and {t2}.")

            self.inferred_arg_types[name] = resolved_sig

    def _declare_function(self, func_ast: FunctionAST) -> None:
        function_name = func_ast.proto.name
        param_count_int = len(func_ast.proto.args)

        arg_types = self.inferred_arg_types.get(function_name, [i32] * param_count_int)  # default to i32
        if len(arg_types) != param_count_int:
            arg_types = [i32] * param_count_int  # default to i32 if type inference failed

        return_types = [i32]  # default return type

        func_type = FunctionType.from_lists(inputs=arg_types, outputs=return_types)
        region = Region(Block(arg_types=arg_types))

        is_private = function_name != "main"  # for dead code elimination
        func_op = FuncOp(function_name, func_type, region, private=is_private)
        self.module.body.blocks[0].add_op(func_op)

    def _ir_gen_function(self, func_ast: FunctionAST) -> None:
        function_name = func_ast.proto.name

        func_op = next((op for op in self.module.body.blocks[0].ops if isinstance(op, FuncOp) and op.sym_name.data == function_name), None)
        if not func_op:
            raise IRGenError(f"function {function_name} not declared")

        block = func_op.body.blocks[0]

        # scope management with builder/symbol table
        parent_builder = self.builder
        self.symbol_table = ScopedDict()
        self.builder = Builder(InsertPoint.at_end(block))

        try:
            # populate args in symbol table
            for name, value in zip(func_ast.proto.args, block.args):
                if name in self.symbol_table:
                    continue
                self.symbol_table[name] = value

            # gen body
            last_val = None
            for expr in func_ast.body:
                last_val = self._ir_gen_expr(expr)

            # last expression implicitly returns 0
            if not (block.ops and isinstance(block.last_op, ReturnOp)):
                val = last_val if last_val is not None else self.builder.insert(ConstantOp(0)).res
                self.builder.insert(ReturnOp(val))

            # match return type to last expression
            return_types = []
            if block.last_op.input:
                return_types = [block.last_op.input.type]
            current_outputs = func_op.function_type.outputs.data
            if list(current_outputs) != return_types:
                new_type = FunctionType.from_lists(inputs=func_op.function_type.inputs.data, outputs=return_types)
                func_op.function_type = new_type

        finally:
            self.symbol_table = None
            self.builder = parent_builder

    def _ir_gen_expr(self, expr: ExprAST) -> SSAValue | None:
        match expr:
            case BinaryExprAST():
                return self._ir_gen_binary_expr(expr)
            case NumberExprAST():
                return self._ir_gen_number_expr(expr)
            case VariableExprAST():
                return self._ir_gen_variable_expr(expr)
            case CallExprAST():
                return self._ir_gen_call_expr(expr)
            case PrintExprAST():
                self._ir_gen_print_expr(expr)
                return None
            case IfExprAST():
                return self._ir_gen_if_expr(expr)
            case StringExprAST():
                return self._ir_gen_string_expr(expr)
            case _:
                raise IRGenError(f"unknown expr type: {expr}")

    def _ir_gen_binary_expr(self, expr: BinaryExprAST) -> SSAValue:
        lhs = self._ir_gen_expr(expr.lhs)
        rhs = self._ir_gen_expr(expr.rhs)
        op_map = {
            "+": AddOp,
            "-": SubOp,
            "*": MulOp,
            "<=": LessThanEqualOp,
        }
        if op_class := op_map.get(expr.op):
            return self.builder.insert(op_class(lhs, rhs)).res
        raise IRGenError(f"unknown binary op {expr.op}")

    def _ir_gen_number_expr(self, expr: NumberExprAST) -> SSAValue:
        return self.builder.insert(ConstantOp(expr.val)).res

    def _ir_gen_variable_expr(self, expr: VariableExprAST) -> SSAValue:
        if self.symbol_table is None or expr.name not in self.symbol_table:
            raise IRGenError(f"undefined var {expr.name}")
        return self.symbol_table[expr.name]

    def _ir_gen_call_expr(self, expr: CallExprAST) -> SSAValue:
        args = [self._ir_gen_expr(arg) for arg in expr.args]
        callee_op = next((op for op in self.module.body.blocks[0].ops if isinstance(op, FuncOp) and op.sym_name.data == expr.callee), None)
        if not callee_op:
            raise IRGenError(f"unknown function called: {expr.callee}")
        if not callee_op.function_type.outputs.data:
            raise IRGenError(f"function {expr.callee} returns void but used as expression")

        expected_types = callee_op.function_type.inputs.data
        if len(args) != len(expected_types):
            raise IRGenError(f"function {expr.callee} expects {len(expected_types)} args but got {len(args)}")

        final_args = []
        for i, (arg_val, expected_type) in enumerate(zip(args, expected_types)):
            if arg_val.type == expected_type:
                final_args.append(arg_val)
                continue

            if arg_val.type == i32 and expected_type == f64:
                cast = self.builder.insert(CastIntToFloatOp(arg_val))  # promote i32 -> f64
                final_args.append(cast.res)
                continue

            raise IRGenError(f"type mismatch at arg {i} for {expr.callee}: expected {expected_type} but got {arg_val.type}")

        ret_type = callee_op.function_type.outputs.data[0]  # assume single ret val
        return self.builder.insert(CallOp(expr.callee, final_args, [ret_type])).res[0]

    def _ir_gen_print_expr(self, expr: PrintExprAST) -> None:
        self.builder.insert(PrintOp(self._ir_gen_expr(expr.arg)))

    def _ir_gen_if_expr(self, expr: IfExprAST) -> SSAValue:
        cond = self._ir_gen_expr(expr.cond)

        # then block
        then_block = Block()
        then_region = Region(then_block)

        cursor_snapshot = self.builder

        self.builder = Builder(InsertPoint.at_end(then_block))
        then_val = self._ir_gen_expr(expr.then_expr)
        self.builder.insert(YieldOp(then_val))

        # else block
        else_block = Block()
        else_region = Region(else_block)

        self.builder = Builder(InsertPoint.at_end(else_block))
        else_val = self._ir_gen_expr(expr.else_expr)
        self.builder.insert(YieldOp(else_val))

        self.builder = cursor_snapshot

        if_op = IfOp(cond, then_val.type, [then_region, else_region])
        self.builder.insert(if_op)
        return if_op.res

    def _ir_gen_string_expr(self, expr: StringExprAST) -> SSAValue:
        return self.builder.insert(StringConstantOp(expr.val)).res
