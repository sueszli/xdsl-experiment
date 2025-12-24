from dataclasses import dataclass

from ast_nodes import BinaryExprAST, CallExprAST, ExprAST, FunctionAST, IfExprAST, ModuleAST, NumberExprAST, PrintExprAST, PrototypeAST, StringExprAST, VariableExprAST
from aziz import AddOp, CallOp, CastIntToFloatOp, ConstantOp, FuncOp, IfOp, LessThanEqualOp, MulOp, PrintOp, ReturnOp, StringConstantOp, StringType, SubOp, YieldOp
from xdsl.builder import Builder, InsertPoint
from xdsl.dialects.builtin import FunctionType, ModuleOp, f64, i32
from xdsl.ir import Attribute, Block, Region, SSAValue
from xdsl.utils.scoped_dict import ScopedDict


class IRGenError(Exception):
    pass


@dataclass(init=False)
class IRGen:
    module: ModuleOp
    builder: Builder  # to insert ops into the current block
    symbol_table: ScopedDict[str, SSAValue] | None = None  # var name -> SSAValue in current scope (drops on exit)

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))

    def ir_gen_module(self, module_ast: ModuleAST) -> ModuleOp:
        functions = [op for op in module_ast.ops if isinstance(op, FunctionAST)]

        # implicit main function
        main_body = [op for op in module_ast.ops if isinstance(op, ExprAST)]
        if main_body:
            loc = main_body[0].loc
            main_func = FunctionAST(loc, PrototypeAST(loc, "main", []), tuple(main_body))
            functions.append(main_func)

        # first pass: collect call signatures and declare functions
        signatures = self._collect_call_signatures(module_ast)
        inferred_types = self._resolve_signatures(signatures)
        for func in functions:
            self._declare_function(func, inferred_types)

        # second pass: generate function bodies
        for func in functions:
            self._ir_gen_function(func)

        # verify type soundness
        self.module.verify()
        return self.module

    def _collect_call_signatures(self, module_ast: ModuleAST) -> dict[str, list[list[Attribute]]]:
        # function name -> list of argument type lists, inferred from calls
        signatures: dict[str, list[list[Attribute]]] = {}

        def _type_of(node: ExprAST) -> Attribute | None:
            match node:
                case StringExprAST():
                    return StringType()
                case NumberExprAST(val=float()):
                    return f64
                case NumberExprAST(val=int()):
                    return i32
                case _:
                    return None

        def visit(node: object):
            if isinstance(node, CallExprAST):
                arg_types = [_type_of(arg) for arg in node.args]

                if all(t is not None for t in arg_types):
                    valid_types = [t for t in arg_types if t is not None]
                    signatures.setdefault(node.callee, []).append(valid_types)  # update

            if hasattr(node, "__dataclass_fields__"):
                for field in node.__dataclass_fields__:
                    visit(getattr(node, field))
            elif isinstance(node, (list, tuple)):
                for item in node:
                    visit(item)

        visit(module_ast)
        return signatures

    def _resolve_signatures(self, signatures: dict[str, list[list[Attribute]]]) -> dict[str, list[Attribute]]:
        resolved: dict[str, list[Attribute]] = {}

        for name, sigs in signatures.items():
            if not sigs:
                continue

            final_sig = list(sigs[0])

            for sig in sigs[1:]:
                if len(sig) != len(final_sig):
                    raise IRGenError(f"function '{name}' called with inconsistent arity. {len(final_sig)} vs {len(sig)}")

                for i, (t1, t2) in enumerate(zip(final_sig, sig)):
                    if t1 == t2:
                        continue

                    if {t1, t2} == {i32, f64}:
                        final_sig[i] = f64  # promote
                    else:
                        raise IRGenError(f"function '{name}' called with inconsistent types at arg {i}. can't reconcile {t1} and {t2}.")

            resolved[name] = final_sig

        return resolved

    def _declare_function(self, func_ast: FunctionAST, inferred_types: dict[str, list[Attribute]]) -> None:
        name = func_ast.proto.name
        arg_names = func_ast.proto.args
        arg_count = len(arg_names)

        # argument types are i32 by default unless inferred otherwise
        arg_types = inferred_types.get(name, [i32] * arg_count)
        if len(arg_types) != arg_count:
            arg_types = [i32] * arg_count

        # return type (just a single i32 for now)
        return_types = [i32]

        func_type = FunctionType.from_lists(inputs=arg_types, outputs=return_types)
        block = Block(arg_types=arg_types)
        region = Region(block)
        is_private = name != "main"  # required for dead code elimination
        func_op = FuncOp(name, func_type, region, private=is_private)
        self.builder.insert(func_op)

    def _ir_gen_function(self, func_ast: FunctionAST) -> None:
        name = func_ast.proto.name

        func_op = next((op for op in self.module.body.blocks[0].ops if isinstance(op, FuncOp) and op.sym_name.data == name), None)
        if not func_op:
            raise IRGenError(f"function {name} not declared")

        block = func_op.body.blocks[0]

        parent_builder = self.builder
        self.builder = Builder(InsertPoint.at_end(block))
        self.symbol_table = ScopedDict()

        try:
            for arg_name, arg_val in zip(func_ast.proto.args, block.args):
                self.symbol_table[arg_name] = arg_val  # bind arguments

            # generate body
            last_val = None
            for expr in func_ast.body:
                last_val = self._ir_gen_expr(expr)

            # ensure return
            if not (block.ops and isinstance(block.last_op, ReturnOp)):
                if last_val:
                    self.builder.insert(ReturnOp(last_val))
                else:
                    zero = self.builder.insert(ConstantOp(0)).res
                    self.builder.insert(ReturnOp(zero))

            # update return type
            if block.ops and isinstance(block.last_op, ReturnOp):
                return_op = block.last_op
                actual_ret_types = [return_op.input.type] if return_op.input else []
                current_ret_types = list(func_op.function_type.outputs.data)

                if current_ret_types != actual_ret_types:
                    new_type = FunctionType.from_lists(inputs=func_op.function_type.inputs.data, outputs=actual_ret_types)
                    func_op.function_type = new_type

        finally:
            # no dangling symbol table or builder on error
            self.symbol_table = None
            self.builder = parent_builder

    def _ir_gen_expr(self, expr: ExprAST) -> SSAValue | None:
        match expr:
            case BinaryExprAST():
                lhs, rhs = self._ir_gen_expr(expr.lhs), self._ir_gen_expr(expr.rhs)
                ops = {"+": AddOp, "-": SubOp, "*": MulOp, "<=": LessThanEqualOp}
                if cls := ops.get(expr.op):
                    return self.builder.insert(cls(lhs, rhs)).res
                raise IRGenError(f"unknown binary op {expr.op}")

            case NumberExprAST():
                return self.builder.insert(ConstantOp(expr.val)).res

            case VariableExprAST():
                if self.symbol_table is None or expr.name not in self.symbol_table:
                    raise IRGenError(f"undefined var {expr.name}")
                return self.symbol_table[expr.name]

            case PrintExprAST():
                val = self._ir_gen_expr(expr.arg)
                self.builder.insert(PrintOp(val))
                return None

            case StringExprAST():
                return self.builder.insert(StringConstantOp(expr.val)).res

            case IfExprAST():
                return self._ir_gen_if(expr)

            case CallExprAST():
                return self._ir_gen_call(expr)

            case _:
                raise IRGenError(f"unknown expr type: {expr}")

    def _ir_gen_if(self, expr: IfExprAST) -> SSAValue:
        cond_val = self._ir_gen_expr(expr.cond)

        then_block = Block()
        else_block = Block()

        original_builder = self.builder

        self.builder = Builder(InsertPoint.at_end(then_block))
        then_val = self._ir_gen_expr(expr.then_expr)
        self.builder.insert(YieldOp(then_val))

        self.builder = Builder(InsertPoint.at_end(else_block))
        else_val = self._ir_gen_expr(expr.else_expr)
        self.builder.insert(YieldOp(else_val))

        self.builder = original_builder

        if_op = IfOp(cond_val, then_val.type, [Region(then_block), Region(else_block)])
        return self.builder.insert(if_op).res

    def _ir_gen_call(self, expr: CallExprAST) -> SSAValue:
        callee_op = next((op for op in self.module.body.blocks[0].ops if isinstance(op, FuncOp) and op.sym_name.data == expr.callee), None)

        if not callee_op:
            raise IRGenError(f"unknown function called: {expr.callee}")

        expected_inputs = callee_op.function_type.inputs.data
        if len(expr.args) != len(expected_inputs):
            raise IRGenError(f"function {expr.callee} expects {len(expected_inputs)} args but got {len(expr.args)}")

        args_values = [self._ir_gen_expr(arg) for arg in expr.args]
        final_args = self._cast_call_arguments(expr.callee, args_values, expected_inputs)  # must match expected types

        if not callee_op.function_type.outputs.data:
            raise IRGenError(f"function {expr.callee} returns void but used as expression")

        ret_type = callee_op.function_type.outputs.data[0]
        call_op = CallOp(expr.callee, final_args, [ret_type])
        return self.builder.insert(call_op).res[0]

    def _cast_call_arguments(self, func_name: str, args: list[SSAValue], expected: list[Attribute]) -> list[SSAValue]:
        final_args = []
        for i, (arg, type_expected) in enumerate(zip(args, expected)):
            if arg.type == type_expected:
                final_args.append(arg)
            elif arg.type == i32 and type_expected == f64:
                cast = self.builder.insert(CastIntToFloatOp(arg))
                final_args.append(cast.res)
            else:
                raise IRGenError(f"type mismatch at arg {i} for {func_name}: expected {type_expected} but got {arg.type}")
        return final_args
