from typing import Any

from xdsl.interpreter import Interpreter, InterpreterFunctions, ReturnedValues, impl, impl_callable, impl_terminator, register_impls

import ops


@register_impls
class AzizFunctions(InterpreterFunctions):
    @impl(ops.AddOp)
    def run_add(self, i: Interpreter, op: ops.AddOp, args: tuple[Any, ...]):
        return (args[0] + args[1],)

    @impl(ops.MulOp)
    def run_mul(self, i: Interpreter, op: ops.MulOp, args: tuple[Any, ...]):
        return (args[0] * args[1],)

    @impl(ops.SubOp)
    def run_sub(self, i: Interpreter, op: ops.SubOp, args: tuple[Any, ...]):
        return (args[0] - args[1],)

    @impl(ops.LessThanEqualOp)
    def run_le(self, i: Interpreter, op: ops.LessThanEqualOp, args: tuple[Any, ...]):
        return (1 if args[0] <= args[1] else 0,)

    @impl(ops.ConstantOp)
    def run_constant(self, i: Interpreter, op: ops.ConstantOp, args: tuple[Any, ...]):
        return (op.attributes["value"].value.data,)

    @impl(ops.PrintOp)
    def run_print(self, i: Interpreter, op: ops.PrintOp, args: tuple[Any, ...]):
        print(args[0])
        return ()

    @impl(ops.CallOp)
    def run_call(self, i: Interpreter, op: ops.CallOp, args: tuple[Any, ...]):
        return i.call_op(op.attributes["callee"].string_value(), args)

    @impl_terminator(ops.ReturnOp)
    def run_return(self, i: Interpreter, op: ops.ReturnOp, args: tuple[Any, ...]):
        return ReturnedValues(args), ()

    @impl_terminator(ops.YieldOp)
    def run_yield(self, i: Interpreter, op: ops.YieldOp, args: tuple[Any, ...]):
        return ReturnedValues(args), ()

    @impl(ops.IfOp)
    def run_if(self, i: Interpreter, op: ops.IfOp, args: tuple[Any, ...]):
        cond = args[0]
        if cond:
            return i.run_ssacfg_region(op.then_region, (), "then")
        else:
            return i.run_ssacfg_region(op.else_region, (), "else")

    @impl_callable(ops.FuncOp)
    def run_func(self, i: Interpreter, op: ops.FuncOp, args: tuple[Any, ...]):
        return i.run_ssacfg_region(op.body, args, op.attributes["sym_name"].data)
