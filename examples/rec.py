# $ PYTHONPATH=. uv run ./examples/rec.py
#
# /// script
# requires-python = "==3.14"
# dependencies = [
#     "xdsl==0.55.4",
# ]
# ///

from src.dialects import aziz
from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects.builtin import FunctionType, ModuleOp, i32
from xdsl.ir import Block, Region, SSAValue


@ModuleOp
@Builder.implicit_region
def module_op():

    # (defun get_msg () "hello world!")
    @Builder.implicit_region
    def get_msg():
        s = aziz.StringConstantOp("hello world!").res
        aziz.ReturnOp(s)

    aziz.FuncOp("get_msg", FunctionType.from_lists([], [aziz.StringType()]), get_msg)

    # (defun factorial (n) (if (<= n 1) 1 (* n (factorial (- n 1)))))
    @Builder.implicit_region([i32])
    def factorial(args: tuple[SSAValue, ...]):
        n = args[0]
        one = aziz.ConstantOp(1).res
        cond = aziz.LessThanEqualOp(n, one).res

        then_block = Block()
        with ImplicitBuilder(then_block):  # new scope
            res_then = aziz.ConstantOp(1).res
            aziz.YieldOp(res_then)

        else_block = Block()
        with ImplicitBuilder(else_block):
            one_c = aziz.ConstantOp(1).res
            sub_res = aziz.SubOp(n, one_c).res
            call_res = aziz.CallOp("factorial", [sub_res], [i32]).res[0]
            mul_res = aziz.MulOp(n, call_res).res
            aziz.YieldOp(mul_res)

        if_res = aziz.IfOp(cond, i32, [Region(then_block), Region(else_block)]).res
        aziz.ReturnOp(if_res)

    aziz.FuncOp("factorial", FunctionType.from_lists([i32], [i32]), factorial)

    # (print (get_msg))
    # (print (factorial 5))
    @Builder.implicit_region
    def main():
        msg_call = aziz.CallOp("get_msg", [], [aziz.StringType()])
        aziz.PrintOp(msg_call.res[0])

        five = aziz.ConstantOp(5).res
        fact_call = aziz.CallOp("factorial", [five], [i32])
        aziz.PrintOp(fact_call.res[0])

        # exit code 0
        zero = aziz.ConstantOp(0).res
        aziz.ReturnOp(zero)

    aziz.FuncOp("main", FunctionType.from_lists([], [i32]), main)


print(module_op)
