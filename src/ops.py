from xdsl.dialects.builtin import FunctionType, IntegerAttr, IntegerType, StringAttr, SymbolNameConstraint, SymbolRefAttr, i32
from xdsl.ir import Attribute, Block, Dialect, Region, SSAValue
from xdsl.irdl import IRDLOperation, attr_def, irdl_op_definition, operand_def, opt_operand_def, region_def, result_def, traits_def, var_operand_def, var_result_def
from xdsl.traits import CallableOpInterface, HasParent, IsTerminator, Pure, SymbolOpInterface


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name, traits = "aziz.constant", traits_def(Pure())
    value, res = attr_def(IntegerAttr[IntegerType]), result_def(IntegerType)

    def __init__(self, value: int):
        super().__init__(result_types=[i32], attributes={"value": IntegerAttr(value, i32)})


@irdl_op_definition
class AddOp(IRDLOperation):
    name, traits = "aziz.add", traits_def(Pure())
    lhs, rhs, res = operand_def(IntegerType), operand_def(IntegerType), result_def(IntegerType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[i32])


@irdl_op_definition
class MulOp(IRDLOperation):
    name, traits = "aziz.mul", traits_def(Pure())
    lhs, rhs, res = operand_def(IntegerType), operand_def(IntegerType), result_def(IntegerType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[i32])


@irdl_op_definition
class PrintOp(IRDLOperation):
    name = "aziz.print"
    input = operand_def(IntegerType)

    def __init__(self, input: SSAValue):
        super().__init__(operands=[input])


class FuncOpCallableInterface(CallableOpInterface):
    @classmethod
    def get_callable_region(cls, op: IRDLOperation) -> Region:
        return op.body

    @classmethod
    def get_argument_types(cls, op: IRDLOperation) -> tuple[Attribute, ...]:
        return op.function_type.inputs.data

    @classmethod
    def get_result_types(cls, op: IRDLOperation) -> tuple[Attribute, ...]:
        return op.function_type.outputs.data


@irdl_op_definition
class FuncOp(IRDLOperation):
    name = "aziz.func"
    body, sym_name, function_type = region_def("single_block"), attr_def(SymbolNameConstraint()), attr_def(FunctionType)
    traits = traits_def(SymbolOpInterface(), FuncOpCallableInterface())

    def __init__(self, name: str, ftype: FunctionType, region: Region):
        super().__init__(attributes={"sym_name": StringAttr(name), "function_type": ftype}, regions=[region])


@irdl_op_definition
class ReturnOp(IRDLOperation):
    name = "aziz.return"
    input, traits = opt_operand_def(IntegerType), traits_def(IsTerminator(), HasParent(FuncOp))

    def __init__(self, input: SSAValue | None = None):
        super().__init__(operands=[input] if input else [])


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "aziz.call"
    callee, arguments, res = attr_def(SymbolRefAttr), var_operand_def(IntegerType), var_result_def(IntegerType)

    def __init__(self, callee: str, args: list[SSAValue], ret_type: list[Attribute]):
        super().__init__(operands=[args], result_types=[ret_type], attributes={"callee": SymbolRefAttr(callee)})


@irdl_op_definition
class SubOp(IRDLOperation):
    name, traits = "aziz.sub", traits_def(Pure())
    lhs, rhs, res = operand_def(IntegerType), operand_def(IntegerType), result_def(IntegerType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[i32])


@irdl_op_definition
class LessThanEqualOp(IRDLOperation):
    name, traits = "aziz.le", traits_def(Pure())
    lhs, rhs, res = operand_def(IntegerType), operand_def(IntegerType), result_def(IntegerType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[i32])


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "aziz.yield"
    input, traits = operand_def(IntegerType), traits_def(IsTerminator())  # HasParent(IfOp) would need IfOp defined first or forward ref

    def __init__(self, input: SSAValue):
        super().__init__(operands=[input])


@irdl_op_definition
class IfOp(IRDLOperation):
    name = "aziz.if"
    cond, res = operand_def(IntegerType), result_def(IntegerType)
    then_region, else_region = region_def(), region_def()

    def __init__(self, cond: SSAValue):
        super().__init__(operands=[cond], result_types=[i32], regions=[Region(Block()), Region(Block())])


Aziz = Dialect("aziz", [ConstantOp, AddOp, SubOp, MulOp, LessThanEqualOp, PrintOp, FuncOp, ReturnOp, CallOp, YieldOp, IfOp], [])
