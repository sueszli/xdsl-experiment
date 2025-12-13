from xdsl.dialects.builtin import AnyFloat, FloatAttr, FunctionType, IntegerAttr, IntegerType, StringAttr, SymbolNameConstraint, SymbolRefAttr, f64, i32
from xdsl.ir import Attribute, Block, Dialect, ParametrizedAttribute, Region, SSAValue
from xdsl.irdl import AnyOf, IRDLOperation, attr_def, irdl_attr_definition, irdl_op_definition, operand_def, opt_operand_def, region_def, result_def, traits_def, var_operand_def, var_result_def
from xdsl.traits import CallableOpInterface, HasParent, IsTerminator, Pure, SymbolOpInterface


@irdl_attr_definition
class StringType(ParametrizedAttribute):
    name = "aziz.string"


string_type = StringType()


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name, traits = "aziz.constant", traits_def(Pure())
    value = attr_def(Attribute)
    res = result_def(AnyOf([IntegerType, AnyFloat]))

    def __init__(self, value: int | float):
        if isinstance(value, float):
            super().__init__(result_types=[f64], attributes={"value": FloatAttr(value, f64)})
        else:
            super().__init__(result_types=[i32], attributes={"value": IntegerAttr(value, i32)})


@irdl_op_definition
class StringConstantOp(IRDLOperation):
    name, traits = "aziz.string_constant", traits_def(Pure())
    value = attr_def(StringAttr)
    res = result_def(StringType)

    def __init__(self, value: str):
        super().__init__(result_types=[string_type], attributes={"value": StringAttr(value)})


@irdl_op_definition
class AddOp(IRDLOperation):
    name, traits = "aziz.add", traits_def(Pure())
    lhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    rhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    res = result_def(AnyOf([IntegerType, AnyFloat]))

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type])


@irdl_op_definition
class MulOp(IRDLOperation):
    name, traits = "aziz.mul", traits_def(Pure())
    lhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    rhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    res = result_def(AnyOf([IntegerType, AnyFloat]))

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type])


@irdl_op_definition
class PrintOp(IRDLOperation):
    name = "aziz.print"
    input = operand_def(AnyOf([IntegerType, AnyFloat, StringType]))

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
    input = opt_operand_def(AnyOf([IntegerType, AnyFloat, StringType]))
    traits = traits_def(IsTerminator(), HasParent(FuncOp))

    def __init__(self, input: SSAValue | None = None):
        super().__init__(operands=[input] if input else [])


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "aziz.call"
    callee = attr_def(SymbolRefAttr)
    arguments = var_operand_def(AnyOf([IntegerType, AnyFloat, StringType]))
    res = var_result_def(AnyOf([IntegerType, AnyFloat, StringType]))

    def __init__(self, callee: str, args: list[SSAValue], ret_type: list[Attribute]):
        super().__init__(operands=[args], result_types=[ret_type], attributes={"callee": SymbolRefAttr(callee)})


@irdl_op_definition
class SubOp(IRDLOperation):
    name, traits = "aziz.sub", traits_def(Pure())
    lhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    rhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    res = result_def(AnyOf([IntegerType, AnyFloat]))

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type])


@irdl_op_definition
class LessThanEqualOp(IRDLOperation):
    name, traits = "aziz.le", traits_def(Pure())
    lhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    rhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    res = result_def(IntegerType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[i32])


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "aziz.yield"
    input = operand_def(AnyOf([IntegerType, AnyFloat, StringType]))
    traits = traits_def(IsTerminator())  # HasParent(IfOp) would need IfOp defined first or forward ref

    def __init__(self, input: SSAValue):
        super().__init__(operands=[input])


@irdl_op_definition
class IfOp(IRDLOperation):
    name = "aziz.if"
    cond = operand_def(IntegerType)
    res = result_def(AnyOf([IntegerType, AnyFloat, StringType]))
    then_region, else_region = region_def(), region_def()

    def __init__(self, cond: SSAValue, result_type: Attribute = i32):
        super().__init__(operands=[cond], result_types=[result_type], regions=[Region(Block()), Region(Block())])


Aziz = Dialect("aziz", [ConstantOp, StringConstantOp, AddOp, SubOp, MulOp, LessThanEqualOp, PrintOp, FuncOp, ReturnOp, CallOp, YieldOp, IfOp], [StringType])
