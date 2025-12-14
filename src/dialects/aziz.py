from collections.abc import Sequence
from typing import cast

from xdsl.dialects.builtin import AnyFloat, FloatAttr, FunctionType, IntegerAttr, IntegerType, StringAttr, SymbolNameConstraint, SymbolRefAttr, f64, i32
from xdsl.ir import Attribute, Block, Dialect, ParametrizedAttribute, Region, SSAValue
from xdsl.irdl import AnyOf, IRDLOperation, attr_def, irdl_attr_definition, irdl_op_definition, operand_def, opt_attr_def, opt_operand_def, region_def, result_def, traits_def, var_operand_def, var_result_def
from xdsl.traits import CallableOpInterface, HasParent, IsTerminator, Pure, SymbolOpInterface
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class StringType(ParametrizedAttribute):
    name = "aziz.string"


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
        super().__init__(result_types=[StringType()], attributes={"value": StringAttr(value)})


@irdl_op_definition
class AddOp(IRDLOperation):
    name, traits = "aziz.add", traits_def(Pure())
    lhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    rhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    res = result_def(AnyOf([IntegerType, AnyFloat]))

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type])

    def verify_(self):
        if self.lhs.type != self.rhs.type:
            raise VerifyException("expected AddOp args to have the same type")


@irdl_op_definition
class SubOp(IRDLOperation):
    name, traits = "aziz.sub", traits_def(Pure())
    lhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    rhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    res = result_def(AnyOf([IntegerType, AnyFloat]))

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type])

    def verify_(self):
        if self.lhs.type != self.rhs.type:
            raise VerifyException("expected SubOp args to have the same type")


@irdl_op_definition
class MulOp(IRDLOperation):
    name, traits = "aziz.mul", traits_def(Pure())
    lhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    rhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    res = result_def(AnyOf([IntegerType, AnyFloat]))

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type])

    def verify_(self):
        if self.lhs.type != self.rhs.type:
            raise VerifyException("expected MulOp args to have the same type")


@irdl_op_definition
class LessThanEqualOp(IRDLOperation):
    name, traits = "aziz.le", traits_def(Pure())
    lhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    rhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    res = result_def(IntegerType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[i32])

    def verify_(self):
        if self.lhs.type != self.rhs.type:
            raise VerifyException("expected LessThanEqualOp args to have the same type")


@irdl_op_definition
class PrintOp(IRDLOperation):
    name = "aziz.print"
    input = operand_def(AnyOf([IntegerType, AnyFloat, StringType]))

    def __init__(self, input: SSAValue):
        super().__init__(operands=[input])


class FuncOpCallableInterface(CallableOpInterface):
    @classmethod
    def get_callable_region(cls, op: IRDLOperation) -> Region:
        return cast(FuncOp, op).body

    @classmethod
    def get_argument_types(cls, op: IRDLOperation) -> tuple[Attribute, ...]:
        return cast(FuncOp, op).function_type.inputs.data

    @classmethod
    def get_result_types(cls, op: IRDLOperation) -> tuple[Attribute, ...]:
        return cast(FuncOp, op).function_type.outputs.data


@irdl_op_definition
class FuncOp(IRDLOperation):
    name = "aziz.func"
    body = region_def("single_block")
    sym_name = attr_def(SymbolNameConstraint())
    function_type = attr_def(FunctionType)
    sym_visibility = opt_attr_def(StringAttr)
    traits = traits_def(SymbolOpInterface(), FuncOpCallableInterface())

    def __init__(
        self,
        name: str,
        ftype: FunctionType,
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
        /,
        private: bool = False,
    ):
        attributes: dict[str, Attribute] = {
            "sym_name": StringAttr(name),
            "function_type": ftype,
        }
        if not isinstance(region, Region):
            region = Region(Block(arg_types=ftype.inputs))
        if private:
            attributes["sym_visibility"] = StringAttr("private")

        super().__init__(attributes=attributes, regions=[region])

    def verify_(self):
        block = self.body.block
        if not block.ops:
            raise VerifyException("expected FuncOp to not be empty")

        last_op = block.last_op
        if not isinstance(last_op, ReturnOp):
            raise VerifyException("expected last op of FuncOp to be a ReturnOp")


@irdl_op_definition
class ReturnOp(IRDLOperation):
    name = "aziz.return"
    input = opt_operand_def(AnyOf([IntegerType, AnyFloat, StringType]))
    traits = traits_def(IsTerminator(), HasParent(FuncOp))

    def __init__(self, input: SSAValue | None = None):
        super().__init__(operands=[input] if input else [])

    def verify_(self) -> None:
        func_op = self.parent_op()
        assert isinstance(func_op, FuncOp)

        function_return_types = func_op.function_type.outputs.data
        if self.input:
            if len(function_return_types) != 1:
                raise VerifyException("expected 1 return value for non-void function")
            if self.input.type != function_return_types[0]:
                raise VerifyException("expected argument type to match function return type")
        else:
            if len(function_return_types) != 0:
                raise VerifyException("expected 0 return values for void function")


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "aziz.call"
    callee = attr_def(SymbolRefAttr)
    arguments = var_operand_def(AnyOf([IntegerType, AnyFloat, StringType]))
    res = var_result_def(AnyOf([IntegerType, AnyFloat, StringType]))

    def __init__(self, callee: str | SymbolRefAttr, operands: Sequence[SSAValue], return_types: Sequence[Attribute]):
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)

        super().__init__(operands=[operands], result_types=[return_types], attributes={"callee": callee})


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "aziz.yield"
    input = operand_def(AnyOf([IntegerType, AnyFloat, StringType]))
    traits = traits_def(IsTerminator())

    def __init__(self, input: SSAValue):
        super().__init__(operands=[input])


@irdl_op_definition
class IfOp(IRDLOperation):
    name = "aziz.if"
    cond = operand_def(IntegerType)
    res = result_def(AnyOf([IntegerType, AnyFloat, StringType]))
    then_region, else_region = region_def(), region_def()

    def __init__(
        self,
        cond: SSAValue,
        result_type: Attribute = i32,
        regions: list[Region] | None = None,
    ):
        if regions is None:
            regions = [Region(Block()), Region(Block())]
        super().__init__(operands=[cond], result_types=[result_type], regions=regions)

    def verify_(self) -> None:
        then_block = self.then_region.block
        else_block = self.else_region.block

        if not then_block.ops or not else_block.ops:
            raise VerifyException("expected IfOp regions to not be empty")

        then_yield = then_block.last_op
        else_yield = else_block.last_op

        if not isinstance(then_yield, YieldOp):
            raise VerifyException("expected last op of then region to be a YieldOp")
        if not isinstance(else_yield, YieldOp):
            raise VerifyException("expected last op of else region to be a YieldOp")
        if then_yield.input.type != else_yield.input.type:
            raise VerifyException(f"if expression branches have different types: {then_yield.input.type} vs {else_yield.input.type}")


Aziz = Dialect(
    "aziz",
    [
        ConstantOp,
        StringConstantOp,
        AddOp,
        SubOp,
        MulOp,
        LessThanEqualOp,
        PrintOp,
        FuncOp,
        ReturnOp,
        CallOp,
        YieldOp,
        IfOp,
    ],
    [StringType],
)
