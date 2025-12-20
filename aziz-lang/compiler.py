from io import StringIO

from rewrites.lower import LowerAzizPass
from rewrites.optimize import OptimizeAzizPass
from xdsl.backend.riscv.lowering.convert_arith_to_riscv import ConvertArithToRiscvPass
from xdsl.backend.riscv.lowering.convert_func_to_riscv_func import ConvertFuncToRiscvFuncPass
from xdsl.backend.riscv.lowering.convert_memref_to_riscv import ConvertMemRefToRiscvPass
from xdsl.backend.riscv.lowering.convert_print_format_to_riscv_debug import ConvertPrintFormatToRiscvDebugPass
from xdsl.backend.riscv.lowering.convert_riscv_scf_to_riscv_cf import ConvertRiscvScfToRiscvCfPass
from xdsl.backend.riscv.lowering.convert_scf_to_riscv_scf import ConvertScfToRiscvPass
from xdsl.context import Context
from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.dead_code_elimination import DeadCodeElimination
from xdsl.transforms.lower_affine import LowerAffinePass
from xdsl.transforms.lower_riscv_func import LowerRISCVFunc
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass
from xdsl.transforms.riscv_allocate_registers import RISCVAllocateRegistersPass
from xdsl.transforms.riscv_scf_loop_range_folding import RiscvScfLoopRangeFoldingPass


def transform(
    ctx: Context,
    module_op: ModuleOp,
    *,
    target: str = "riscv-assembly",
):
    if target == "aziz":
        return

    CanonicalizePass().apply(ctx, module_op)

    if target == "aziz-opt":
        return

    OptimizeAzizPass().apply(ctx, module_op)

    if target == "aziz-inline":
        return

    # custom aziz dialect -> generic mlir dialects
    LowerAzizPass().apply(ctx, module_op)
    CanonicalizePass().apply(ctx, module_op)
    module_op.verify()

    if target == "aziz-lowered":
        return

    # Lowering generic dialects to RISC-V specific dialects

    LowerAffinePass().apply(ctx, module_op)

    if target == "scf":
        return

    ConvertFuncToRiscvFuncPass().apply(ctx, module_op)
    ConvertMemRefToRiscvPass().apply(ctx, module_op)
    ConvertPrintFormatToRiscvDebugPass().apply(ctx, module_op)
    ConvertArithToRiscvPass().apply(ctx, module_op)
    ConvertScfToRiscvPass().apply(ctx, module_op)
    DeadCodeElimination().apply(ctx, module_op)
    ReconcileUnrealizedCastsPass().apply(ctx, module_op)

    module_op.verify()

    if target == "riscv":
        return

    # Perform optimizations that don't depend on register allocation
    # e.g. constant folding
    CanonicalizePass().apply(ctx, module_op)
    RiscvScfLoopRangeFoldingPass().apply(ctx, module_op)
    CanonicalizePass().apply(ctx, module_op)

    module_op.verify()

    if target == "riscv-opt":
        return

    RISCVAllocateRegistersPass(allow_infinite=True).apply(ctx, module_op)

    module_op.verify()

    if target == "riscv-regalloc":
        return

    # Perform optimizations that depend on register allocation
    # e.g. redundant moves
    CanonicalizePass().apply(ctx, module_op)

    module_op.verify()

    if target == "riscv-regalloc-opt":
        return

    LowerRISCVFunc(insert_exit_syscall=True).apply(ctx, module_op)
    ConvertRiscvScfToRiscvCfPass().apply(ctx, module_op)

    if target == "riscv-lowered":
        return

    raise ValueError(f"Unknown target option {target}")


def compile(program: str) -> str:
    ctx = context()

    op = parse_aziz(program, ctx)
    transform(ctx, op, target="riscv-lowered")

    io = StringIO()
    riscv.print_assembly(op, io)

    return io.getvalue()
