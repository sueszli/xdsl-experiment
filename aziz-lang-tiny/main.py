# /// script
# requires-python = "==3.14"
# dependencies = [
#     "xdsl==0.56.0",
#     "unicorn==2.1.4",
#     "pyelftools==0.32",
# ]
# ///

import argparse
import subprocess
import tempfile
from parser import AzizParser
from pathlib import Path

import aziz
from ir_gen import IRGen
from lower import LowerAzizPass, LowerPrintfToLLVMCallPass
from optimize import OptimizeAzizPass
from xdsl.context import Context
from xdsl.dialects import affine, arith, func, printf, scf
from xdsl.dialects.builtin import Builtin
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.lower_affine import LowerAffinePass


def main():
    parser = argparse.ArgumentParser(description="aziz language")
    parser.add_argument("file", help="source file")
    args = parser.parse_args()
    assert args.file.endswith(".aziz")
    src = Path(args.file).read_text()

    # aziz dialect -> llvm dialect mlir -> llvm ir -> printed, then piped into bash script
    # uv run aziz-lang-tiny/main.py examples/rec.aziz | ...

    module_ast = AzizParser(None, src).parse_module()
    module_op = IRGen().ir_gen_module(module_ast)

    print(f"{'-' * 30} MLIR before optimization")
    print(module_op)

    ctx = context()
    OptimizeAzizPass().apply(ctx, module_op)
    module_op.verify()

    print(f"{'-' * 30} MLIR after optimization")
    print(module_op)

    LowerAzizPass().apply(ctx, module_op)
    LowerAffinePass().apply(ctx, module_op)
    CanonicalizePass().apply(ctx, module_op)
    LowerPrintfToLLVMCallPass().apply(ctx, module_op)
    module_op.verify()

    print(f"{'-' * 30} MLIR lowered to LLVM dialect")
    print(module_op)

    llvm_ir, llvm_exec_out, llvm_exec_err = execute_llvm(module_op)
    print(llvm_ir)

    print(f"{'-' * 30} LLVM output")
    print(llvm_exec_out)
    print(llvm_exec_err)


def context() -> Context:
    ctx = Context()
    ctx.load_dialect(aziz.Aziz)
    ctx.load_dialect(affine.Affine)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(printf.Printf)
    ctx.load_dialect(scf.Scf)
    return ctx


def execute_llvm(module_op_llvm):
    res = subprocess.run(["mlir-opt", "--convert-scf-to-cf", "--convert-func-to-llvm", "--convert-arith-to-llvm", "--convert-cf-to-llvm", "--reconcile-unrealized-casts"], input=str(module_op_llvm), capture_output=True, text=True)
    assert res.returncode == 0, f"mlir-opt failed:\n{res.stderr}"
    mlir_opt = res.stdout

    res = subprocess.run(["mlir-translate", "--mlir-to-llvmir"], input=mlir_opt, capture_output=True, text=True)
    assert res.returncode == 0, f"mlir-translate failed:\n{res.stderr}"
    llvm_ir = res.stdout

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
        f.write(llvm_ir)
        ll_path = f.name

    obj_path = ll_path.replace(".ll", ".o")
    res = subprocess.run(["llc", "-filetype=obj", "-o", obj_path, ll_path], capture_output=True, text=True)
    assert res.returncode == 0, f"llc failed:\n{res.stderr}"
    with tempfile.NamedTemporaryFile(delete=False) as f:
        output_file = f.name
    res = subprocess.run(["clang", "-o", output_file, obj_path], capture_output=True, text=True)
    assert res.returncode == 0, f"clang failed:\n{res.stderr}"

    res = subprocess.run([output_file], capture_output=True, text=True)
    llvm_exec_out = res.stdout
    llvm_exec_err = res.stderr
    return llvm_ir, llvm_exec_out, llvm_exec_err


if __name__ == "__main__":
    main()
