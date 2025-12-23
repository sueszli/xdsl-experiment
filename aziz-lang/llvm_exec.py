import subprocess
import tempfile


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
