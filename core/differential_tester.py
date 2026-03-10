"""
Differential Tester for the Code Modernization Engine.

Verifies functional parity between original and modernized code by compiling 
and running both programs, then comparing their outputs.
"""

import os
import subprocess
import tempfile
from difflib import unified_diff
import logging


def _compile_and_run_cpp(
    source_path: str,
    gpp_exe: str,
    tmp_dir: str,
    exe_name: str
) -> tuple[str, str, bool]:
    """
    Compiles a C++ source file and runs the resulting executable.

    Args:
        source_path: Path to the C++ source file.
        gpp_exe: Path to the g++ executable.
        tmp_dir: Temporary directory for compilation.
        exe_name: Name for the compiled executable.

    Returns:
        tuple[str, str, bool]: 
            - stdout_text: Standard output from execution.
            - stderr_text: Standard error from compilation/execution.
            - success: True if both compile and run succeeded, False otherwise.
    """
    exe_path = os.path.join(tmp_dir, exe_name)
    compile_cmd = f'"{gpp_exe}" -std=c++20 -Wall "{source_path}" -o "{exe_path}"'
    
    try:
        result = subprocess.run(
            compile_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        return "", f"Compilation timed out: {exc!r}", False
    except Exception as exc:
        return "", f"Compilation failed: {exc!r}", False

    if result.returncode != 0:
        stderr_text = (result.stderr or "").strip()
        return "", stderr_text, False

    try:
        run_result = subprocess.run(
            exe_path,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        return "", f"Execution timed out: {exc!r}", False
    except Exception as exc:
        return "", f"Execution failed: {exc!r}", False

    stdout_text = (run_result.stdout or "").strip()
    stderr_text = (run_result.stderr or "").strip()
    return stdout_text, stderr_text, True


def _normalize_output(text: str) -> str:
    """
    Normalizes program output for fuzzy comparison.

    Handles line ending variations and trailing whitespace/newlines.

    Args:
        text: Output text to normalize.

    Returns:
        str: Normalized output string.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    stripped_lines = [line.rstrip() for line in lines]
    
    while stripped_lines and stripped_lines[-1] == "":
        stripped_lines.pop()
        
    return "\n".join(stripped_lines)


def run_differential_test(
    original_cpp_path: str,
    modernized_code: str,
    gpp_exe: str = r"C:\msys64\mingw64\bin\g++.exe",
) -> tuple[bool, str]:
    """
    Runs a differential test by comparing the execution of legacy and modernized code.

    Args:
        original_cpp_path: Path to the original C++ source file.
        modernized_code: Modernized C++ code as a string.
        gpp_exe: Path to the g++ compiler executable.

    Returns:
        tuple[bool, str]: 
            - parity_ok: True if outputs match, False otherwise.
            - diff_text: Unified diff string if outputs differ, else empty string.
    """
    if not os.path.isfile(original_cpp_path):
        logging.error(f"Original file not found: {original_cpp_path}")
        return False, ""
        
    if not modernized_code.strip():
        logging.error("No modernized code provided.")
        return False, ""

    with tempfile.TemporaryDirectory() as tmp_dir:
        modernized_cpp_path = os.path.join(tmp_dir, "modernized.cpp")
        with open(modernized_cpp_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(modernized_code)

        logging.info("Compiling and running ORIGINAL (test.cpp)...")
        orig_stdout, orig_stderr, orig_ok = _compile_and_run_cpp(
            original_cpp_path,
            gpp_exe,
            tmp_dir,
            "original.exe",
        )
        
        if not orig_ok:
            logging.error("ORIGINAL FAILED:")
            logging.error(orig_stderr)
            return False, ""

        logging.info("Compiling and running MODERNIZED code...")
        mod_stdout, mod_stderr, mod_ok = _compile_and_run_cpp(
            modernized_cpp_path,
            gpp_exe,
            tmp_dir,
            "modernized.exe",
        )
        
        if not mod_ok:
            logging.error("MODERNIZED FAILED:")
            logging.error(mod_stderr)
            return False, ""

        norm_orig = _normalize_output(orig_stdout)
        norm_mod = _normalize_output(mod_stdout)

        if norm_orig == norm_mod:
            logging.info("PASSED: Functional Parity Confirmed")
            return True, ""

        logging.warning("FAILED: Outputs differ. Generating diff...")
        orig_lines = (norm_orig + "\n").splitlines(keepends=True)
        mod_lines = (norm_mod + "\n").splitlines(keepends=True)
        
        diff_lines = unified_diff(
            orig_lines,
            mod_lines,
            fromfile="Original (test.cpp)",
            tofile="Modernized",
        )

        diff_text_parts: list[str] = []
        for line in diff_lines:
            diff_text_parts.append(line)
            print(line, end="")

        diff_text = "".join(diff_text_parts)
        return False, diff_text


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m core.differential_tester <original.cpp> [modernized_code_string_or_path]")
        sys.exit(1)

    orig_path = sys.argv[1]

    if len(sys.argv) >= 3:
        modernized_arg = sys.argv[2]
        if os.path.isfile(modernized_arg):
            with open(modernized_arg, "r", encoding="utf-8") as f:
                modernized_code = f.read()
        else:
            modernized_code = modernized_arg
    else:
        print("ERROR: Provide modernized code as second argument or path to file containing it.")
        sys.exit(1)

    success, _diff_text = run_differential_test(orig_path, modernized_code)
    sys.exit(0 if success else 1)