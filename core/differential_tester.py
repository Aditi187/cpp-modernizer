import os
import platform
import re
import subprocess
import tempfile
import time
import shutil
from dataclasses import dataclass
from difflib import unified_diff


# ---------------------------------------------------------------------------
# Sanitizer configuration
# ---------------------------------------------------------------------------

_SANITIZER_COMPILE_FLAGS: list[str] = [
    "-fsanitize=address,undefined",
    "-fno-omit-frame-pointer",
]


def _sanitizers_available() -> bool:
    """Return False on Windows/MinGW where ASan/UBSan libs are typically missing."""
    return platform.system() != "Windows"

_SANITIZER_ERROR_PATTERN = re.compile(
    r"(?:AddressSanitizer|UndefinedBehaviorSanitizer|LeakSanitizer|ERROR:\s*(?:address|leak|undefined))",
    re.IGNORECASE,
)


def _detect_sanitizer_errors(stderr_text: str) -> list[str]:
    """Return a list of sanitizer diagnostic lines found in *stderr_text*."""
    if not stderr_text:
        return []
    findings: list[str] = []
    for line in stderr_text.splitlines():
        if _SANITIZER_ERROR_PATTERN.search(line):
            findings.append(line.strip())
    return findings


def _parse_peak_memory_kb(stderr_text: str) -> int | None:
    """Extract peak resident-set size (KB) from ASan or /usr/bin/time output.

    AddressSanitizer prints a stats line like::

        SUMMARY: AddressSanitizer: 1234 byte(s) allocated

    or on Linux with ``ASAN_OPTIONS=print_stats=1``:

        ==PID== ASAN: ... rss: 12345 kB

    We also recognise GNU ``/usr/bin/time -v`` output::

        Maximum resident set size (kbytes): 12345

    Returns *None* when no metric is found.
    """
    if not stderr_text:
        return None

    # GNU time -v format
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr_text)
    if m:
        return int(m.group(1))

    # ASan stats (bytes allocated → convert to KB)
    m = re.search(r"(\d+)\s+byte\(s\)\s+allocated", stderr_text)
    if m:
        return max(1, int(m.group(1)) // 1024)

    return None


def resolve_gpp_exe(explicit_path: str | None = None) -> str:
    return resolve_cpp_compiler(explicit_path)


def resolve_cpp_compiler(explicit_path: str | None = None) -> str:
    if explicit_path:
        return explicit_path

    env_candidates = [
        os.environ.get("CXX", "").strip(),
        os.environ.get("GPP_EXE", "").strip(),
        os.environ.get("CLANGXX_EXE", "").strip(),
    ]
    for candidate in env_candidates:
        if candidate:
            return candidate

    preferred_bins = ["g++-13", "clang++-16", "g++", "clang++"]
    for binary in preferred_bins:
        found = shutil.which(binary)
        if found:
            return found

    return "g++"


def _verify_compiler(gpp_exe: str, timeout_seconds: int = 5) -> None:
    try:
        result = subprocess.run(
            [gpp_exe, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
        )
    except Exception as exc:
        raise RuntimeError(f"g++ sanity check failed: {exc!r}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"g++ sanity check failed with exit code {result.returncode}: {result.stderr}"
        )


def compile_cpp_source(
    code: str,
    gpp_exe: str | None = None,
    timeout_seconds: int = 10,
    enable_sanitizers: bool = True,
) -> dict:
    compiler = resolve_gpp_exe(gpp_exe)
    _verify_compiler(compiler)

    # Auto-disable sanitizers on Windows/MinGW (libs not available).
    if enable_sanitizers and not _sanitizers_available():
        enable_sanitizers = False

    start_time = time.time()

    with tempfile.TemporaryDirectory() as tmp_dir:
        cpp_path = os.path.join(tmp_dir, "modernized.cpp")
        exe_path = os.path.join(tmp_dir, "modernized.exe")

        with open(cpp_path, "w", encoding="utf-8") as cpp_file:
            cpp_file.write(code)

        cmd = [compiler, "-std=c++23", "-Wall", cpp_path, "-o", exe_path]
        if enable_sanitizers:
            cmd[2:2] = _SANITIZER_COMPILE_FLAGS

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                "success": False,
                "errors": [f"Compilation timed out after {timeout_seconds} seconds."],
                "warnings": [],
                "compilation_time_ms": elapsed_ms,
                "raw_stdout": "",
                "raw_stderr": "",
                "compiler": compiler,
            }
        except FileNotFoundError:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                "success": False,
                "errors": [
                    "g++ compiler not found. Please install a C++ compiler "
                    "and ensure it is on your PATH or set GPP_EXE."
                ],
                "warnings": [],
                "compilation_time_ms": elapsed_ms,
                "raw_stdout": "",
                "raw_stderr": "",
                "compiler": compiler,
            }
        except Exception as exc:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                "success": False,
                "errors": [f"Compilation failed: {exc!r}"],
                "warnings": [],
                "compilation_time_ms": elapsed_ms,
                "raw_stdout": "",
                "raw_stderr": "",
                "compiler": compiler,
            }

    elapsed_ms = int((time.time() - start_time) * 1000)
    stdout_text = (result.stdout or "").strip()
    stderr_text = (result.stderr or "").strip()

    success = result.returncode == 0
    error_lines = stderr_text.splitlines() if stderr_text else []
    warning_lines = stdout_text.splitlines() if stdout_text else []

    return {
        "success": success,
        "errors": error_lines,
        "warnings": warning_lines,
        "compilation_time_ms": elapsed_ms,
        "raw_stdout": stdout_text,
        "raw_stderr": stderr_text,
        "compiler": compiler,
    }


def _compile_and_run_cpp(
    source_path: str,
    gpp_exe: str,
    tmp_dir: str,
    exe_name: str,
    input_data: str | None = None,
    enable_sanitizers: bool = True,
) -> dict:
    _verify_compiler(gpp_exe)

    exe_path = os.path.join(tmp_dir, exe_name)

    # Auto-disable sanitizers on Windows/MinGW (libs not available).
    if enable_sanitizers and not _sanitizers_available():
        enable_sanitizers = False

    compile_start = time.time()
    compile_cmd = [gpp_exe, "-std=c++23", "-Wall", source_path, "-o", exe_path]
    if enable_sanitizers:
        compile_cmd[2:2] = _SANITIZER_COMPILE_FLAGS

    # Tell ASan to report leaks and print stats so we can extract peak memory.
    sanitizer_env = dict(os.environ)
    if enable_sanitizers:
        sanitizer_env["ASAN_OPTIONS"] = "detect_leaks=1:print_stats=1:halt_on_error=0"
        sanitizer_env["UBSAN_OPTIONS"] = "print_stacktrace=1:halt_on_error=0"

    try:
        compile_result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
            env=sanitizer_env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "compile_success": False,
            "run_success": False,
            "stdout": "",
            "stderr": f"Compilation timed out: {exc!r}",
            "compile_time_ms": int((time.time() - compile_start) * 1000),
            "run_time_ms": 0,
        }
    except Exception as exc:
        return {
            "compile_success": False,
            "run_success": False,
            "stdout": "",
            "stderr": f"Compilation failed: {exc!r}",
            "compile_time_ms": int((time.time() - compile_start) * 1000),
            "run_time_ms": 0,
        }

    compile_time_ms = int((time.time() - compile_start) * 1000)

    if compile_result.returncode != 0:
        stderr_text = (compile_result.stderr or "").strip()
        return {
            "compile_success": False,
            "run_success": False,
            "stdout": (compile_result.stdout or "").strip(),
            "stderr": stderr_text,
            "compile_time_ms": compile_time_ms,
            "run_time_ms": 0,
        }

    run_start = time.time()

    try:
        run_result = subprocess.run(
            [exe_path],
            input=input_data if input_data is not None else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
            env=sanitizer_env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "compile_success": True,
            "run_success": False,
            "stdout": "",
            "stderr": f"Execution timed out: {exc!r}",
            "compile_time_ms": compile_time_ms,
            "run_time_ms": int((time.time() - run_start) * 1000),
        }
    except Exception as exc:
        return {
            "compile_success": True,
            "run_success": False,
            "stdout": "",
            "stderr": f"Execution failed: {exc!r}",
            "compile_time_ms": compile_time_ms,
            "run_time_ms": int((time.time() - run_start) * 1000),
        }

    run_time_ms = int((time.time() - run_start) * 1000)

    stdout_text = (run_result.stdout or "").strip()
    stderr_text = (run_result.stderr or "").strip()
    run_success = run_result.returncode == 0

    sanitizer_findings = _detect_sanitizer_errors(stderr_text)
    peak_memory_kb = _parse_peak_memory_kb(stderr_text)

    return {
        "compile_success": True,
        "run_success": run_success,
        "exit_code": run_result.returncode,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "compile_time_ms": compile_time_ms,
        "run_time_ms": run_time_ms,
        "sanitizer_findings": sanitizer_findings,
        "peak_memory_kb": peak_memory_kb,
    }


def _normalize_output(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    stripped_lines = [line.rstrip() for line in lines]
    while stripped_lines and stripped_lines[-1] == "":
        stripped_lines.pop()
    return "\n".join(stripped_lines)


def _extract_error_location(stderr_text: str, source_label: str) -> str | None:
    for line in stderr_text.splitlines():
        if source_label in line:
            return line.strip()
    return None


@dataclass
class DifferentialTestResult:
    parity_ok: bool
    diff_text: str
    original: dict
    modernized: dict
    gpp_exe: str
    sanitizer_clean: bool = True
    sanitizer_findings: list[str] | None = None
    memory_delta_kb: int | None = None


def run_differential_test(
    original_cpp_path: str,
    modernized_code: str,
    gpp_exe: str | None = None,
    input_data: str | None = None,
) -> dict:
    compiler = resolve_gpp_exe(gpp_exe)
    _verify_compiler(compiler)

    if not os.path.isfile(original_cpp_path):
        return DifferentialTestResult(
            parity_ok=False,
            diff_text="",
            original={
                "compile_success": False,
                "run_success": False,
                "stdout": "",
                "stderr": f"Original file not found: {original_cpp_path}",
                "compile_time_ms": 0,
                "run_time_ms": 0,
            },
            modernized={
                "compile_success": False,
                "run_success": False,
                "stdout": "",
                "stderr": "",
                "compile_time_ms": 0,
                "run_time_ms": 0,
            },
            gpp_exe=compiler,
        ).__dict__

    if not modernized_code.strip():
        return DifferentialTestResult(
            parity_ok=False,
            diff_text="",
            original={
                "compile_success": False,
                "run_success": False,
                "stdout": "",
                "stderr": "",
                "compile_time_ms": 0,
                "run_time_ms": 0,
            },
            modernized={
                "compile_success": False,
                "run_success": False,
                "stdout": "",
                "stderr": "No modernized code provided.",
                "compile_time_ms": 0,
                "run_time_ms": 0,
            },
            gpp_exe=compiler,
        ).__dict__

    with tempfile.TemporaryDirectory() as tmp_dir:
        modernized_cpp_path = os.path.join(tmp_dir, "modernized.cpp")

        with open(modernized_cpp_path, "w", encoding="utf-8") as f:
            f.write(modernized_code)

        original_result = _compile_and_run_cpp(
            original_cpp_path,
            compiler,
            tmp_dir,
            "original.exe",
            input_data=input_data,
            enable_sanitizers=False,  # Don't sanitize legacy code—only the modernized version.
        )

        if not (original_result["compile_success"] and original_result["run_success"]):
            return DifferentialTestResult(
                parity_ok=False,
                diff_text="",
                original=original_result,
                modernized={
                    "compile_success": False,
                    "run_success": False,
                    "stdout": "",
                    "stderr": "",
                    "compile_time_ms": 0,
                    "run_time_ms": 0,
                },
                gpp_exe=compiler,
            ).__dict__

        modernized_result = _compile_and_run_cpp(
            modernized_cpp_path,
            compiler,
            tmp_dir,
            "modernized.exe",
            input_data=input_data,
            enable_sanitizers=True,
        )

        if not modernized_result["compile_success"]:
            location = _extract_error_location(
                modernized_result["stderr"], os.path.basename(modernized_cpp_path)
            )
            if location:
                modernized_result["stderr"] = (
                    modernized_result["stderr"] + "\n" + f"First error location: {location}"
                )
            return DifferentialTestResult(
                parity_ok=False,
                diff_text="",
                original=original_result,
                modernized=modernized_result,
                gpp_exe=compiler,
            ).__dict__

        if not modernized_result["run_success"]:
            return DifferentialTestResult(
                parity_ok=False,
                diff_text="",
                original=original_result,
                modernized=modernized_result,
                gpp_exe=compiler,
            ).__dict__

        norm_orig = _normalize_output(original_result["stdout"])
        norm_mod = _normalize_output(modernized_result["stdout"])
        norm_orig_err = _normalize_output(original_result.get("stderr", ""))
        norm_mod_err = _normalize_output(modernized_result.get("stderr", ""))
        orig_exit = int(original_result.get("exit_code", 0))
        mod_exit = int(modernized_result.get("exit_code", 0))

        # --- Sanitizer & memory analysis on the *modernized* run ---
        mod_sanitizer_findings = modernized_result.get("sanitizer_findings") or []
        sanitizer_clean = len(mod_sanitizer_findings) == 0

        orig_peak = original_result.get("peak_memory_kb")
        mod_peak = modernized_result.get("peak_memory_kb")
        memory_delta_kb: int | None = None
        if orig_peak is not None and mod_peak is not None:
            memory_delta_kb = mod_peak - orig_peak  # negative = improvement

        if norm_orig == norm_mod and norm_orig_err == norm_mod_err and orig_exit == mod_exit:
            # Even if stdout matches, flag failure when sanitizers detected issues.
            return DifferentialTestResult(
                parity_ok=sanitizer_clean,  # fail if sanitizer found problems
                diff_text="" if sanitizer_clean else (
                    "Output matched, but sanitizer detected issues:\n"
                    + "\n".join(mod_sanitizer_findings)
                ),
                original=original_result,
                modernized=modernized_result,
                gpp_exe=compiler,
                sanitizer_clean=sanitizer_clean,
                sanitizer_findings=mod_sanitizer_findings if mod_sanitizer_findings else None,
                memory_delta_kb=memory_delta_kb,
            ).__dict__

        orig_lines = (norm_orig + "\n").splitlines(keepends=True)
        mod_lines = (norm_mod + "\n").splitlines(keepends=True)
        orig_err_lines = (norm_orig_err + "\n").splitlines(keepends=True)
        mod_err_lines = (norm_mod_err + "\n").splitlines(keepends=True)

        diff_lines = unified_diff(
            orig_lines,
            mod_lines,
            fromfile="Original (test.cpp)",
            tofile="Modernized",
        )
        err_diff_lines = unified_diff(
            orig_err_lines,
            mod_err_lines,
            fromfile="Original stderr",
            tofile="Modernized stderr",
        )

        diff_text_parts: list[str] = []
        for line in diff_lines:
            diff_text_parts.append(line)
        for line in err_diff_lines:
            diff_text_parts.append(line)
        if orig_exit != mod_exit:
            diff_text_parts.append(
                f"\nExit code mismatch: original={orig_exit}, modernized={mod_exit}\n"
            )

        diff_text = "".join(diff_text_parts)

        return DifferentialTestResult(
            parity_ok=False,
            diff_text=diff_text,
            original=original_result,
            modernized=modernized_result,
            gpp_exe=compiler,
            sanitizer_clean=sanitizer_clean,
            sanitizer_findings=mod_sanitizer_findings if mod_sanitizer_findings else None,
            memory_delta_kb=memory_delta_kb,
        ).__dict__