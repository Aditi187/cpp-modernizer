import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from difflib import unified_diff
from typing import Any


# ---------------------------------------------------------------------------
# Sanitizer configuration
# ---------------------------------------------------------------------------

_SANITIZER_COMPILE_FLAGS: list[str] = [
    "-fsanitize=address,undefined",
    "-fno-omit-frame-pointer",
]

_SANITIZER_ERROR_PATTERN = re.compile(
    r"(?:AddressSanitizer|UndefinedBehaviorSanitizer|LeakSanitizer|ERROR:\s*(?:address|leak|undefined))",
    re.IGNORECASE,
)

_CRASH_STDERR_PATTERN = re.compile(
    r"segmentation fault|access violation|illegal instruction|floating point exception|aborted|stack overflow",
    re.IGNORECASE,
)

_VERIFIED_COMPILERS: set[str] = set()


def _sanitizers_available() -> bool:
    """Return False on Windows/MinGW where ASan/UBSan libs are typically missing."""
    return platform.system() != "Windows"


def _detect_sanitizer_errors(stderr_text: str) -> list[str]:
    """Return a list of sanitizer diagnostic lines found in stderr output."""
    if not stderr_text:
        return []
    findings: list[str] = []
    for line in stderr_text.splitlines():
        if _SANITIZER_ERROR_PATTERN.search(line):
            findings.append(line.strip())
    return findings


def _parse_peak_memory_kb(stderr_text: str) -> int | None:
    """Extract peak resident-set size (KB) from ASan or /usr/bin/time output.

    On Windows this commonly returns None because /usr/bin/time -v is unavailable.
    """
    if not stderr_text:
        return None

    # GNU time -v format.
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr_text)
    if match:
        return int(match.group(1))

    # ASan stats (bytes allocated -> convert to KB).
    match = re.search(r"(\d+)\s+byte\(s\)\s+allocated", stderr_text)
    if match:
        return max(1, int(match.group(1)) // 1024)

    return None


def resolve_cpp_compiler(explicit_path: str | None = None) -> str:
    """Resolve the C++ compiler path using explicit arg, env vars, and PATH probes."""
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


def _verify_compiler(compiler_path: str, timeout_seconds: int = 5) -> None:
    """Verify that the compiler is invokable, caching successful checks."""
    if compiler_path in _VERIFIED_COMPILERS:
        return

    try:
        result = subprocess.run(
            [compiler_path, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
        )
    except Exception as exc:
        raise RuntimeError(f"C++ compiler sanity check failed: {exc!r}") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"C++ compiler sanity check failed with exit code {result.returncode}: {result.stderr}"
        )

    _VERIFIED_COMPILERS.add(compiler_path)


def _build_compile_command(
    compiler_path: str,
    source_path: str,
    exe_path: str,
    enable_sanitizers: bool,
) -> list[str]:
    cmd = [compiler_path, "-std=c++23", "-Wall"]
    if enable_sanitizers:
        cmd.extend(_SANITIZER_COMPILE_FLAGS)
    cmd.extend([source_path, "-o", exe_path])
    return cmd


def _build_run_env(enable_sanitizers: bool) -> dict[str, str]:
    env = dict(os.environ)
    if enable_sanitizers:
        env["ASAN_OPTIONS"] = "detect_leaks=1:print_stats=1:halt_on_error=0"
        env["UBSAN_OPTIONS"] = "print_stacktrace=1:halt_on_error=0"
    return env


def _compile_to_exe(
    source_path: str,
    compiler_path: str,
    tmp_dir: str,
    exe_name: str,
    enable_sanitizers: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Compile one C++ source file into an executable and return compile metadata."""
    exe_path = os.path.join(tmp_dir, exe_name)

    if enable_sanitizers and not _sanitizers_available():
        enable_sanitizers = False

    compile_cmd = _build_compile_command(compiler_path, source_path, exe_path, enable_sanitizers)
    compile_env = _build_run_env(enable_sanitizers)

    start = time.time()
    try:
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
            env=compile_env,
        )
    except subprocess.TimeoutExpired:
        return {
            "compile_success": False,
            "stderr": f"Compilation timed out after {timeout_seconds} seconds.",
            "stdout": "",
            "compile_time_ms": int((time.time() - start) * 1000),
            "exe_path": exe_path,
            "enable_sanitizers": enable_sanitizers,
            "timed_out": True,
        }
    except Exception as exc:
        return {
            "compile_success": False,
            "stderr": f"Compilation failed: {exc!r}",
            "stdout": "",
            "compile_time_ms": int((time.time() - start) * 1000),
            "exe_path": exe_path,
            "enable_sanitizers": enable_sanitizers,
            "timed_out": False,
        }

    return {
        "compile_success": result.returncode == 0,
        "stderr": (result.stderr or "").strip(),
        "stdout": (result.stdout or "").strip(),
        "compile_time_ms": int((time.time() - start) * 1000),
        "exe_path": exe_path,
        "enable_sanitizers": enable_sanitizers,
        "timed_out": False,
    }


def _detect_crash_reason(exit_code: int | None, stderr_text: str, timed_out: bool) -> str:
    """Infer crash reason in a platform-tolerant way from stderr and exit code."""
    if timed_out:
        return "timeout"

    if _CRASH_STDERR_PATTERN.search(stderr_text or ""):
        return "Process crashed (detected from stderr)."

    if exit_code is None:
        return "execution_error"

    # Negative return codes usually mean signal termination on Unix-like systems.
    if exit_code < 0:
        return f"Process terminated by signal {-exit_code} (platform-dependent)."

    if exit_code != 0:
        return f"Process exited with non-zero status {exit_code}."

    return ""


def _run_exe(
    exe_path: str,
    input_data: str | None,
    timeout_seconds: int,
    env: dict[str, str],
) -> dict[str, Any]:
    """Run a compiled executable once and return runtime diagnostics."""
    run_start = time.time()
    use_time_v = platform.system() != "Windows" and os.path.isfile("/usr/bin/time")
    run_cmd = [exe_path] if not use_time_v else ["/usr/bin/time", "-v", exe_path]

    try:
        run_result = subprocess.run(
            run_cmd,
            input=input_data if input_data is not None else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
            env=env,
        )
        timed_out = False
    except subprocess.TimeoutExpired:
        return {
            "compile_success": True,
            "run_success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout_seconds} seconds.",
            "run_time_ms": int((time.time() - run_start) * 1000),
            "exit_code": None,
            "sanitizer_findings": [],
            "peak_memory_kb": None,
            "timed_out": True,
            "crash_reason": "timeout",
        }
    except Exception as exc:
        return {
            "compile_success": True,
            "run_success": False,
            "stdout": "",
            "stderr": f"Execution failed: {exc!r}",
            "run_time_ms": int((time.time() - run_start) * 1000),
            "exit_code": None,
            "sanitizer_findings": [],
            "peak_memory_kb": None,
            "timed_out": False,
            "crash_reason": "execution_error",
        }

    run_time_ms = int((time.time() - run_start) * 1000)
    stdout_text = (run_result.stdout or "").strip()
    stderr_text = (run_result.stderr or "").strip()
    exit_code = int(run_result.returncode)

    crash_reason = _detect_crash_reason(exit_code, stderr_text, timed_out)
    sanitizer_findings = _detect_sanitizer_errors(stderr_text)
    peak_memory_kb = _parse_peak_memory_kb(stderr_text)

    return {
        "compile_success": True,
        "run_success": exit_code == 0 and not crash_reason,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "run_time_ms": run_time_ms,
        "exit_code": exit_code,
        "sanitizer_findings": sanitizer_findings,
        "peak_memory_kb": peak_memory_kb,
        "timed_out": False,
        "crash_reason": crash_reason,
    }


def compile_cpp_source(
    code: str,
    gpp_exe: str | None = None,
    timeout_seconds: int = 10,
    enable_sanitizers: bool = True,
) -> dict:
    """Compile C++ source code and return compiler diagnostics.

    Parameters:
    - code: complete C++ source text
    - gpp_exe: optional explicit compiler path/name
    - timeout_seconds: compile timeout
    - enable_sanitizers: enable ASan/UBSan where available
    """
    compiler_path = resolve_cpp_compiler(gpp_exe)
    _verify_compiler(compiler_path)

    start_time = time.time()
    with tempfile.TemporaryDirectory() as tmp_dir:
        cpp_path = os.path.join(tmp_dir, "modernized.cpp")
        with open(cpp_path, "w", encoding="utf-8") as cpp_file:
            cpp_file.write(code)

        compile_result = _compile_to_exe(
            source_path=cpp_path,
            compiler_path=compiler_path,
            tmp_dir=tmp_dir,
            exe_name="modernized.exe",
            enable_sanitizers=enable_sanitizers,
            timeout_seconds=timeout_seconds,
        )

    elapsed_ms = int((time.time() - start_time) * 1000)
    stderr_text = str(compile_result.get("stderr") or "")
    stdout_text = str(compile_result.get("stdout") or "")

    if not compile_result.get("compile_success", False):
        return {
            "success": False,
            "errors": stderr_text.splitlines() if stderr_text else ["Compilation failed."],
            "warnings": [],
            "compilation_time_ms": elapsed_ms,
            "raw_stdout": stdout_text,
            "raw_stderr": stderr_text,
            "compiler": compiler_path,
        }

    return {
        "success": True,
        "errors": [],
        "warnings": stdout_text.splitlines() if stdout_text else [],
        "compilation_time_ms": elapsed_ms,
        "raw_stdout": stdout_text,
        "raw_stderr": stderr_text,
        "compiler": compiler_path,
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
    test_cases_run: int = 0
    failed_case_index: int | None = None
    performance_delta_ms: int | None = None


def run_differential_test(
    original_cpp_path: str,
    modernized_code: str,
    gpp_exe: str | None = None,
    input_data: str | None = None,
    input_cases: list[str] | None = None,
    compile_timeout_seconds: int = 10,
    run_timeout_seconds: int = 10,
) -> dict:
    """Compile original/modernized code once, run each test input, and compare parity.

    Security note: this executes compiled binaries from provided source code.
    Timeouts reduce risk but do not provide full sandbox isolation.
    """
    compiler_path = resolve_cpp_compiler(gpp_exe)
    _verify_compiler(compiler_path)

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
            gpp_exe=compiler_path,
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
            gpp_exe=compiler_path,
            test_cases_run=0,
        ).__dict__

    effective_input_cases = input_cases if input_cases is not None else []
    if not effective_input_cases:
        effective_input_cases = [input_data if input_data is not None else ""]

    with tempfile.TemporaryDirectory() as tmp_dir:
        modernized_cpp_path = os.path.join(tmp_dir, "modernized.cpp")
        with open(modernized_cpp_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(modernized_code)

        original_compile = _compile_to_exe(
            source_path=original_cpp_path,
            compiler_path=compiler_path,
            tmp_dir=tmp_dir,
            exe_name="original.exe",
            enable_sanitizers=False,
            timeout_seconds=compile_timeout_seconds,
        )
        if not original_compile.get("compile_success", False):
            return DifferentialTestResult(
                parity_ok=False,
                diff_text="Original code failed to compile. Please fix legacy source before differential testing.",
                original={
                    "compile_success": False,
                    "run_success": False,
                    "stdout": str(original_compile.get("stdout") or ""),
                    "stderr": str(original_compile.get("stderr") or ""),
                    "compile_time_ms": int(original_compile.get("compile_time_ms", 0) or 0),
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
                gpp_exe=compiler_path,
                test_cases_run=0,
            ).__dict__

        modernized_compile = _compile_to_exe(
            source_path=modernized_cpp_path,
            compiler_path=compiler_path,
            tmp_dir=tmp_dir,
            exe_name="modernized.exe",
            enable_sanitizers=True,
            timeout_seconds=compile_timeout_seconds,
        )
        if not modernized_compile.get("compile_success", False):
            modernized_stderr = str(modernized_compile.get("stderr") or "")
            location = _extract_error_location(modernized_stderr, os.path.basename(modernized_cpp_path))
            if location:
                modernized_stderr = modernized_stderr + "\n" + f"First error location: {location}"
            return DifferentialTestResult(
                parity_ok=False,
                diff_text="Modernized code failed to compile. See compiler diagnostics.",
                original={
                    "compile_success": True,
                    "run_success": True,
                    "stdout": str(original_compile.get("stdout") or ""),
                    "stderr": str(original_compile.get("stderr") or ""),
                    "compile_time_ms": int(original_compile.get("compile_time_ms", 0) or 0),
                    "run_time_ms": 0,
                },
                modernized={
                    "compile_success": False,
                    "run_success": False,
                    "stdout": str(modernized_compile.get("stdout") or ""),
                    "stderr": modernized_stderr,
                    "compile_time_ms": int(modernized_compile.get("compile_time_ms", 0) or 0),
                    "run_time_ms": 0,
                },
                gpp_exe=compiler_path,
                test_cases_run=0,
            ).__dict__

        original_compile_ms = int(original_compile.get("compile_time_ms", 0) or 0)
        modernized_compile_ms = int(modernized_compile.get("compile_time_ms", 0) or 0)

        original_env = _build_run_env(enable_sanitizers=False)
        modernized_env = _build_run_env(enable_sanitizers=bool(modernized_compile.get("enable_sanitizers")))
        original_exe_path = str(original_compile.get("exe_path") or "")
        modernized_exe_path = str(modernized_compile.get("exe_path") or "")

        original_cases: list[dict[str, Any]] = []
        modernized_cases: list[dict[str, Any]] = []
        all_sanitizer_findings: list[str] = []
        total_original_run_ms = 0
        total_modernized_run_ms = 0
        max_original_peak: int | None = None
        max_modernized_peak: int | None = None

        for case_index, case_input in enumerate(effective_input_cases):
            original_case_result = _run_exe(
                exe_path=original_exe_path,
                input_data=case_input,
                timeout_seconds=run_timeout_seconds,
                env=original_env,
            )
            original_case_result["compile_time_ms"] = original_compile_ms

            modernized_case_result = _run_exe(
                exe_path=modernized_exe_path,
                input_data=case_input,
                timeout_seconds=run_timeout_seconds,
                env=modernized_env,
            )
            modernized_case_result["compile_time_ms"] = modernized_compile_ms

            original_cases.append(original_case_result)
            modernized_cases.append(modernized_case_result)

            total_original_run_ms += int(original_case_result.get("run_time_ms", 0) or 0)
            total_modernized_run_ms += int(modernized_case_result.get("run_time_ms", 0) or 0)

            original_peak_case = original_case_result.get("peak_memory_kb")
            modernized_peak_case = modernized_case_result.get("peak_memory_kb")
            if isinstance(original_peak_case, int):
                max_original_peak = (
                    original_peak_case
                    if max_original_peak is None
                    else max(max_original_peak, original_peak_case)
                )
            if isinstance(modernized_peak_case, int):
                max_modernized_peak = (
                    modernized_peak_case
                    if max_modernized_peak is None
                    else max(max_modernized_peak, modernized_peak_case)
                )

            case_findings = modernized_case_result.get("sanitizer_findings") or []
            all_sanitizer_findings.extend([str(item) for item in case_findings])

            if not original_case_result.get("run_success", False):
                reason = str(original_case_result.get("crash_reason") or "original runtime failure")
                return DifferentialTestResult(
                    parity_ok=False,
                    diff_text=(
                        f"Case {case_index}: original program runtime failure: {reason}.\n"
                        + str(original_case_result.get("stderr", ""))
                    ),
                    original={
                        "compile_success": True,
                        "run_success": False,
                        "compile_time_ms": original_compile_ms,
                        "run_time_ms": total_original_run_ms,
                        "cases": original_cases,
                    },
                    modernized={
                        "compile_success": True,
                        "run_success": False,
                        "compile_time_ms": modernized_compile_ms,
                        "run_time_ms": total_modernized_run_ms,
                        "cases": modernized_cases,
                    },
                    gpp_exe=compiler_path,
                    sanitizer_clean=False,
                    sanitizer_findings=all_sanitizer_findings if all_sanitizer_findings else None,
                    test_cases_run=case_index + 1,
                    failed_case_index=case_index,
                    performance_delta_ms=total_modernized_run_ms - total_original_run_ms,
                ).__dict__

            if not modernized_case_result.get("run_success", False):
                reason = str(modernized_case_result.get("crash_reason") or "modernized runtime failure")
                return DifferentialTestResult(
                    parity_ok=False,
                    diff_text=(
                        f"Case {case_index}: modernized program runtime failure: {reason}.\n"
                        + str(modernized_case_result.get("stderr", ""))
                    ),
                    original={
                        "compile_success": True,
                        "run_success": True,
                        "compile_time_ms": original_compile_ms,
                        "run_time_ms": total_original_run_ms,
                        "cases": original_cases,
                    },
                    modernized={
                        "compile_success": True,
                        "run_success": False,
                        "compile_time_ms": modernized_compile_ms,
                        "run_time_ms": total_modernized_run_ms,
                        "cases": modernized_cases,
                    },
                    gpp_exe=compiler_path,
                    sanitizer_clean=False,
                    sanitizer_findings=all_sanitizer_findings if all_sanitizer_findings else None,
                    test_cases_run=case_index + 1,
                    failed_case_index=case_index,
                    performance_delta_ms=total_modernized_run_ms - total_original_run_ms,
                ).__dict__

            norm_orig = _normalize_output(str(original_case_result.get("stdout", "")))
            norm_mod = _normalize_output(str(modernized_case_result.get("stdout", "")))
            norm_orig_err = _normalize_output(str(original_case_result.get("stderr", "")))
            norm_mod_err = _normalize_output(str(modernized_case_result.get("stderr", "")))
            original_exit = int(original_case_result.get("exit_code", 0) or 0)
            modernized_exit = int(modernized_case_result.get("exit_code", 0) or 0)

            sanitizer_clean = len(case_findings) == 0
            outputs_match = norm_orig == norm_mod
            stderr_match = norm_orig_err == norm_mod_err
            exit_match = original_exit == modernized_exit

            if outputs_match and stderr_match and exit_match and sanitizer_clean:
                continue

            stdout_diff = "".join(
                unified_diff(
                    (norm_orig + "\n").splitlines(keepends=True),
                    (norm_mod + "\n").splitlines(keepends=True),
                    fromfile=f"case_{case_index}_original_stdout",
                    tofile=f"case_{case_index}_modernized_stdout",
                )
            )
            stderr_diff = "".join(
                unified_diff(
                    (norm_orig_err + "\n").splitlines(keepends=True),
                    (norm_mod_err + "\n").splitlines(keepends=True),
                    fromfile=f"case_{case_index}_original_stderr",
                    tofile=f"case_{case_index}_modernized_stderr",
                )
            )

            diff_parts: list[str] = [f"Case {case_index} mismatch diagnostics:\n"]
            if not outputs_match:
                diff_parts.append("stdout diff:\n" + stdout_diff)
            if not stderr_match:
                diff_parts.append("stderr diff:\n" + stderr_diff)
            if not exit_match:
                diff_parts.append(
                    f"exit code mismatch: original={original_exit}, modernized={modernized_exit}\n"
                )
            if not sanitizer_clean:
                diff_parts.append(
                    "sanitizer diagnostics:\n" + "\n".join([str(item) for item in case_findings]) + "\n"
                )

            memory_delta_kb: int | None = None
            if max_original_peak is not None and max_modernized_peak is not None:
                memory_delta_kb = max_modernized_peak - max_original_peak

            return DifferentialTestResult(
                parity_ok=False,
                diff_text="".join(diff_parts),
                original={
                    "compile_success": True,
                    "run_success": True,
                    "compile_time_ms": original_compile_ms,
                    "run_time_ms": total_original_run_ms,
                    "cases": original_cases,
                },
                modernized={
                    "compile_success": True,
                    "run_success": True,
                    "compile_time_ms": modernized_compile_ms,
                    "run_time_ms": total_modernized_run_ms,
                    "cases": modernized_cases,
                },
                gpp_exe=compiler_path,
                sanitizer_clean=False if all_sanitizer_findings else True,
                sanitizer_findings=all_sanitizer_findings if all_sanitizer_findings else None,
                memory_delta_kb=memory_delta_kb,
                test_cases_run=case_index + 1,
                failed_case_index=case_index,
                performance_delta_ms=total_modernized_run_ms - total_original_run_ms,
            ).__dict__

        memory_delta_kb: int | None = None
        if max_original_peak is not None and max_modernized_peak is not None:
            memory_delta_kb = max_modernized_peak - max_original_peak

        return DifferentialTestResult(
            parity_ok=len(all_sanitizer_findings) == 0,
            diff_text="" if len(all_sanitizer_findings) == 0 else (
                "All outputs matched, but sanitizer detected issues:\n"
                + "\n".join(all_sanitizer_findings)
            ),
            original={
                "compile_success": True,
                "run_success": True,
                "compile_time_ms": original_compile_ms,
                "run_time_ms": total_original_run_ms,
                "cases": original_cases,
            },
            modernized={
                "compile_success": True,
                "run_success": True,
                "compile_time_ms": modernized_compile_ms,
                "run_time_ms": total_modernized_run_ms,
                "cases": modernized_cases,
            },
            gpp_exe=compiler_path,
            sanitizer_clean=len(all_sanitizer_findings) == 0,
            sanitizer_findings=all_sanitizer_findings if all_sanitizer_findings else None,
            memory_delta_kb=memory_delta_kb,
            test_cases_run=len(effective_input_cases),
            failed_case_index=None,
            performance_delta_ms=total_modernized_run_ms - total_original_run_ms,
        ).__dict__
