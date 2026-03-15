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
    compile_timeout_seconds: int = 10,
    run_timeout_seconds: int = 10,
) -> dict:
    exe_path = os.path.join(tmp_dir, exe_name)
    _verify_compiler(gpp_exe)

    if enable_sanitizers and not _sanitizers_available():
        enable_sanitizers = False

    sanitizer_env = dict(os.environ)
    if enable_sanitizers:
        sanitizer_env["ASAN_OPTIONS"] = "detect_leaks=1:print_stats=1:halt_on_error=0"
        sanitizer_env["UBSAN_OPTIONS"] = "print_stacktrace=1:halt_on_error=0"

    compile_start = time.time()
    compile_cmd = [gpp_exe, "-std=c++23", "-Wall", source_path, "-o", exe_path]
    if enable_sanitizers:
        compile_cmd[2:2] = _SANITIZER_COMPILE_FLAGS

    try:
        compile_result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=compile_timeout_seconds,
            env=sanitizer_env,
        )
    except subprocess.TimeoutExpired:
        return {
            "compile_success": False,
            "run_success": False,
            "stdout": "",
            "stderr": f"Compilation timed out after {compile_timeout_seconds} seconds.",
            "compile_time_ms": int((time.time() - compile_start) * 1000),
            "run_time_ms": 0,
            "exit_code": None,
            "sanitizer_findings": [],
            "peak_memory_kb": None,
            "timed_out": True,
            "crash_reason": "",
        }
    except Exception as exc:
        return {
            "compile_success": False,
            "run_success": False,
            "stdout": "",
            "stderr": f"Compilation failed: {exc!r}",
            "compile_time_ms": int((time.time() - compile_start) * 1000),
            "run_time_ms": 0,
            "exit_code": None,
            "sanitizer_findings": [],
            "peak_memory_kb": None,
            "timed_out": False,
            "crash_reason": "",
        }

    compile_time_ms = int((time.time() - compile_start) * 1000)
    if compile_result.returncode != 0:
        return {
            "compile_success": False,
            "run_success": False,
            "stdout": (compile_result.stdout or "").strip(),
            "stderr": (compile_result.stderr or "").strip(),
            "compile_time_ms": compile_time_ms,
            "run_time_ms": 0,
            "exit_code": None,
            "sanitizer_findings": [],
            "peak_memory_kb": None,
            "timed_out": False,
            "crash_reason": "",
        }

    run_start = time.time()
    use_time_v = platform.system() != "Windows" and os.path.isfile("/usr/bin/time")
    run_cmd = [exe_path]
    if use_time_v:
        run_cmd = ["/usr/bin/time", "-v", exe_path]

    try:
        run_result = subprocess.run(
            run_cmd,
            input=input_data if input_data is not None else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=run_timeout_seconds,
            env=sanitizer_env,
        )
    except subprocess.TimeoutExpired:
        return {
            "compile_success": True,
            "run_success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {run_timeout_seconds} seconds.",
            "compile_time_ms": compile_time_ms,
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
            "compile_time_ms": compile_time_ms,
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

    crash_reason = ""
    exit_code = int(run_result.returncode)
    if exit_code < 0:
        signal_name = {
            6: "abort()",
            4: "illegal instruction",
            8: "floating point exception",
            11: "segmentation fault",
        }.get(-exit_code, f"signal {-exit_code}")
        crash_reason = f"Process crashed with {signal_name}."
    elif exit_code in {132, 134, 136, 139}:
        crash_reason = {
            132: "Process crashed: illegal instruction.",
            134: "Process crashed: abort().",
            136: "Process crashed: floating point exception.",
            139: "Process crashed: segmentation fault.",
        }[exit_code]
    elif re.search(r"segmentation fault|illegal instruction|floating point exception|aborted", stderr_text, re.IGNORECASE):
        crash_reason = "Process crashed (detected from stderr)."

    sanitizer_findings = _detect_sanitizer_errors(stderr_text)
    peak_memory_kb = _parse_peak_memory_kb(stderr_text)
    run_success = exit_code == 0 and not crash_reason

    return {
        "compile_success": True,
        "run_success": run_success,
        "exit_code": exit_code,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "compile_time_ms": compile_time_ms,
        "run_time_ms": run_time_ms,
        "sanitizer_findings": sanitizer_findings,
        "peak_memory_kb": peak_memory_kb,
        "timed_out": False,
        "crash_reason": crash_reason,
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
            test_cases_run=0,
        ).__dict__

    effective_input_cases = input_cases if input_cases is not None else []
    if not effective_input_cases:
        effective_input_cases = [input_data if input_data is not None else ""]

    with tempfile.TemporaryDirectory() as tmp_dir:
        modernized_cpp_path = os.path.join(tmp_dir, "modernized.cpp")

        with open(modernized_cpp_path, "w", encoding="utf-8") as f:
            f.write(modernized_code)

        original_compile_probe = _compile_and_run_cpp(
            original_cpp_path,
            compiler,
            tmp_dir,
            "original.exe",
            input_data=effective_input_cases[0],
            enable_sanitizers=False,  # Don't sanitize legacy code—only the modernized version.
            compile_timeout_seconds=compile_timeout_seconds,
            run_timeout_seconds=run_timeout_seconds,
        )

        if not original_compile_probe["compile_success"]:
            return DifferentialTestResult(
                parity_ok=False,
                diff_text="Original code failed to compile. Please fix legacy source before differential testing.",
                original=original_compile_probe,
                modernized={
                    "compile_success": False,
                    "run_success": False,
                    "stdout": "",
                    "stderr": "",
                    "compile_time_ms": 0,
                    "run_time_ms": 0,
                },
                gpp_exe=compiler,
                test_cases_run=0,
            ).__dict__

        modernized_compile_probe = _compile_and_run_cpp(
            modernized_cpp_path,
            compiler,
            tmp_dir,
            "modernized.exe",
            input_data=effective_input_cases[0],
            enable_sanitizers=True,
            compile_timeout_seconds=compile_timeout_seconds,
            run_timeout_seconds=run_timeout_seconds,
        )

        if not modernized_compile_probe["compile_success"]:
            location = _extract_error_location(
                modernized_compile_probe["stderr"], os.path.basename(modernized_cpp_path)
            )
            if location:
                modernized_compile_probe["stderr"] = (
                    modernized_compile_probe["stderr"] + "\n" + f"First error location: {location}"
                )
            return DifferentialTestResult(
                parity_ok=False,
                diff_text="Modernized code failed to compile. See compiler diagnostics.",
                original=original_compile_probe,
                modernized=modernized_compile_probe,
                gpp_exe=compiler,
                test_cases_run=0,
            ).__dict__

        original_compile_ms = int(original_compile_probe.get("compile_time_ms", 0) or 0)
        modernized_compile_ms = int(modernized_compile_probe.get("compile_time_ms", 0) or 0)
        original_cases: list[dict] = []
        modernized_cases: list[dict] = []
        all_sanitizer_findings: list[str] = []
        total_original_run_ms = 0
        total_modernized_run_ms = 0
        max_original_peak: int | None = None
        max_modernized_peak: int | None = None

        for case_index, case_input in enumerate(effective_input_cases):
            original_case_result = _compile_and_run_cpp(
                original_cpp_path,
                compiler,
                tmp_dir,
                f"original_case_{case_index}.exe",
                input_data=case_input,
                enable_sanitizers=False,
                compile_timeout_seconds=compile_timeout_seconds,
                run_timeout_seconds=run_timeout_seconds,
            )
            modernized_case_result = _compile_and_run_cpp(
                modernized_cpp_path,
                compiler,
                tmp_dir,
                f"modernized_case_{case_index}.exe",
                input_data=case_input,
                enable_sanitizers=True,
                compile_timeout_seconds=compile_timeout_seconds,
                run_timeout_seconds=run_timeout_seconds,
            )

            original_cases.append(original_case_result)
            modernized_cases.append(modernized_case_result)

            total_original_run_ms += int(original_case_result.get("run_time_ms", 0) or 0)
            total_modernized_run_ms += int(modernized_case_result.get("run_time_ms", 0) or 0)

            orig_peak_case = original_case_result.get("peak_memory_kb")
            mod_peak_case = modernized_case_result.get("peak_memory_kb")
            if isinstance(orig_peak_case, int):
                max_original_peak = orig_peak_case if max_original_peak is None else max(max_original_peak, orig_peak_case)
            if isinstance(mod_peak_case, int):
                max_modernized_peak = mod_peak_case if max_modernized_peak is None else max(max_modernized_peak, mod_peak_case)

            case_findings = modernized_case_result.get("sanitizer_findings") or []
            all_sanitizer_findings.extend([str(item) for item in case_findings])

            if not original_case_result.get("compile_success", False):
                return DifferentialTestResult(
                    parity_ok=False,
                    diff_text=(
                        f"Case {case_index}: original program failed to compile.\n"
                        + str(original_case_result.get("stderr", ""))
                    ),
                    original={
                        "compile_success": False,
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
                    gpp_exe=compiler,
                    sanitizer_clean=False,
                    sanitizer_findings=all_sanitizer_findings if all_sanitizer_findings else None,
                    test_cases_run=case_index + 1,
                    failed_case_index=case_index,
                    performance_delta_ms=total_modernized_run_ms - total_original_run_ms,
                ).__dict__

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
                    gpp_exe=compiler,
                    sanitizer_clean=False,
                    sanitizer_findings=all_sanitizer_findings if all_sanitizer_findings else None,
                    test_cases_run=case_index + 1,
                    failed_case_index=case_index,
                    performance_delta_ms=total_modernized_run_ms - total_original_run_ms,
                ).__dict__

            if not modernized_case_result.get("compile_success", False):
                return DifferentialTestResult(
                    parity_ok=False,
                    diff_text=(
                        f"Case {case_index}: modernized program failed to compile.\n"
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
                        "compile_success": False,
                        "run_success": False,
                        "compile_time_ms": modernized_compile_ms,
                        "run_time_ms": total_modernized_run_ms,
                        "cases": modernized_cases,
                    },
                    gpp_exe=compiler,
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
                    gpp_exe=compiler,
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
            orig_exit = int(original_case_result.get("exit_code", 0) or 0)
            mod_exit = int(modernized_case_result.get("exit_code", 0) or 0)

            sanitizer_clean = len(case_findings) == 0
            outputs_match = norm_orig == norm_mod
            stderr_match = norm_orig_err == norm_mod_err
            exit_match = orig_exit == mod_exit

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
                    f"exit code mismatch: original={orig_exit}, modernized={mod_exit}\n"
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
                gpp_exe=compiler,
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
            gpp_exe=compiler,
            sanitizer_clean=len(all_sanitizer_findings) == 0,
            sanitizer_findings=all_sanitizer_findings if all_sanitizer_findings else None,
            memory_delta_kb=memory_delta_kb,
            test_cases_run=len(effective_input_cases),
            failed_case_index=None,
            performance_delta_ms=total_modernized_run_ms - total_original_run_ms,
        ).__dict__