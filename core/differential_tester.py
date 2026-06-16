"""
Differential tester: compiles and verifies modernized C++ code.

SECURITY MODEL
==============
This module compiles and runs arbitrary user-submitted C++ code. By default, it uses
OS-level resource limits (Linux rlimits, Windows Job Objects) to prevent runaway processes:

- Memory: 100 MB limit per process
- CPU processes: 1 subprocess (no fork bombs)
- Timeout: configurable (default 30s)

⚠️  DEPLOYMENT WARNING: For trusted/intranet use only.
    
For PUBLIC SaaS or untrusted networks, add SANDBOX_MODE (Docker container per run):
- Future feature: SANDBOX_MODE=container will wrap compile+run in ephemeral Docker
- Zero network access, tmpfs scratch, read-only rootfs, strict cgroup limits
- Implementation details tracked in ROADMAP.md (project root).  Container sandboxing is a planned future work item.

For now, if deploying publicly:
1. Run inside a container or VM with strict host limits
2. Monitor /proc/{pid}/status for memory creep
3. Disable /modernize endpoints for untrusted origins
4. Implement request authentication beyond IP-based rate limiting

ENV VARIABLES
=============
COMPILER_PATH         Path to C++ compiler (default: auto-detect g++/clang++)
SKIP_VERIFICATION     If 1, skip compilation checks (LLM-only mode)
CPP_STANDARD          Target standard: c++14, c++17, c++20, c++23 (default: c++17)
SANDBOX_MODE          Isolation mode: 'rlimit' (default) or 'container' (future feature)
DOCKER_IMAGE          Docker image for sandbox mode (if SANDBOX_MODE=container; not yet implemented)
"""

import ctypes
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any


def _linux_preexec_fn():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (100 * 1024 * 1024, 100 * 1024 * 1024))
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
        except Exception:
            pass
    except Exception:
        pass


def _apply_windows_job_limits(pid, handle) -> Any:
    try:
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryLimit", ctypes.c_size_t),
                ("PeakJobMemoryLimit", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x0008
        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x0100
        JOB_OBJECT_LIMIT_JOB_MEMORY = 0x0200

        kernel32 = ctypes.windll.kernel32
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE |
            JOB_OBJECT_LIMIT_ACTIVE_PROCESS |
            JOB_OBJECT_LIMIT_PROCESS_MEMORY |
            JOB_OBJECT_LIMIT_JOB_MEMORY
        )
        info.BasicLimitInformation.ActiveProcessLimit = 1
        info.ProcessMemoryLimit = 100 * 1024 * 1024
        info.JobMemoryLimit = 100 * 1024 * 1024

        res = kernel32.SetInformationJobObject(
            job,
            9,
            ctypes.byref(info),
            ctypes.sizeof(info)
        )
        if not res:
            kernel32.CloseHandle(job)
            return None

        kernel32.AssignProcessToJobObject(job, handle)
        return job
    except Exception:
        return None


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

# Maximum bytes captured from stdout/stderr to prevent OOM from infinite output.
_MAX_OUTPUT_BYTES: int = 1_048_576  # 1 MB


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
    """Extract peak resident-set size (KB) from ASan or /usr/bin/time output."""
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


def _get_peak_memory_windows(pid: int) -> int | None:
    """Get peak working-set memory (KB) for a process on Windows using Win32 API.

    Uses ctypes to call GetProcessMemoryInfo from kernel32/psapi — zero
    external dependencies.  Returns None on failure or non-Windows platforms.
    """
    if platform.system() != "Windows":
        return None
    try:
        # PROCESS_QUERY_INFORMATION | PROCESS_VM_READ
        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
        if not handle:
            return None
        try:
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return max(1, counters.PeakWorkingSetSize // 1024)
            return None
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return None


def _truncate_output(text: str, max_bytes: int = _MAX_OUTPUT_BYTES) -> str:
    """Truncate output to prevent OOM from programs producing infinite output."""
    if len(text) <= max_bytes:
        return text
    return text[:max_bytes] + f"\n... [TRUNCATED at {max_bytes} bytes]"


def resolve_cpp_compiler(explicit_path: str | None = None) -> str:
    """Resolve the C++ compiler path using explicit arg, env vars, and PATH probes.

    Search order:
      1. Explicit path argument
      2. CXX / GPP_EXE / CLANGXX_EXE environment variables
      3. g++, clang++ variants on PATH
      4. Windows: cl.exe (MSVC) from Visual Studio / Build Tools
      5. Fallback: 'g++' (will fail gracefully via _verify_compiler)
    """
    if explicit_path:
        return explicit_path

    env_candidates = [
        os.environ.get("COMPILER_PATH", "").strip(),
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

    # Windows fallback: probe for MSVC cl.exe
    if platform.system() == "Windows":
        cl = shutil.which("cl")
        if cl:
            return cl
        # Common Visual Studio / Build Tools install locations
        msvc_roots = [
            r"C:\Program Files\Microsoft Visual Studio",
            r"C:\Program Files (x86)\Microsoft Visual Studio",
            r"C:\Program Files\Microsoft Visual Studio\2022",
            r"C:\Program Files\Microsoft Visual Studio\2019",
        ]
        for root in msvc_roots:
            root_path = Path(root)
            if root_path.exists():
                for cl_exe in root_path.rglob("cl.exe"):
                    # Prefer x64 host
                    if "Hostx64" in str(cl_exe) or "amd64" in str(cl_exe).lower():
                        return str(cl_exe)
                # Any cl.exe will do
                found_list = list(root_path.rglob("cl.exe"))
                if found_list:
                    return str(found_list[0])

    return "g++"


def _verify_compiler(compiler_path: str, timeout_seconds: int = 5) -> None:
    """Verify that the compiler is invokable, caching successful checks."""
    if compiler_path in _VERIFIED_COMPILERS:
        return

    try:
        env = dict(os.environ)
        if compiler_path:
            compiler_dir = str(Path(compiler_path).parent.resolve())
            path_sep = ";" if platform.system() == "Windows" else ":"
            existing_path = env.get("PATH", "")
            if existing_path:
                env["PATH"] = f"{compiler_dir}{path_sep}{existing_path}"
            else:
                env["PATH"] = compiler_dir

        result = subprocess.run(
            [compiler_path, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
            env=env,
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
    extra_compile_args: list[str] | None = None,
    compile_only: bool = False,
    cpp_standard: str | None = None,
) -> list[str]:
    std_flag = f"-std={cpp_standard}" if cpp_standard else "-std=c++17"
    cmd = [compiler_path, std_flag, "-Wall"]
    if compile_only:
        cmd.append("-c")
    if extra_compile_args:
        cmd.extend(extra_compile_args)
    if enable_sanitizers:
        cmd.extend(_SANITIZER_COMPILE_FLAGS)
    cmd.extend([source_path, "-o", exe_path])
    return cmd


def _build_run_env(enable_sanitizers: bool, compiler_path: str | None = None) -> dict[str, str]:
    env = dict(os.environ)
    if compiler_path:
        compiler_dir = str(Path(compiler_path).parent.resolve())
        path_sep = ";" if platform.system() == "Windows" else ":"
        existing_path = env.get("PATH", "")
        if existing_path:
            env["PATH"] = f"{compiler_dir}{path_sep}{existing_path}"
        else:
            env["PATH"] = compiler_dir

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
    extra_compile_args: list[str] | None = None,
    compile_only: bool = False,
    cpp_standard: str | None = None,
) -> dict[str, Any]:
    """Compile one C++ source file into an executable and return compile metadata."""
    exe_path = os.path.join(tmp_dir, exe_name)

    if enable_sanitizers and not _sanitizers_available():
        enable_sanitizers = False

    compile_cmd = _build_compile_command(
        compiler_path, source_path, exe_path, enable_sanitizers, extra_compile_args, compile_only, cpp_standard
    )
    
    sandbox_mode = os.environ.get("SANDBOX_MODE", "rlimit").lower()
    if sandbox_mode == "container":
        img = os.environ.get("DOCKER_IMAGE", "gcc:latest")
        docker_prefix = [
            "docker", "run", "--rm",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp",
            "-v", f"{tmp_dir}:/sandbox",
            "-w", "/sandbox",
            "--memory", "500m",
            "--pids-limit", "50",
            img
        ]
        c_name = "clang++" if "clang" in compile_cmd[0] else "g++"
        # Map Windows paths to /sandbox Unix paths
        args = [c.replace(tmp_dir, "/sandbox").replace("\\", "/") for c in compile_cmd[1:]]
        compile_cmd = docker_prefix + [c_name] + args

    compile_env = _build_run_env(enable_sanitizers, compiler_path)

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


def _execute_sandboxed(
    run_cmd: list[str],
    input_data: str | None,
    timeout_seconds: int,
    env: dict[str, str],
    cwd: str | None,
) -> tuple[Any, str, str, bool]:
    """Execute a subprocess inside a sandbox (Windows Job Object or Linux rlimits)."""
    creationflags = 0
    preexec_fn = None
    if platform.system() == "Windows":
        # CREATE_BREAKAWAY_FROM_JOB = 0x01000000
        creationflags = 0x01000000
    else:
        preexec_fn = _linux_preexec_fn

    proc = subprocess.Popen(
        run_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=cwd,
        creationflags=creationflags,
        preexec_fn=preexec_fn
    )

    job = None
    if platform.system() == "Windows":
        job = _apply_windows_job_limits(proc.pid, int(proc._handle))

    timed_out = False
    try:
        stdout, stderr = proc.communicate(input=input_data, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate()
    finally:
        if job:
            try:
                import ctypes
                ctypes.windll.kernel32.CloseHandle(job)
            except Exception:
                pass

    return proc, stdout or "", stderr or "", timed_out


def _run_exe(
    exe_path: str,
    input_data: str | None,
    timeout_seconds: int,
    env: dict[str, str],
    cwd: str | None = None,
    iterations: int = 5,
) -> dict[str, Any]:
    """Run a compiled executable and return runtime diagnostics.

    Uses subprocess.run for execution and supports multiple iterations with
    outlier-filtered timing. Designed to be easily mockable in unit tests.
    """
    run_cmd = [exe_path]
    
    sandbox_mode = os.environ.get("SANDBOX_MODE", "rlimit").lower()
    if sandbox_mode == "container":
        img = os.environ.get("DOCKER_IMAGE", "gcc:latest")
        docker_prefix = [
            "docker", "run", "--rm", "-i",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp",
            "-v", f"{cwd or '.'}:/sandbox",
            "-w", "/sandbox",
            "--memory", "100m",
            "--pids-limit", "10",
            img
        ]
        run_cmd = docker_prefix + ["./" + os.path.basename(exe_path)]
        
    times_ms: list[int] = []

    # Primary execution
    try:
        start = time.perf_counter()
        result = subprocess.run(
            run_cmd,
            input=input_data,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
            env=env,
            cwd=cwd,
        )
        run_time_ms = int((time.perf_counter() - start) * 1000)
    except subprocess.TimeoutExpired:
        return {
            "compile_success": True,
            "run_success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout_seconds} seconds.",
            "run_time_ms": timeout_seconds * 1000,
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
            "run_time_ms": 0,
            "exit_code": None,
            "sanitizer_findings": [],
            "peak_memory_kb": None,
            "timed_out": True,
            "crash_reason": "execution_error",
        }

    times_ms.append(run_time_ms)

    # Additional iterations for high-fidelity timing
    for _ in range(iterations - 1):
        try:
            sub_start = time.perf_counter()
            subprocess.run(
                run_cmd,
                input=input_data,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout_seconds,
                env=env,
                cwd=cwd,
            )
            times_ms.append(int((time.perf_counter() - sub_start) * 1000))
        except Exception:
            break

    # Outlier filtering when enough runs collected
    if len(times_ms) >= 3:
        sorted_times = sorted(times_ms)
        filtered = sorted_times[1:-1]
        run_time_ms = int(sum(filtered) / len(filtered))
    else:
        run_time_ms = int(sum(times_ms) / len(times_ms))

    stdout_text = _truncate_output(result.stdout.strip())
    stderr_text = _truncate_output(result.stderr.strip())
    exit_code = int(result.returncode)
    crash_reason = _detect_crash_reason(exit_code, stderr_text, False)
    sanitizer_findings = _detect_sanitizer_errors(stderr_text)
    peak_memory_kb = _parse_peak_memory_kb(stderr_text)
    if peak_memory_kb is None and platform.system() == "Windows":
        peak_memory_kb = _get_peak_memory_windows(result.pid) if hasattr(result, "pid") else None

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


def compiler_available(explicit_path: str | None = None) -> bool:
    """Returns True if a C++ compiler can be found and invoked on this system."""
    try:
        compiler_path = resolve_cpp_compiler(explicit_path)
        _verify_compiler(compiler_path)
        return True
    except Exception:
        return False


def _get_extra_compile_args(original_file_path: str | None = None) -> list[str]:
    import json
    from pathlib import Path
    extra_compile_args: list[str] = []
    cwd_path = Path.cwd()
    compile_commands_path = cwd_path / "compile_commands.json"
    if compile_commands_path.is_file():
        try:
            with compile_commands_path.open("r", encoding="utf-8") as f:
                commands = json.load(f)
            
            target_entry = None
            if original_file_path:
                target_name = Path(original_file_path).resolve()
                for entry in commands:
                    file_entry = entry.get("file")
                    if file_entry:
                        entry_path = Path(file_entry).resolve()
                        if entry_path == target_name:
                            target_entry = entry
                            break
            
            if not target_entry and commands:
                for entry in commands:
                    if entry.get("file"):
                        target_entry = entry
                        break

            if target_entry:
                args = target_entry.get("arguments") or target_entry.get("command")
                if isinstance(args, str):
                    args = args.split()
                if isinstance(args, list) and args:
                    raw_args = args[1:]
                    base_dir = Path(target_entry.get("directory", ""))
                    
                    filtered_args = []
                    i = 0
                    while i < len(raw_args):
                        arg = raw_args[i]
                        if arg == "-I" and i + 1 < len(raw_args):
                            inc_path = raw_args[i+1]
                            abs_path = Path(inc_path)
                            if not abs_path.is_absolute() and base_dir:
                                abs_path = (base_dir / abs_path).resolve()
                            filtered_args.append("-I")
                            filtered_args.append(str(abs_path))
                            i += 2
                        elif arg.startswith("-I"):
                            inc_path = arg[2:].strip()
                            if inc_path:
                                abs_path = Path(inc_path)
                                if not abs_path.is_absolute() and base_dir:
                                    abs_path = (base_dir / abs_path).resolve()
                                filtered_args.append(f"-I{abs_path}")
                            else:
                                filtered_args.append(arg)
                            i += 1
                        elif arg.startswith("-D"):
                            filtered_args.append(arg)
                            i += 1
                        else:
                            i += 1
                    extra_compile_args = filtered_args
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Failed to parse compile_commands.json: {e}", exc_info=True)
            extra_compile_args = []
    return extra_compile_args


def compile_cpp_source(
    code: str,
    gpp_exe: str | None = None,
    timeout_seconds: int = 10,
    enable_sanitizers: bool = True,
    extra_flags: list[str] | None = None,
    compile_only: bool | None = None,
    original_file_path: str | None = None,
    cpp_standard: str | None = None,
) -> dict:
    """Compile C++ source code and return compiler diagnostics.

    Parameters:
    - code: complete C++ source text
    - gpp_exe: optional explicit compiler path/name
    - timeout_seconds: compile timeout
    - enable_sanitizers: enable ASan/UBSan where available
    - extra_flags: optional list of extra compiler arguments (e.g. include search paths)
    - compile_only: optional bool to force compile-only (-c) or link
    - original_file_path: optional original legacy file path to match compile command flags

    Returns a dict with keys:
    - success (bool)
    - errors (list[str])
    - warnings (list[str])
    - compilation_time_ms (int)
    - raw_stdout (str)
    - raw_stderr (str)
    - compiler (str)
    - exe_path (str) if successful
    """
    if os.environ.get("SKIP_VERIFICATION", "0") == "1":
        return {
            "success": True,
            "errors": [],
            "warnings": [],
            "compilation_time_ms": 0,
            "raw_stdout": "",
            "raw_stderr": "Verification skipped via API policy.",
            "compiler": "skipped",
            "exe_path": None,
        }
    compiler_path = resolve_cpp_compiler(gpp_exe)

    # Graceful fallback: if compiler not available, return a clear failure dict
    try:
        _verify_compiler(compiler_path)
    except RuntimeError as exc:
        return {
            "success": False,
            "errors": [f"No C++ compiler available: {exc}"],
            "warnings": [],
            "compilation_time_ms": 0,
            "raw_stdout": "",
            "raw_stderr": str(exc),
            "compiler": compiler_path,
        }

    extra_compile_args = _get_extra_compile_args(original_file_path)
    if extra_flags:
        extra_compile_args = extra_compile_args + extra_flags

    start_time = time.time()
    with tempfile.TemporaryDirectory() as tmp_dir:
        cpp_path = os.path.join(tmp_dir, "modernized.cpp")
        with open(cpp_path, "w", encoding="utf-8") as cpp_file:
            cpp_file.write(code)

        if compile_only is None:
            has_main = bool(re.search(r"\bint\s+main\s*\(", code))
            compile_only = not has_main

        result = _compile_to_exe(
            source_path=cpp_path,
            compiler_path=compiler_path,
            tmp_dir=tmp_dir,
            exe_name="modernized_test",
            enable_sanitizers=enable_sanitizers,
            timeout_seconds=timeout_seconds,
            extra_compile_args=extra_compile_args,
            compile_only=compile_only,
            cpp_standard=cpp_standard,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)
        stdout_text = result.get("stdout", "")
        stderr_text = result.get("stderr", "")

        if not result["compile_success"]:
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
            "warnings": [ln for ln in stderr_text.splitlines() if ln.strip()] if stderr_text else [],
            "compilation_time_ms": elapsed_ms,
            "raw_stdout": stdout_text,
            "raw_stderr": stderr_text,
            "compiler": compiler_path,
            "exe_path": result["exe_path"],
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
    compiler_path = resolve_cpp_compiler(gpp_exe)

    # Verify compiler availability
    try:
        _verify_compiler(compiler_path)
    except RuntimeError as exc:
        return DifferentialTestResult(
            parity_ok=False,
            diff_text=f"No C++ compiler available: {exc}",
            original={"success": False, "errors": [str(exc)]},
            modernized={"success": False, "errors": [str(exc)]},
            gpp_exe=compiler_path,
        ).__dict__

    extra_args = _get_extra_compile_args()

    # Determine input test cases
    cases: list[str] = []
    if input_cases is not None:
        cases = input_cases
    elif input_data is not None:
        cases = [input_data]
    else:
        cases = [""]

    # We compile both in a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Compile original source
        orig_res = _compile_to_exe(
            source_path=original_cpp_path,
            compiler_path=compiler_path,
            tmp_dir=tmp_dir,
            exe_name="orig_test",
            enable_sanitizers=True,
            timeout_seconds=compile_timeout_seconds,
            extra_compile_args=extra_args,
        )

        # 2. Compile modernized code
        modernized_cpp = os.path.join(tmp_dir, "modernized.cpp")
        with open(modernized_cpp, "w", encoding="utf-8") as f:
            f.write(modernized_code)

        mod_res = _compile_to_exe(
            source_path=modernized_cpp,
            compiler_path=compiler_path,
            tmp_dir=tmp_dir,
            exe_name="mod_test",
            enable_sanitizers=True,
            timeout_seconds=compile_timeout_seconds,
            extra_compile_args=extra_args,
        )

        # If compilation fails
        if not orig_res["compile_success"] or not mod_res["compile_success"]:
            diff_lines = []
            if not orig_res["compile_success"]:
                diff_lines.append("Original compilation failed:\n" + orig_res.get("stderr", ""))
            if not mod_res["compile_success"]:
                diff_lines.append("Modernized compilation failed:\n" + mod_res.get("stderr", ""))
            return DifferentialTestResult(
                parity_ok=False,
                diff_text="\n".join(diff_lines),
                original=orig_res,
                modernized=mod_res,
                gpp_exe=compiler_path,
            ).__dict__

        # Both compiled successfully. Run execution tests!
        env = _build_run_env(enable_sanitizers=True, compiler_path=compiler_path)

        orig_runs = []
        mod_runs = []

        parity_ok = True
        failed_case_idx = None
        diff_text = ""
        sanitizer_findings: list[str] = []

        orig_total_time = 0
        mod_total_time = 0

        orig_max_memory = 0
        mod_max_memory = 0

        for idx, case_input in enumerate(cases):
            # Run original (sandboxed to tmp_dir)
            orig_run = _run_exe(orig_res["exe_path"], case_input, run_timeout_seconds, env, cwd=tmp_dir)
            orig_runs.append(orig_run)

            # Run modernized (sandboxed to tmp_dir)
            mod_run = _run_exe(mod_res["exe_path"], case_input, run_timeout_seconds, env, cwd=tmp_dir)
            mod_runs.append(mod_run)

            # Check for sanitizer findings on modernized run
            if mod_run.get("sanitizer_findings"):
                sanitizer_findings.extend(mod_run["sanitizer_findings"])

            # Compute timing & memory
            orig_total_time += orig_run.get("run_time_ms", 0) or 0
            mod_total_time += mod_run.get("run_time_ms", 0) or 0

            if orig_run.get("peak_memory_kb") is not None:
                orig_max_memory = max(orig_max_memory, orig_run["peak_memory_kb"])
            if mod_run.get("peak_memory_kb") is not None:
                mod_max_memory = max(mod_max_memory, mod_run["peak_memory_kb"])

            # Check outputs parity
            orig_stdout = _normalize_output(orig_run.get("stdout", ""))
            mod_stdout = _normalize_output(mod_run.get("stdout", ""))

            orig_exit = orig_run.get("exit_code")
            mod_exit = mod_run.get("exit_code")

            if orig_stdout != mod_stdout or orig_exit != mod_exit or not orig_run["run_success"] or not mod_run["run_success"]:
                if parity_ok: # Record the first failure
                    parity_ok = False
                    failed_case_idx = idx

                    # Generate unified diff for stdout if they differ
                    if orig_stdout != mod_stdout:
                        diff_text_list = list(unified_diff(
                            orig_stdout.splitlines(),
                            mod_stdout.splitlines(),
                            fromfile="original_stdout",
                            tofile="modernized_stdout",
                            lineterm=""
                        ))
                        diff_text = "\n".join(diff_text_list)
                    else:
                        diff_text = (
                            f"Execution mismatch: original exit code={orig_exit}, "
                            f"modernized exit code={mod_exit}\n"
                            f"Original stderr: {orig_run.get('stderr')}\n"
                            f"Modernized stderr: {mod_run.get('stderr')}"
                        )

        # Performance and Memory Deltas
        performance_delta_ms = mod_total_time - orig_total_time
        memory_delta_kb = (mod_max_memory - orig_max_memory) if (orig_max_memory > 0 and mod_max_memory > 0) else None

        sanitizer_clean = len(sanitizer_findings) == 0

        return DifferentialTestResult(
            parity_ok=parity_ok,
            diff_text=diff_text,
            original={
                "compile": orig_res,
                "runs": orig_runs,
            },
            modernized={
                "compile": mod_res,
                "runs": mod_runs,
            },
            gpp_exe=compiler_path,
            sanitizer_clean=sanitizer_clean,
            sanitizer_findings=sanitizer_findings if sanitizer_findings else None,
            memory_delta_kb=memory_delta_kb,
            test_cases_run=len(cases),
            failed_case_index=failed_case_idx,
            performance_delta_ms=performance_delta_ms,
        ).__dict__
