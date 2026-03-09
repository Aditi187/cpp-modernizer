"""
Differential Tester for the Code Modernization Engine.  # This top-level string explains that this module verifies functional parity between original and modernized code.
Compiles and runs both the original and modernized programs, then compares their outputs.  # This line describes the core behavior in plain English.
"""  # This closing triple quote ends the module-level documentation.

import os  # This import lets us work with file paths and check that files exist.
import subprocess  # This import allows us to run the compiler and the compiled executables.
import tempfile  # This import helps us create temporary directories and files for compiling the modernized code.
from difflib import unified_diff  # This import brings in the unified_diff function so we can show exactly what changed when outputs differ.


def _compile_and_run_cpp(
    source_path: str,
    gpp_exe: str,
    tmp_dir: str,
    exe_name: str,
) -> tuple[str, str, bool]:
    """
    Compile a C++ source file and run the resulting executable.  # This docstring explains the purpose of this helper.
    Returns (stdout_text, stderr_text, success).  # This line documents the return value in plain English.
    """
    exe_path = os.path.join(tmp_dir, exe_name)  # This line builds the full path where the compiled executable will be written.

    compile_cmd = f'"{gpp_exe}" -std=c++20 -Wall "{source_path}" -o "{exe_path}"'  # This line builds the g++ command string with quoted paths for safety.

    try:  # This try block attempts to compile the source file.
        result = subprocess.run(  # This call runs the g++ compiler as a child process.
            compile_cmd,  # This argument is the full compile command that the shell will execute.
            shell=True,  # This flag runs the command through the system shell.
            capture_output=True,  # This flag captures both stdout and stderr from the compiler.
            text=True,  # This flag decodes the captured output as text strings.
            timeout=10,  # This flag enforces a hard 10-second timeout so compilation cannot hang indefinitely.
        )  # This closing parenthesis ends the subprocess.run call.
    except subprocess.TimeoutExpired as exc:  # This except block catches the case where compilation takes too long.
        return "", f"Compilation timed out: {exc!r}", False  # This return reports the timeout as a compilation failure.
    except Exception as exc:  # This except block catches any error while trying to run the compiler.
        return "", f"Compilation failed: {exc!r}", False  # This return gives an empty stdout, an error message in stderr, and success=False.

    if result.returncode != 0:  # This condition checks whether the compiler reported an error (non-zero exit code).
        stderr_text = (result.stderr or "").strip()  # This line gets the compiler's error output so we can report it.
        return "", stderr_text, False  # This return signals that compilation failed and includes the compiler error text.

    try:  # This try block attempts to run the compiled executable.
        run_result = subprocess.run(  # This call executes the compiled program and waits for it to finish.
            exe_path,  # This argument is the path to the executable we just compiled.
            shell=True,  # This flag runs the executable through the shell so it can be found and executed.
            capture_output=True,  # This flag captures the program's stdout and stderr.
            text=True,  # This flag decodes the captured output as text.
            timeout=10,  # This flag enforces a hard 10-second timeout to protect against infinite loops in generated code.
        )  # This closing parenthesis ends the subprocess.run call.
    except subprocess.TimeoutExpired as exc:  # This except block catches the case where the executable runs for too long.
        return "", f"Execution timed out: {exc!r}", False  # This return reports the timeout as an execution failure.
    except Exception as exc:  # This except block catches any other error while trying to run the executable.
        return "", f"Execution failed: {exc!r}", False  # This return reports the execution failure with an error message.

    stdout_text = (run_result.stdout or "").strip()  # This line extracts and trims the program's standard output.
    stderr_text = (run_result.stderr or "").strip()  # This line extracts and trims the program's standard error.
    return stdout_text, stderr_text, True  # This return passes back the outputs and success=True because both compile and run succeeded.


def _normalize_output(text: str) -> str:  # This helper normalizes program output for fuzzy comparison.
    """
    Normalize output by:
      - converting all newlines to '\\n',
      - stripping trailing whitespace from each line,
      - and trimming trailing blank lines.  # This docstring explains that we ignore superficial formatting differences.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")  # This line normalizes different newline conventions to '\\n'.
    lines = normalized.split("\n")  # This line splits into logical lines without newline characters.
    stripped_lines = [line.rstrip() for line in lines]  # This line removes trailing spaces and tabs from each line.
    # Drop trailing blank lines to avoid spurious differences at the end of output.
    while stripped_lines and stripped_lines[-1] == "":
        stripped_lines.pop()
    return "\n".join(stripped_lines)  # This line rejoins the normalized lines into a single string.


def run_differential_test(
    original_cpp_path: str,
    modernized_code: str,
    gpp_exe: str = r"C:\msys64\mingw64\bin\g++.exe",
) -> tuple[bool, str]:
    """
    Run the differential test: compile and run both original and modernized code, then compare outputs.  # This docstring explains the main entry point.
    Returns (parity_ok, diff_text) where diff_text is empty on success, or a unified diff
    string when outputs differ.  # This line documents the return value.
    """  # This closing triple quote ends the docstring.

    if not os.path.isfile(original_cpp_path):  # This check ensures the original C++ file exists before we try to compile it.
        print(f"ERROR: Original file not found: {original_cpp_path}")  # This print informs the user that the path is invalid.
        return False, ""  # This return signals failure so the caller knows the test could not run.

    if not modernized_code.strip():  # This check ensures we have actual modernized code to test; empty input would be meaningless.
        print("ERROR: No modernized code provided.")  # This print tells the user that the modernized code string was empty.
        return False, ""  # This return signals failure so the caller knows the test could not run.

    with tempfile.TemporaryDirectory() as tmp_dir:  # This context manager creates a temporary directory that will be cleaned up when we are done.
        modernized_cpp_path = os.path.join(tmp_dir, "modernized.cpp")  # This line builds the path for the temporary modernized source file.

        with open(modernized_cpp_path, "w", encoding="utf-8") as f:  # This line opens the file for writing so we can save the modernized code to it.
            f.write(modernized_code)  # This line writes the modernized code string into the temporary file so it can be compiled.

        print("Compiling and running ORIGINAL (test.cpp)...")  # This print tells the user that we are now processing the original code.
        orig_stdout, orig_stderr, orig_ok = _compile_and_run_cpp(  # This line calls our helper to compile and run the original test.cpp file.
            original_cpp_path,  # This argument is the path to the original C++ source file.
            gpp_exe,  # This argument is the path to the g++ compiler executable.
            tmp_dir,  # This argument is the temporary directory where we will create the original executable.
            "original.exe",  # This argument is the name for the compiled original executable.
        )  # This closing parenthesis ends the call to _compile_and_run_cpp.

        if not orig_ok:  # This condition checks whether the original code failed to compile or run.
            print("ORIGINAL FAILED:")  # This print marks the section that shows why the original failed.
            print(orig_stderr)  # This print shows the compiler or runtime error from the original code.
            return False, ""  # This return stops the test because we cannot compare outputs if the original did not run successfully.

        print("Compiling and running MODERNIZED code...")  # This print tells the user that we are now processing the modernized code.
        mod_stdout, mod_stderr, mod_ok = _compile_and_run_cpp(  # This line calls our helper to compile and run the modernized code.
            modernized_cpp_path,  # This argument is the path to the temporary file holding the modernized code.
            gpp_exe,  # This argument is the path to the g++ compiler executable.
            tmp_dir,  # This argument is the temporary directory where we will create the modernized executable.
            "modernized.exe",  # This argument is the name for the compiled modernized executable.
        )  # This closing parenthesis ends the call to _compile_and_run_cpp.

        if not mod_ok:  # This condition checks whether the modernized code failed to compile or run.
            print("MODERNIZED FAILED:")  # This print marks the section that shows why the modernized code failed.
            print(mod_stderr)  # This print shows the compiler or runtime error from the modernized code.
            return False, ""  # This return stops the test because we cannot compare outputs if the modernized code did not run successfully.

        # Normalize outputs to ignore trailing spaces and newline style differences.
        norm_orig = _normalize_output(orig_stdout)
        norm_mod = _normalize_output(mod_stdout)

        if norm_orig == norm_mod:  # This condition checks whether the two normalized outputs are identical.
            print("PASSED: Functional Parity Confirmed")  # This print is the exact message requested when outputs match.
            return True, ""  # This return signals success so the caller knows functional parity was confirmed.

        print("FAILED: Outputs differ. Here is exactly what changed:")  # This print tells the user that parity was not confirmed and a diff follows.
        print("-" * 60)  # This print adds a visual separator so the diff is easy to spot.

        orig_lines = (norm_orig + "\n").splitlines(keepends=True)  # This line splits the normalized original output into lines, keeping newlines so the diff is accurate.
        mod_lines = (norm_mod + "\n").splitlines(keepends=True)  # This line splits the normalized modernized output into lines in the same way.

        diff_lines = unified_diff(  # This call generates a unified diff between the two outputs so you can see exactly what changed.
            orig_lines,  # This argument is the normalized original output as a list of lines.
            mod_lines,  # This argument is the normalized modernized output as a list of lines.
            fromfile="Original (test.cpp)",  # This label appears in the diff header for the first file.
            tofile="Modernized",  # This label appears in the diff header for the second file.
        )  # This closing parenthesis ends the unified_diff call.

        diff_text_parts: list[str] = []  # This list will accumulate the diff lines both for printing and for returning to callers.
        for line in diff_lines:  # This loop iterates over each line of the generated diff.
            diff_text_parts.append(line)  # This line records the diff line.
            print(line, end="")  # This print outputs the diff line; end="" avoids adding an extra newline since lines may already have one.

        diff_text = "".join(diff_text_parts)  # This line joins the diff lines into a single string.
        return False, diff_text  # This return signals failure so the caller knows the AI should fix its logic to match the original behavior and provides the diff.


if __name__ == "__main__":  # This block runs only when this file is executed as a script (e.g. python -m core.differential_tester), not when imported.
    import sys  # This import lets us read command-line arguments.

    if len(sys.argv) < 2:  # This condition checks that the user provided at least one argument (the path to the modernized code or a way to get it).
        print("Usage: python -m core.differential_tester <original.cpp> [modernized_code_string_or_path]")  # This print explains how to call the script.
        print("  If second arg is a file path, its contents are used as modernized code.")  # This line clarifies that the second arg can be a path.
        sys.exit(1)  # This line exits with a non-zero status to indicate incorrect usage.

    orig_path = sys.argv[1]  # This line reads the path to the original C++ file from the first command-line argument.

    if len(sys.argv) >= 3:  # This condition checks whether the user provided a second argument for the modernized code.
        modernized_arg = sys.argv[2]  # This line reads the second argument, which may be raw code or a file path.
        if os.path.isfile(modernized_arg):  # This check determines whether the argument is a path to an existing file.
            with open(modernized_arg, "r", encoding="utf-8") as f:  # This line opens the file for reading.
                modernized_code = f.read()  # This line reads the entire file contents as the modernized code.
        else:  # This branch handles the case where the argument is the raw modernized code string (or an invalid path).
            modernized_code = modernized_arg  # This line uses the argument directly as the modernized code.
    else:  # This branch runs when no second argument was provided; we cannot run the test without modernized code.
        print("ERROR: Provide modernized code as second argument or path to file containing it.")  # This print tells the user what is missing.
        sys.exit(1)  # This line exits because we cannot proceed without the modernized code.

    success, _diff_text = run_differential_test(orig_path, modernized_code)  # This line runs the full differential test and captures the result.
    sys.exit(0 if success else 1)  # This line exits with 0 on success (parity confirmed) or 1 on failure, for use in scripts or CI.