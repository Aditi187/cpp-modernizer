import os
import shlex
import subprocess

from typing import Optional

from fastmcp import FastMCP


mcp_server = FastMCP("air-gapped-code-tools")


@mcp_server.tool()
def read_code(file_path: str) -> str:
    """Read the contents of a text file from the local filesystem.

    Returns the contents or an error message.
    """

    if not os.path.isfile(file_path):
        return f"ERROR: File not found or not a regular file: {file_path}"

    try:
        with open(file_path, "r", encoding="utf-8") as file_handle:
            return file_handle.read()
    except UnicodeDecodeError:
        return (
            "ERROR: Could not decode the file as UTF-8 text. "
            "The file may be binary or use an unsupported encoding."
        )
    except Exception as error:
        return f"ERROR: Unexpected problem while reading the file: {error!r}"


@mcp_server.tool()
def run_compiler(command: str, working_directory: Optional[str] = None) -> str:
    """Run a compiler command and return its output or an error message."""

    try:
        command_parts = shlex.split(command)
    except ValueError as error:
        return f"ERROR: Could not parse the command string: {error!r}"

    if working_directory is not None and not os.path.isdir(working_directory):
        return f"ERROR: The working directory does not exist or is not a directory: {working_directory}"

    try:
        completed_process = subprocess.run(
            command_parts,
            cwd=working_directory or None,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return (
            "ERROR: The compiler program could not be found. "
            "Please confirm that the command name (such as 'g++' or 'javac') is installed and on your PATH."
        )
    except Exception as error:
        return f"ERROR: Unexpected problem while running the compiler: {error!r}"

    exit_code = completed_process.returncode
    standard_output = (completed_process.stdout or "").strip()
    error_output = (completed_process.stderr or "").strip()

    if exit_code == 0:
        if standard_output:
            return f"SUCCESS (exit code 0):\n{standard_output}"
        else:
            return "SUCCESS (exit code 0): The compiler finished successfully and produced no output."

    if error_output and standard_output:
        combined_message = (
            f"FAILURE (exit code {exit_code}):\n"
            f"Standard output:\n{standard_output}\n\n"
            f"Error output:\n{error_output}"
        )
    elif error_output:
        combined_message = (
            f"FAILURE (exit code {exit_code}):\n{error_output}"
        )
    elif standard_output:
        combined_message = (
            f"FAILURE (exit code {exit_code}):\n{standard_output}"
        )
    else:
        combined_message = (
            f"FAILURE (exit code {exit_code}): The compiler failed but produced no output."
        )

    return combined_message  # This return sends whichever combined failure message we built back to the AI caller.


if __name__ == "__main__":
    mcp_server.run()