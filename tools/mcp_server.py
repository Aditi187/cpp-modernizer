import os  # This import lets us work with file paths on your computer, for example to check whether a path is a file.
import shlex  # This import provides a safe way to split a command string (like 'g++ main.cpp') into the list format Python's subprocess expects.
import subprocess  # This import allows Python to start other programs on your computer, such as compilers like g++ or javac.

from typing import Optional  # This import lets us describe that a value might be missing (None), which helps document the behavior of our functions.

from fastmcp import FastMCP  # This import brings in the FastMCP class, which builds a Model Context Protocol server around our Python functions.


# Create a single MCP server instance that will host all of our tools.
mcp_server = FastMCP("air-gapped-code-tools")  # This line creates the MCP server with a short name so the AI client can recognize and connect to it.


@mcp_server.tool()  # This decorator tells FastMCP that the function below should be exposed as an MCP tool the AI can call.
def read_code(file_path: str) -> str:  # This function definition declares the read_code tool, which takes a file path as text and returns the file contents as text.
    """
    Read the contents of a text file from the local filesystem.

    The function returns either the full file contents on success,
    or a readable error message string if something goes wrong.
    """  # This triple-quoted string explains in plain English what the read_code tool does and how its result should be interpreted.

    # Explain our input in simple language.
    # - file_path: a string giving either an absolute path like "C:\\my_folder\\code.cpp"
    #              or a relative path like "src/main.cpp" from where the server is running.

    # First, we check whether the path actually points to a real file on disk.
    if not os.path.isfile(file_path):  # This condition tests if file_path refers to a normal file; if it does not, we treat that as an error.
        return f"ERROR: File not found or not a regular file: {file_path}"  # This return sends back a clear, human-friendly error message when the file does not exist.

    try:  # This keyword starts a block where we will attempt to read the file, catching any problems that might occur.
        with open(file_path, "r", encoding="utf-8") as file_handle:  # This line opens the file for reading as text using UTF-8, and gives us a handle named file_handle.
            file_contents = file_handle.read()  # This line reads the entire file into a single string called file_contents.
        return file_contents  # This return line sends the successfully read text back to the AI caller.
    except UnicodeDecodeError:  # This branch runs if Python cannot interpret the file as UTF-8 text, which often means the file is binary or uses an unusual encoding.
        return (
            "ERROR: Could not decode the file as UTF-8 text. "
            "The file may be binary or use an unsupported encoding."
        )  # This return gives a descriptive message so a non-coder understands why the file could not be read as normal text.
    except Exception as error:  # This broad except catches any other unexpected problems, such as permission issues or transient I/O errors.
        return f"ERROR: Unexpected problem while reading the file: {error!r}"  # This return embeds the technical error detail inside a human-readable sentence.


@mcp_server.tool()  # This decorator registers the next function as another MCP tool named run_compiler that the AI can invoke.
def run_compiler(command: str, working_directory: Optional[str] = None) -> str:  # This function defines the run_compiler tool, which runs a local compile command and reports success or failure as text.
    """
    Run a local compiler command (for example 'g++' or 'javac') and return its result.

    On success, the function returns a message that includes any output from the compiler.
    On failure, the function returns a message that focuses on the error output so you can see what went wrong.
    """  # This triple-quoted string documents in plain English how to use run_compiler and what kind of information it returns.

    # Explain our inputs in simple language.
    # - command: a single string exactly as you would type it in a terminal,
    #            for example "g++ -Wall main.cpp" or "javac MyProgram.java".
    # - working_directory: an optional folder path where the command should be run;
    #                      if you leave it empty, the server runs the command in its own start folder.

    # Convert the single command string into a list of parts understood by subprocess without using the shell.
    try:  # This try block wraps the parsing step so we can handle malformed command strings gracefully.
        command_parts = shlex.split(command)  # This line safely breaks the command string into pieces, for example "g++ -Wall main.cpp" becomes ["g++", "-Wall", "main.cpp"].
    except ValueError as error:  # This except branch triggers if the command string has mismatched quotes or other parsing issues.
        return f"ERROR: Could not parse the command string: {error!r}"  # This return gives a clear explanation that the problem is with how the command text was written.

    # If the caller gave us a working directory, we double-check that it exists and is actually a directory.
    if working_directory is not None and not os.path.isdir(working_directory):  # This condition ensures we do not try to run a command inside a folder that does not exist.
        return f"ERROR: The working directory does not exist or is not a directory: {working_directory}"  # This return informs the user that their folder path input needs to be corrected.

    try:  # This try block wraps the actual compiler execution, so we can catch operating system errors cleanly.
        completed_process = subprocess.run(  # This call actually starts the compiler program and waits until it finishes.
            command_parts,  # This argument supplies the compiler program and its arguments as a list, which is safer than passing a raw string into a shell.
            cwd=working_directory or None,  # This argument tells subprocess which folder to treat as "current"; if None, it uses the server's start directory.
            capture_output=True,  # This flag collects both standard output and error output so we can return them as text instead of printing them to a console.
            text=True,  # This flag tells subprocess to decode the output as text strings instead of raw bytes, making the result easier to read.
        )  # This closing parenthesis ends the subprocess.run call that executes the compiler.
    except FileNotFoundError:  # This except branch runs when the system cannot even find the program named in the command, for example if 'g++' is not installed.
        return (
            "ERROR: The compiler program could not be found. "
            "Please confirm that the command name (such as 'g++' or 'javac') is installed and on your PATH."
        )  # This return explains that the error is about the compiler not existing or not being visible to the system.
    except Exception as error:  # This broad except catches any other unexpected operating system level failures.
        return f"ERROR: Unexpected problem while running the compiler: {error!r}"  # This return gives a readable message along with the technical exception for debugging.

    # At this point, the compiler ran and returned an exit code that tells us whether it succeeded or failed.
    exit_code = completed_process.returncode  # This line saves the numerical exit code, where 0 normally means success and any non-zero value means failure.

    # Clean up the compiler's outputs by stripping leading and trailing blank lines.
    standard_output = (completed_process.stdout or "").strip()  # This line normalizes the normal output text, turning None into an empty string and removing extra whitespace.
    error_output = (completed_process.stderr or "").strip()  # This line normalizes the error output text in the same way, so we can present it neatly.

    # If the exit code is zero, we treat the run as successful and return the output in a success message.
    if exit_code == 0:  # This condition checks whether the compiler reported success using the conventional exit code of zero.
        if standard_output:  # This nested check looks to see if the compiler printed anything useful to standard output.
            return f"SUCCESS (exit code 0):\n{standard_output}"  # This return reports both that the command worked and shows the compiler's normal output.
        else:  # This branch covers the case where the compiler succeeded but printed nothing at all.
            return "SUCCESS (exit code 0): The compiler finished successfully and produced no output."  # This return reassures you that success with no printed output is normal for many compilers.

    # If we reach this point, the exit code is non-zero, which normally means something went wrong.
    if error_output and standard_output:  # This condition handles the case where the compiler produced both normal output and error output.
        combined_message = (
            f"FAILURE (exit code {exit_code}):\n"
            f"Standard output:\n{standard_output}\n\n"
            f"Error output:\n{error_output}"
        )  # This variable builds a detailed message that keeps normal and error text separate so you can see all the details.
    elif error_output:  # This branch handles the common case where the compiler only produced error output.
        combined_message = (
            f"FAILURE (exit code {exit_code}):\n{error_output}"
        )  # This variable focuses on the error text, because that usually explains what went wrong with the compilation.
    elif standard_output:  # This branch covers the rare case where the compiler failed but only produced normal output.
        combined_message = (
            f"FAILURE (exit code {exit_code}):\n{standard_output}"
        )  # This variable passes along the normal output so you at least see whatever messages the compiler generated.
    else:  # This final branch covers the edge case where the compiler failed but printed absolutely nothing.
        combined_message = (
            f"FAILURE (exit code {exit_code}): The compiler failed but produced no output."
        )  # This variable still reports the failure and exit code even though there is no text from the compiler itself.

    return combined_message  # This return sends whichever combined failure message we built back to the AI caller.


if __name__ == "__main__":  # This line checks whether this file is being run directly as a script rather than being imported as a module.
    mcp_server.run()  # This line starts the MCP server event loop, so the AI client can connect and call the read_code and run_compiler tools.