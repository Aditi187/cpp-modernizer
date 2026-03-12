import os  
import shlex  
import subprocess  
import re  
import glob
import sys
from datetime import datetime

from typing import Optional 
from fastmcp import FastMCP  

# Make the project root importable so we can reach core.parser.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.parser import CppParser
    _PARSER_AVAILABLE = True
except Exception:
    _PARSER_AVAILABLE = False



mcp_server = FastMCP("air-gapped-code-tools")  


ALLOWED_ROOT = os.path.abspath(
    os.getcwd()
)  


def _contains_parent_traversal(path: str) -> bool:
    """
    Return True if the path tries to go up a directory using '..'.

    This function checks the normalized path segments, which makes it clear
    whether the caller attempted any parent-directory traversal.
    """

    normalized = os.path.normpath(path)
    parts = normalized.split(os.sep)
    return ".." in parts


def _ensure_within_allowed_root(path: str) -> str:
    """
    Normalize a path and ensure it is inside ALLOWED_ROOT.

    Returns the absolute, normalized path if it is allowed.
    Raises a ValueError if the path attempts to escape the sandbox.
    """

    if _contains_parent_traversal(path):
        raise ValueError(
            f"Sandbox violation: path uses '..' to traverse upwards: {path}"
        )

    if os.path.isabs(path):
        absolute_path = os.path.abspath(path)
    else:
        absolute_path = os.path.abspath(os.path.join(ALLOWED_ROOT, path))

    common_root = os.path.commonpath([ALLOWED_ROOT, absolute_path])
    if common_root != ALLOWED_ROOT:
        raise ValueError(
            "Sandbox violation: path resolves outside the allowed project root.\n"
            f"Requested path: {absolute_path}\n"
            f"Allowed root:  {ALLOWED_ROOT}"
        )

    return absolute_path


def _tool_log(tool_name: str, detail: str) -> None:
    """Write a concise tool-invocation log line to stderr."""

    print(f"Tool '{tool_name}' called: {detail}", file=sys.stderr, flush=True)


@mcp_server.tool() 
def read_code(file_path: str) -> str:  
    """
    Read the contents of a text file from the local filesystem.

    The function returns either the full file contents on success,
    or a readable error message string if something goes wrong.
    """  

    _tool_log("read_code", f"file_path='{file_path}'")

    try:
        absolute_path = _ensure_within_allowed_root(file_path)
    except ValueError as sandbox_error:
        return f"ERROR: {sandbox_error}"

  
    if not os.path.isfile(
        absolute_path
    ):  
        return f"ERROR: File not found or not a regular file: {absolute_path}"  # This return sends back a clear, human-friendly error message when the file does not exist.

    try:  
        with open(
            absolute_path, "r", encoding="utf-8"
        ) as file_handle:  # This line opens the file for reading as text using UTF-8, and gives us a handle named file_handle.
            file_contents = file_handle.read()  # This line reads the entire file into a single string called file_contents.
        return file_contents  # This return line sends the successfully read text back to the AI caller.
    except UnicodeDecodeError:  # This branch runs if Python cannot interpret the file as UTF-8 text, which often means the file is binary or uses an unusual encoding.
        return (
            "ERROR: Could not decode the file as UTF-8 text. "
            "The file may be binary or use an unsupported encoding."
        )  # This return gives a descriptive message so a non-coder understands why the file could not be read as normal text.
    except Exception as error:  # This broad except catches any other unexpected problems, such as permission issues or transient I/O errors.
        return f"ERROR: Unexpected problem while reading the file: {error!r}"  # This return embeds the technical error detail inside a human-readable sentence.


@mcp_server.tool()
def write_code(file_path: str, content: str) -> str:
    """
    Write text content to a file on disk inside the sandboxed project root.

    If parent directories do not yet exist (but are still inside ALLOWED_ROOT),
    they are created automatically. The file is written using UTF-8 encoding.
    """

    _tool_log("write_code", f"file_path='{file_path}', content_len={len(content)}")

    try:
        absolute_path = _ensure_within_allowed_root(file_path)
    except ValueError as sandbox_error:
        return f"ERROR: {sandbox_error}"

    directory = os.path.dirname(absolute_path)
    if directory:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as error:
            return f"ERROR: Could not create parent directories: {error!r}"

    try:
        with open(absolute_path, "w", encoding="utf-8") as file_handle:
            file_handle.write(content)
        return f"SUCCESS: Wrote {len(content)} characters to {absolute_path}"
    except Exception as error:
        return f"ERROR: Unexpected problem while writing the file: {error!r}"


def _extract_source_locations(text: str) -> set[tuple[str, str, Optional[str]]]:
    """
    Extract file/line(/column) locations from compiler output text.

    Examples of patterns we detect:
    - "file.cpp:10:5: error: ..."
    - "path/to/file.c:42: warning: ..."
    """

    locations: set[tuple[str, str, Optional[str]]] = set()
    if not text:
        return locations

    # First pattern: file:line:column:
    pattern_with_column = re.compile(r"([^\s:]+):(\d+):(\d+)")
    # Second pattern: file:line:
    pattern_without_column = re.compile(r"([^\s:]+):(\d+)(?!:)")

    for match in pattern_with_column.finditer(text):
        file_name, line, column = match.groups()
        locations.add((file_name, line, column))

    for match in pattern_without_column.finditer(text):
        file_name, line = match.groups()
        # Only add if we do not already have a (file, line, column) entry.
        if not any(
            loc_file == file_name and loc_line == line
            for (loc_file, loc_line, _loc_col) in locations
        ):
            locations.add((file_name, line, None))

    return locations


def _append_location_hints(
    message: str, standard_output: str, error_output: str
) -> str:
    """
    Append a hint section to a compiler message if we detect source locations.

    This gives the LLM a clear suggestion to inspect specific file/line positions
    in the source code when debugging compilation failures.
    """

    locations = set()
    locations.update(_extract_source_locations(error_output))
    locations.update(_extract_source_locations(standard_output))

    if not locations:
        return message

    hint_lines = [
        "",
        "HINT: The compiler output references specific source locations.",
        "You should inspect these files at the indicated lines (and columns, when present):",
    ]

    for file_name, line, column in sorted(locations):
        if column is not None:
            hint_lines.append(f"- {file_name}, line {line}, column {column}")
        else:
            hint_lines.append(f"- {file_name}, line {line}")

    return message + "\n" + "\n".join(hint_lines)


@mcp_server.tool()  # This decorator registers the next function as another MCP tool named run_compiler that the AI can invoke.
def run_compiler(command: str, working_directory: Optional[str] = None) -> str:  # This function defines the run_compiler tool, which runs a local compile command and reports success or failure as text.
    """
    Run a local compiler command (for example 'g++' or 'javac') and return its result.

    On success, the function returns a message that includes any output from the compiler.
    On failure, the function returns a message that focuses on the error output so you can see what went wrong.
    """  
    _tool_log(
        "run_compiler",
        f"command='{command}', working_directory='{working_directory or '.'}'",
    )

    try:   
        command_parts = shlex.split(command)   
    except ValueError as error:  # This except branch triggers if the command string has mismatched quotes or other parsing issues.
        return f"ERROR: Could not parse the command string: {error!r}"  # This return gives a clear explanation that the problem is with how the command text was written.

    if working_directory is not None:
        try:
            working_directory_resolved = _ensure_within_allowed_root(working_directory)
        except ValueError as sandbox_error:
            return f"ERROR: {sandbox_error}"

        if not os.path.isdir(
            working_directory_resolved
        ):  # This condition ensures we do not try to run a command inside a folder that does not exist.
            return f"ERROR: The working directory does not exist or is not a directory: {working_directory_resolved}"  # This return informs the user that their folder path input needs to be corrected.
    else:
        working_directory_resolved = None

    try:  # This try block wraps the actual compiler execution, so we can catch operating system errors cleanly.
        completed_process = subprocess.run(  # This call actually starts the compiler program and waits until it finishes.
            command_parts,  # This argument supplies the compiler program and its arguments as a list, which is safer than passing a raw string into a shell.
            cwd=working_directory_resolved or None,  # This argument tells subprocess which folder to treat as "current"; if None, it uses the server's start directory.
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

    # Enrich the failure message with explicit guidance about which file/line
    # locations in the source code deserve attention.
    return _append_location_hints(
        combined_message, standard_output=standard_output, error_output=error_output
    )  # This return sends whichever combined failure message we built back to the AI caller, possibly annotated with helpful source-location hints.


@mcp_server.tool()
def run_binary(path: str, timeout: int = 5) -> str:
    """
    Execute a compiled binary inside the sandbox and capture its output.

    The binary must reside under ALLOWED_ROOT. A timeout (in seconds) is enforced
    to prevent infinite loops or long-running programs from hanging the server.
    """

    _tool_log("run_binary", f"path='{path}', timeout={timeout}")

    try:
        absolute_path = _ensure_within_allowed_root(path)
    except ValueError as sandbox_error:
        return f"ERROR: {sandbox_error}"

    if not os.path.isfile(absolute_path):
        return f"ERROR: Binary not found or not a regular file: {absolute_path}"

    # Use the binary's directory as the working directory so any relative paths
    # used by the program remain project-local.
    working_directory = os.path.dirname(absolute_path) or ALLOWED_ROOT

    normalized_timeout = max(1, int(timeout))
    completed_process = None
    permission_retry_attempted = False

    while True:
        try:
            completed_process = subprocess.run(
                [absolute_path],
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=normalized_timeout,
            )
            break
        except subprocess.TimeoutExpired:
            return (
                f"ERROR: Binary execution exceeded timeout of {timeout} seconds and was terminated."
            )
        except PermissionError as permission_error:
            if permission_retry_attempted:
                return (
                    "ERROR: Permission denied when trying to execute the binary, "
                    "even after applying executable permissions. "
                    f"Details: {permission_error!r}"
                )

            permission_retry_attempted = True
            _tool_log(
                "run_binary",
                f"permission denied for '{absolute_path}', attempting chmod 755 and retry",
            )
            try:
                os.chmod(absolute_path, 0o755)
            except Exception as chmod_error:
                return (
                    "ERROR: Permission denied when executing the binary and failed "
                    f"to apply chmod 755: {chmod_error!r}"
                )
        except Exception as error:
            return f"ERROR: Unexpected problem while running the binary: {error!r}"

    exit_code = completed_process.returncode
    standard_output = (completed_process.stdout or "").strip()
    error_output = (completed_process.stderr or "").strip()

    if exit_code == 0:
        if standard_output or error_output:
            return (
                f"SUCCESS (exit code 0):\n"
                f"Standard output:\n{standard_output or '[no standard output]'}\n\n"
                f"Error output:\n{error_output or '[no error output]'}"
            )
        return "SUCCESS (exit code 0): The binary finished successfully and produced no output."

    if error_output and standard_output:
        combined_message = (
            f"FAILURE (exit code {exit_code}):\n"
            f"Standard output:\n{standard_output}\n\n"
            f"Error output:\n{error_output}"
        )
    elif error_output:
        combined_message = f"FAILURE (exit code {exit_code}):\n{error_output}"
    elif standard_output:
        combined_message = f"FAILURE (exit code {exit_code}):\n{standard_output}"
    else:
        combined_message = (
            f"FAILURE (exit code {exit_code}): The binary failed but produced no output."
        )

    return combined_message


@mcp_server.tool()
def list_directory(path: str) -> str:
    """
    List the contents of a directory inside the sandboxed project root.

    Returns a human-readable listing of files and subdirectories.
    """

    _tool_log("list_directory", f"path='{path}'")

    # Allow an empty string or '.' to mean the sandbox root.
    target = path or "."

    try:
        absolute_path = _ensure_within_allowed_root(target)
    except ValueError as sandbox_error:
        return f"ERROR: {sandbox_error}"

    if not os.path.isdir(absolute_path):
        return f"ERROR: Not a directory or does not exist: {absolute_path}"

    try:
        entries = sorted(os.listdir(absolute_path))
    except Exception as error:
        return f"ERROR: Could not list directory contents: {error!r}"

    if not entries:
        return f"Directory is empty: {absolute_path}"

    lines = [f"Listing for {absolute_path}:"]
    for name in entries:
        full_path = os.path.join(absolute_path, name)
        if os.path.isdir(full_path):
            lines.append(f"[DIR]  {name}")
        else:
            lines.append(f"[FILE] {name}")

    return "\n".join(lines)


@mcp_server.tool()
def list_tree(path: str, depth: int = 2) -> str:
    """
    Render a visual directory tree from a path inside the sandbox.

    The depth parameter controls how many nested levels are expanded.
    """

    _tool_log("list_tree", f"path='{path}', depth={depth}")

    target = path or "."
    try:
        absolute_path = _ensure_within_allowed_root(target)
    except ValueError as sandbox_error:
        return f"ERROR: {sandbox_error}"

    if not os.path.isdir(absolute_path):
        return f"ERROR: Not a directory or does not exist: {absolute_path}"

    max_depth = max(0, int(depth))
    root_display = os.path.relpath(absolute_path, ALLOWED_ROOT)
    if root_display == ".":
        root_display = os.path.basename(ALLOWED_ROOT) or ALLOWED_ROOT

    lines: list[str] = [f"Tree for {root_display} (depth={max_depth}):"]

    def _walk(current_path: str, prefix: str, current_depth: int) -> None:
        if current_depth >= max_depth:
            return

        try:
            entries = sorted(os.listdir(current_path))
        except Exception as error:
            lines.append(f"{prefix}[ERROR] {error!r}")
            return

        for index, entry in enumerate(entries):
            full_path = os.path.join(current_path, entry)
            is_last = index == len(entries) - 1
            connector = "`-- " if is_last else "|-- "
            child_prefix = "    " if is_last else "|   "

            if os.path.isdir(full_path):
                lines.append(f"{prefix}{connector}{entry}/")
                _walk(full_path, prefix + child_prefix, current_depth + 1)
            else:
                lines.append(f"{prefix}{connector}{entry}")

    _walk(absolute_path, prefix="", current_depth=0)
    return "\n".join(lines)


@mcp_server.tool()
def search_code(query: str, file_pattern: str = "*") -> str:
    """
    Search text across project files using glob + regex.

    Returns matching lines in a grep-like "file:line: content" format.
    """

    _tool_log("search_code", f"query='{query}', file_pattern='{file_pattern}'")

    if not query:
        return "ERROR: Query must be a non-empty regex string."

    try:
        regex = re.compile(query)
    except re.error as regex_error:
        return f"ERROR: Invalid regex query: {regex_error}"

    pattern = file_pattern or "*"
    glob_pattern = os.path.join(ALLOWED_ROOT, "**", pattern)
    candidate_paths = glob.glob(glob_pattern, recursive=True)

    # Keep only regular files inside ALLOWED_ROOT.
    files: list[str] = []
    for matched in candidate_paths:
        if not os.path.isfile(matched):
            continue
        try:
            _ensure_within_allowed_root(matched)
        except ValueError:
            continue
        files.append(matched)

    files = sorted(set(files))
    if not files:
        return f"No files matched pattern '{pattern}'."

    matches: list[str] = []
    files_scanned = 0
    files_with_decode_errors = 0
    max_results = 500

    for file_path in files:
        files_scanned += 1
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                for line_number, line_text in enumerate(handle, start=1):
                    if regex.search(line_text):
                        rel_path = os.path.relpath(file_path, ALLOWED_ROOT)
                        matches.append(f"{rel_path}:{line_number}: {line_text.rstrip()}")
                        if len(matches) >= max_results:
                            break
        except UnicodeDecodeError:
            files_with_decode_errors += 1
        except Exception:
            continue

        if len(matches) >= max_results:
            break

    if not matches:
        return (
            f"No matches found for /{query}/ across {files_scanned} files "
            f"(pattern='{pattern}')."
        )

    header = [
        f"Found {len(matches)} matches for /{query}/ in {files_scanned} files "
        f"(pattern='{pattern}')."
    ]
    if files_with_decode_errors:
        header.append(
            f"Skipped {files_with_decode_errors} non-text file(s) due to UTF-8 decode errors."
        )
    if len(matches) >= max_results:
        header.append(f"Results truncated at {max_results} matches.")

    return "\n".join(header + [""] + matches)


@mcp_server.tool()
def get_file_info(file_path: str) -> str:
    """
    Return metadata for a file: size, modified time, and line count.
    """

    _tool_log("get_file_info", f"file_path='{file_path}'")

    try:
        absolute_path = _ensure_within_allowed_root(file_path)
    except ValueError as sandbox_error:
        return f"ERROR: {sandbox_error}"

    if not os.path.isfile(absolute_path):
        return f"ERROR: File not found or not a regular file: {absolute_path}"

    try:
        stat_info = os.stat(absolute_path)
        size_bytes = stat_info.st_size
        modified_iso = datetime.fromtimestamp(stat_info.st_mtime).isoformat()

        line_count = 0
        with open(absolute_path, "r", encoding="utf-8") as handle:
            for _ in handle:
                line_count += 1

        relative_path = os.path.relpath(absolute_path, ALLOWED_ROOT)
        return (
            f"File: {relative_path}\n"
            f"Size (bytes): {size_bytes}\n"
            f"Last modified: {modified_iso}\n"
            f"Line count: {line_count}"
        )
    except UnicodeDecodeError:
        return (
            "ERROR: Could not decode the file as UTF-8 for line counting. "
            "File may be binary or use a different encoding."
        )
    except Exception as error:
        return f"ERROR: Unexpected problem while gathering file info: {error!r}"


# ---------------------------------------------------------------------------
# Helpers for get_context_for_function
# ---------------------------------------------------------------------------

_PRIMITIVE_TYPES: set[str] = {
    "int", "float", "double", "char", "bool", "void", "long", "short",
    "unsigned", "signed", "auto", "size_t", "ptrdiff_t", "nullptr_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int8_t",  "int16_t",  "int32_t",  "int64_t",
    "wchar_t", "char8_t", "char16_t", "char32_t",
    # common macros / keywords that look like identifiers
    "const", "static", "inline", "virtual", "explicit", "constexpr",
    "override", "final", "typename", "class", "struct", "enum",
    "template", "namespace", "return", "new", "delete", "nullptr",
}


def _extract_candidate_type_names(text: str) -> list[str]:
    """Return unique identifier tokens from *text* that might be custom C++ types.

    Filters out primitive types, keywords, and ``std::`` identifiers.
    """
    tokens = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', text)
    seen: set[str] = set()
    result: list[str] = []
    for tok in tokens:
        if tok in seen or tok in _PRIMITIVE_TYPES or tok.startswith("std"):
            continue
        seen.add(tok)
        result.append(tok)
    return result


def _classify_ownership(source_code: str) -> list[str]:
    """Infer ownership / lifecycle notes from a type's source code."""
    hints: list[str] = []

    # Deleted copy constructor / copy-assignment operator → non-copyable
    if re.search(r'=\s*delete', source_code):
        if re.search(r'\b(operator\s*=|\w+\s*\(\s*const\s+\w+\s*&)', source_code):
            hints.append("non-copyable (copy constructor or assignment deleted)")
        else:
            hints.append("has explicitly deleted special member(s)")

    # Exclusive ownership
    if re.search(r'\bstd::unique_ptr\b', source_code):
        hints.append("owns exclusive resource via std::unique_ptr — not copyable by default")

    # Shared ownership
    if re.search(r'\bstd::shared_ptr\b', source_code):
        hints.append("shares ownership via std::shared_ptr")

    # Polymorphic base
    if re.search(r'\bvirtual\b', source_code):
        hints.append("polymorphic (has virtual method(s)); prefer pointer/reference semantics")

    # Explicit destructor — likely manages a resource
    if re.search(r'~\w+\s*\(', source_code):
        hints.append("manages resources (explicit destructor defined)")

    return hints if hints else ["standard value semantics (copyable and movable)"]


def _collect_type_bundle(
    seed_type_names: list[str],
    type_definitions: dict[str, str],
    types_meta: list[dict],
) -> dict[str, dict]:
    """Recursively resolve *seed_type_names* and all their base classes.

    Returns ``{type_name: {source_code, bases, ownership_hints}}``.
    Only types present in *type_definitions* are included.
    """
    name_to_meta: dict[str, dict] = {}
    for t in types_meta:
        n = str(t.get("name") or "")
        if n:
            name_to_meta[n] = t

    bundle: dict[str, dict] = {}
    queue = list(seed_type_names)
    visited: set[str] = set()

    while queue:
        type_name = queue.pop(0)
        if type_name in visited:
            continue
        visited.add(type_name)

        source_code = type_definitions.get(type_name)
        if source_code is None:
            continue  # Not a known custom type in this file.

        meta = name_to_meta.get(type_name, {})
        bases = [str(b) for b in (meta.get("bases") or [])]
        ownership_hints = _classify_ownership(source_code)

        bundle[type_name] = {
            "source_code": source_code,
            "bases": bases,
            "ownership_hints": ownership_hints,
        }

        # Queue base classes for recursive resolution.
        queue.extend(b for b in bases if b not in visited)

    return bundle


def _format_type_bundle(bundle: dict[str, dict]) -> str:
    """Render the type bundle as a structured, LLM-readable section."""
    if not bundle:
        return ""

    lines: list[str] = ["=== TYPE BUNDLE (memory layout & ownership model) ==="]
    for type_name in sorted(bundle):
        info = bundle[type_name]
        lines.append(f"\n--- Type: {type_name} ---")
        bases = info.get("bases") or []
        if bases:
            lines.append(f"Inherits from: {', '.join(bases)}")
        hints = info.get("ownership_hints") or []
        lines.append(f"Ownership/Lifecycle: {'; '.join(hints)}")
        lines.append("Definition:")
        lines.append(str(info.get("source_code", "")))

    return "\n".join(lines)


@mcp_server.tool()
def get_context_for_function(file_path: str, function_fqn: str) -> str:
    """
    Return a rich modernization context bundle for a single C++ function.

    The bundle contains four sections:

    1. **Function context** — full signature and body of the requested function.
    2. **Called function signatures** — signatures of every internally called function
       so the LLM respects existing contracts.
    3. **Type Bundle** — definitions of every custom struct/class used in the
       function's parameters or return type, resolved recursively through the
       class hierarchy (base classes are included too), each annotated with
       ownership and lifecycle notes (non-copyable, polymorphic, etc.).
    4. **Required headers** — ``#include`` directives inferred for this function.

    This context is designed to let an LLM modernize the function safely and
    produce immediately compilable C++ without misunderstanding ownership.
    """
    _tool_log(
        "get_context_for_function",
        f"file_path='{file_path}', function_fqn='{function_fqn}'",
    )

    if not _PARSER_AVAILABLE:
        return (
            "ERROR: CppParser is unavailable. "
            "Ensure 'tree-sitter-cpp' is installed in the active virtual environment."
        )

    if not function_fqn or not function_fqn.strip():
        return "ERROR: function_fqn must be a non-empty fully-qualified function name."

    try:
        absolute_path = _ensure_within_allowed_root(file_path)
    except ValueError as sandbox_error:
        return f"ERROR: {sandbox_error}"

    if not os.path.isfile(absolute_path):
        return f"ERROR: File not found or not a regular file: {absolute_path}"

    try:
        parser = CppParser()
        project_map = parser.parse_file(absolute_path)
    except Exception as exc:
        return f"ERROR: Failed to parse C++ file: {exc!r}"

    functions: dict = project_map.get("functions") or {}
    if not isinstance(functions, dict) or function_fqn not in functions:
        available = sorted(functions.keys()) if isinstance(functions, dict) else []
        return (
            f"ERROR: Function '{{function_fqn}}' not found in {os.path.basename(absolute_path)}.\n"
            f"Available FQNs: {available}"
        ).replace("{{function_fqn}}", function_fqn)

    try:
        base_context = parser.get_context_for_function(function_fqn)
    except Exception as exc:
        return f"ERROR: Context extraction failed: {exc!r}"

    fn_meta = functions[function_fqn]
    signature = str(fn_meta.get("signature") or "")
    body = str(base_context.get("body") or "")
    called_signatures: dict = base_context.get("called_function_signatures") or {}

    # Collect seed type names from structured parameters + full signature text.
    parameters: list = fn_meta.get("parameters") or []
    param_type_text = " ".join(str(p.get("type") or "") for p in parameters)
    seed_candidates = _extract_candidate_type_names(param_type_text + " " + signature)

    type_definitions: dict = project_map.get("type_definitions") or {}
    types_meta: list = project_map.get("types") or []

    # Only keep candidates that are recognized custom types in this translation unit.
    seed_types = [t for t in seed_candidates if t in type_definitions]
    type_bundle = _collect_type_bundle(seed_types, type_definitions, types_meta)

    # --- Format output sections ---
    sections: list[str] = []

    sections.append(f"=== FUNCTION CONTEXT: {function_fqn} ===")
    sections.append(f"\nSignature:\n{signature}")
    sections.append(f"\nBody:\n{body}")

    if called_signatures:
        sections.append("\n=== CALLED FUNCTION SIGNATURES ===")
        for callee_fqn, callee_sig in sorted(called_signatures.items()):
            sections.append(f"  {callee_fqn}:\n    {callee_sig}")

    type_bundle_text = _format_type_bundle(type_bundle)
    if type_bundle_text:
        sections.append("\n" + type_bundle_text)

    include_reqs: list = (project_map.get("include_requirements") or {}).get(function_fqn, [])
    if include_reqs:
        sections.append(f"\n=== REQUIRED HEADERS ===\n{', '.join(include_reqs)}")

    return "\n".join(sections)


if __name__ == "__main__":  # This line checks whether this file is being run directly as a script rather than being imported as a module.
    mcp_server.run()  # This line starts the MCP server event loop, so the AI client can connect and call the read_code and run_compiler tools.