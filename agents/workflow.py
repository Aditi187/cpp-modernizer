"""
LangGraph Workflow for Code Modernization Engine  # This top-level string briefly explains that this file defines the multi-step modernization workflow.
Implements a 3-node workflow: Analyzer -> Modernizer -> Verifier (with feedback loop)  # This line describes the three main nodes and the feedback loop structure.
"""  # This closing triple quote ends the module-level documentation string.

import sys  # This import allows us to manipulate the module search path.
import os as os_module  # This import is used to work with file paths.
sys.path.insert(0, os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__))))  # This line adds the parent directory to sys.path so we can import core package.

from langgraph.graph import StateGraph, END  # This import pulls in the LangGraph classes we use to define nodes and mark the workflow end state.
from typing import TypedDict, Any  # This import lets us define a strongly-typed dictionary for our shared workflow state.
import json  # This import is used to store analysis results as nicely formatted JSON strings inside the state.
import os  # This import lets us work with file paths, for example to check whether input code is actually a file path.
import re  # This import provides regular expression tools so we can clean AI output by removing markdown fences.
import subprocess  # This import allows us to run external programs like the local C++ compiler (g++).
import tempfile  # This import helps us create temporary files and folders where we can write code for parsing and compilation.
import time  # This import lets us measure how long a compilation step takes so we can record it in the verification result.
import requests  # This import allows us to make HTTP requests to check if the local Ollama server is reachable.

from langchain_ollama import ChatOllama  # This import provides the chat interface to models served by Ollama (e.g. deepseek-coder).

from core.parser import extract_functions_from_cpp_file  # This import brings in our Tree-sitter parser so the analyzer node can find function blocks in C++ files.


def check_ollama_health() -> bool:
    """
    Health check at the very beginning of the script: pings the local Ollama server (usually at http://localhost:11434).
    Returns True if the server responds; otherwise prints a clear message and returns False.
    """  # This docstring explains the purpose of the health check in plain English.
    try:  # This try block attempts to reach the Ollama API so we know the application is running.
        response = requests.get("http://localhost:11434/api/tags", timeout=5)  # This line sends a GET request to the standard Ollama port; timeout avoids hanging.
        if response.status_code == 200:  # This condition checks that Ollama returned a normal success response.
            print("✅ OLLAMA CONNECTION VERIFIED")  # This print confirms that the health check passed so the user knows the script can proceed.
            return True  # This return tells the caller that Ollama is running and the modernization loop can run.
        else:  # This branch handles any non-200 response from the server.
            print("OLLAMA NOT RUNNING: Please start the Ollama application.")  # This print gives the user a clear, actionable message.
            return False  # This return tells the caller not to run the workflow until Ollama is started.
    except requests.exceptions.RequestException:  # This except block runs when we cannot connect at all (e.g. connection refused or timeout).
        print("OLLAMA NOT RUNNING: Please start the Ollama application.")  # This line prints exactly the message requested so the user knows what to do.
        return False  # This return ensures the script does not continue without a working Ollama server.


class ModernizationState(TypedDict):
    """State object passed through the workflow"""
    code: str  # This field holds either the raw source code text or a path to a source file, depending on how the workflow is invoked.
    language: str  # This field records the language name (for example "cpp"), so nodes can adjust behavior based on the language.
    analysis: str  # This field stores a JSON-formatted string describing analysis results such as discovered functions.
    modernized_code: str  # This field holds the most recent modernized version of the code produced by the modernizer node.
    verification_result: dict  # This field records the outcome of compiling the modernized code, including success flag and any errors.
    error_log: str  # This field accumulates human-readable error messages that should be fed back into the model for retries.
    attempt_count: int  # This field tracks how many times the modernizer node has been run, which controls the retry loop.


def analyzer_node(state: ModernizationState) -> ModernizationState:
    """
    Node 1: Analyzer - Finds and catalogs old code patterns
    """
    print("\n🔍 ANALYZER NODE")  # This print helps you see in the console when the analyzer node is running.
    print(f"Language: {state['language']}")  # This print shows which language label the workflow thinks it is processing.
    print(f"Input Code (first 200 chars):\n{state['code'][:200]}...")  # This print gives you a quick peek at the beginning of the input code for context.

    is_cpp = state["language"].lower() in {"cpp", "c++", "c++20"}  # This line checks whether the language label indicates C++, which is when we want to use the Tree-sitter parser.

    functions_info = []  # type: ignore[var-annotated]  # This variable will hold the list of functions discovered by the parser, if any.
    parser_error: str = ""  # This string will store any error message encountered while trying to parse the code.

    if is_cpp:  # This condition ensures we only run the C++ parser when we are actually working with C++ code.
        code_value = state["code"]  # This line saves the code field to a shorter variable name, which may be either text or a file path.
        temp_file_path: str | None = None  # This variable will keep track of a temporary file path if we need to create one.

        if os.path.isfile(code_value):  # This check handles the case where the caller passed in a real file path instead of raw code text.
            cpp_path = code_value  # This line simply reuses the existing file path when it already points to an on-disk C++ file.
        else:  # This branch handles the case where the caller passed in raw C++ source code as text.
            temp_file = tempfile.NamedTemporaryFile(  # This call creates a new temporary file that will hold the C++ source code long enough for parsing.
                delete=False,  # This argument keeps the file on disk after closing so the parser can open it.
                suffix=".cpp",  # This argument gives the temporary file a .cpp extension so tools can recognize it as C++.
                mode="w",  # This argument opens the file for writing text rather than reading.
                encoding="utf-8",  # This argument ensures the text we write is encoded in UTF-8, matching our parser expectations.
            )  # This closing parenthesis ends the NamedTemporaryFile call.
            temp_file_path = temp_file.name  # This line records the path of the temporary file so we can pass it into the parser and delete it later.
            temp_file.write(code_value)  # This line writes the raw C++ code text into the temporary file so the parser can read it from disk.
            temp_file.flush()  # This line forces the operating system to write any buffered data to disk immediately.
            temp_file.close()  # This line closes the temporary file handle so there are no open descriptors left when the parser runs.
            cpp_path = temp_file_path  # This line sets the parser input path to the newly created temporary C++ file.

        try:  # This try block attempts to run the parser on the chosen C++ file path.
            functions_info = extract_functions_from_cpp_file(cpp_path)  # type: ignore[arg-type]  # This call uses our Tree-sitter-based helper to find function names and bodies.
        except Exception as exc:  # This except block catches any parsing-related errors so they do not crash the workflow.
            parser_error = f"Analyzer failed to parse C++ file: {exc!r}"  # This line records a human-readable message explaining why parsing failed.
        finally:  # This block always runs, whether parsing succeeded or failed.
            if temp_file_path is not None and os.path.exists(temp_file_path):  # This check ensures we only delete the temporary file if we actually created one and it still exists.
                os.remove(temp_file_path)  # This line deletes the temporary file to avoid leaving unnecessary files on disk.

    analysis: dict[str, Any] = {  # This dictionary will hold the structured analysis information that we will turn into JSON.
        "language": state["language"],  # This entry records the language label that was used for this analysis.
        "function_summary": {  # This nested dictionary describes the functions we discovered (if any).
            "count": len(functions_info),  # This entry records how many functions the parser found in the C++ file.
            "names": [fn.get("name", "") for fn in functions_info],  # This list comprehension collects just the function names from the parser output.
        },  # This closing brace ends the function_summary dictionary.
        "parser_error": parser_error,  # This entry stores any error message from the parser, or an empty string if everything went fine.
    }  # This closing brace ends the analysis dictionary.

    state["analysis"] = json.dumps(analysis, indent=2)  # This line converts the analysis dictionary into a human-readable JSON string and stores it in the state.
    print(f"Analysis (JSON):\n{state['analysis']}")  # This print displays the analysis JSON in the console so you can see exactly what the analyzer found.

    return state  # This return passes the updated state object along to the next node in the workflow.


def modernizer_node(state: ModernizationState) -> ModernizationState:
    """
    Node 2: Modernizer - Rewrites code to modern standards
    Receives error_log if coming from verifier feedback
    """
    print("\n✏️  MODERNIZER NODE")  # This print marks the start of a modernization attempt in the console.
    print(f"Attempt: {state['attempt_count']}")  # This print shows how many modernization attempts have already been made.

    if state["error_log"]:  # This condition checks whether the verifier reported any previous compilation errors.
        print(f"Previous Errors:\n{state['error_log']}")  # This print shows those errors so you can see what the model is being asked to fix.

    # Decide which version of the code we want to improve: the last modernized version (if any) or the original legacy code.
    source_to_improve = state["modernized_code"] or state["code"]  # This line prefers the most recent modernized code but falls back to the original code on the first attempt.

    print(f"Source Code To Modernize (first 200 chars):\n{source_to_improve[:200]}...")  # This print shows the first part of the code that will be sent to the model.

    is_cpp = state["language"].lower() in {"cpp", "c++", "c++20"}  # This line checks whether we are working with C++ so we know whether to ask for C++20 modernization.

    modernized = source_to_improve  # This line initializes the modernized code to the source as a safe default in case the model call fails.

    if is_cpp and ChatOllama is not None:  # This condition ensures we only call the local Ollama model when we have both C++ code and the library installed.
        # Build a detailed natural-language prompt that explains the task and enforces strict C++20 rules for the deepseek-coder model.
        prompt_parts: list[str] = []  # This list will collect different sections of the prompt before joining them into a single string.
        prompt_parts.append(  # This call adds the high-level instruction describing the assistant's role and strict syntax requirements.
            "You are a Master C++20 Developer. "
            "Never use JavaScript keywords like 'function'. "
            "Always use proper C++ syntax (for example: 'int main()', 'auto', 'void', correct return types, and semicolons). "
            "Rewrite the given legacy C++ code into clean, idiomatic, and safe C++20 while preserving the original behavior."
        )  # This comment explains that we clearly tell the model to avoid JavaScript-style code and stick to real C++20.
        if state["analysis"]:  # This condition checks whether the analyzer produced any structured information about functions.
            prompt_parts.append(  # This call adds a section containing the analyzer's JSON output so the model can see function structure.
                "Here is a JSON analysis of the current code (functions and metadata):\n"
                f"{state['analysis']}"
            )  # This comment clarifies that we are sharing function-level analysis with the model.
        if state["error_log"]:  # This condition checks whether previous compilation attempts produced errors that need to be fixed.
            prompt_parts.append(  # This call adds a section describing the last compilation errors, which the model should address.
                "Here are compilation errors from the last attempt. Fix these while modernizing:\n"
                f"{state['error_log']}"
            )  # This comment clarifies that we expect the model to use these errors as feedback for corrections.
        prompt_parts.append(  # This call adds the actual source code and the final instructions on how to respond.
            "Here is the code to modernize into valid C++20:\n"
            "```cpp\n"
            f"{source_to_improve}\n"
            "```\n\n"
            "Respond with ONLY the full modernized C++20 source code, with no explanations or commentary outside the code."
        )  # This comment explains that we instruct the model to return only code so parsing later is easier.

        full_prompt = "\n\n".join(prompt_parts)  # This line joins all prompt sections together with blank lines so the text is readable and well structured.

        try:  # This try block attempts to call the local deepseek-coder model through the ChatOllama interface.
            llm = ChatOllama(model='deepseek-coder:6.7b', temperature=0)  # This line creates a chat client configured to talk to the specific deepseek-coder:6.7b model deterministically.
            response = llm.invoke(full_prompt)  # This line sends the combined prompt to the model and waits for a single response message.
            print(f'DEBUG: AI Output: {response.content}')  # This print lets you see the raw AI output in the console so you can debug whether Ollama is actually responding.
            raw_text = getattr(response, "content", str(response))  # This line extracts the response text, falling back to a string form if there is no .content attribute.

            # The model is asked to return only code, but we still defensively strip out any markdown triple backticks so only raw code remains.
            cleaned = re.sub(r"```(?:[^\n]*)\n?", "", raw_text)  # This line removes any lines starting with ``` (optionally followed by a language tag like cpp) from the response text.
            modernized = cleaned.strip()  # This line trims leading and trailing whitespace so we keep just the raw code body.
        except Exception as exc:  # This except block catches any errors from contacting the local Ollama server.
            error_message = f"Ollama call failed in modernizer: {exc!r}"  # This line builds a human-readable description of what went wrong.
            print(error_message)  # This print displays the failure message in the console for debugging.
            if state["error_log"]:  # This condition checks whether there is already some error log text recorded.
                state["error_log"] += f"\n{error_message}"  # This line appends the new error description to the existing log so nothing is lost.
            else:  # This branch handles the case where the error log was previously empty.
                state["error_log"] = error_message  # This line initializes the error log with the new error description.
    else:  # This branch runs if we are not in a C++ context or the Ollama integration is not available.
        # For non-C++ languages or missing Ollama, we keep a very simple placeholder modernization behavior.
        modernized = source_to_improve.replace("var ", "auto ")  # This line performs a minimal token replacement, just to show that some transformation happened.

    state["modernized_code"] = modernized  # This line stores the modernized (or minimally transformed) code back into the shared workflow state.
    state["attempt_count"] += 1  # This line increments the attempt counter so the router can track how many modernization passes we have performed.

    print(f"Modernized Code (first 200 chars):\n{modernized[:200]}...")  # This print shows the beginning of the modernized code so you can see what changed.

    return state  # This return passes the updated state on to the verifier node.


def verifier_node(state: ModernizationState) -> ModernizationState:
    """
    Node 3: Verifier - Validates modernized code via MCP compiler tool
    """
    print("\n✅ VERIFIER NODE")  # This print marks the start of the verification step in the console.
    print("Validating modernized code with real g++ compilation...")  # This print explains that we are about to run the actual compiler on the code.

    code_to_verify = state["modernized_code"]  # This line uses only the modernized_code field, because we want to compile exactly what the modernizer produced.

    if not code_to_verify.strip():  # This condition checks if the AI returned no code (empty or whitespace-only); if so, we stop the workflow entirely.
        message = "CRITICAL: Ollama returned no code. Check if the model is loaded in RAM."  # This line sets the exact critical error message requested for empty AI output.
        print(message)  # This print makes the critical condition visible in the console so you know why the workflow is stopping.
        verification_result = {  # This dictionary records a failure result so the workflow can finish without calling the compiler.
            "success": False,  # This field marks the verification as a failure because there was nothing to compile.
            "errors": [message],  # This list stores the same message for the verification result.
            "warnings": [],  # This empty list indicates that there were no compiler warnings.
            "compilation_time_ms": 0,  # This field records zero time because we never invoked the compiler.
            "raw_stdout": "",  # This field is empty because the compiler never ran.
            "raw_stderr": "",  # This field is empty for the same reason.
        }  # This closing brace ends the verification_result dictionary for the empty-code case.
        state["verification_result"] = verification_result  # This line stores the failure result in the shared state.
        state["error_log"] = message  # This line copies the exact error message into the error log for visibility in later inspection.
        state["attempt_count"] = 3  # This line sets the attempt count to the max so the router will go to END and stop the loop instead of retrying.
        print(f"❌ Verification FAILED: {message}")  # This print reinforces the failure message in the console.
        return state  # This return exits early so we do not try to call g++ with empty input.

    start_time = time.time()  # This line records the current time so we can later compute how long the compilation took.

    with tempfile.TemporaryDirectory() as tmp_dir:  # This context manager creates a temporary directory that will be automatically cleaned up afterward.
        cpp_path = os.path.join(tmp_dir, "modernized.cpp")  # This line builds the path to a temporary C++ source file inside that directory.
        exe_path = os.path.join(tmp_dir, "modernized.exe")  # This line builds the path to the compiled executable that g++ will produce on Windows.

        with open(cpp_path, "w", encoding="utf-8") as cpp_file:  # This line opens the temporary C++ file for writing using UTF-8 encoding.
            cpp_file.write(code_to_verify)  # This line writes the modernized code text into the file so that g++ can compile it.
            cpp_file.flush()  # This line flushes Python's buffering so all text is handed off to the operating system.
            os.fsync(cpp_file.fileno())  # This line forces the operating system to write the data to disk immediately, which is especially important on Windows.

        if not os.path.exists(cpp_path):  # This condition double-checks that the temporary source file really exists on disk before compiling.
            print("FILE MISSING!", cpp_path)  # This print alerts you if, for some reason, the file is not present when we expect it to be.

        gpp_exe = r"C:\msys64\mingw64\bin\g++.exe"  # This line sets the absolute path to your g++ compiler so we always use this specific executable.
        compile_command_str = f'"{gpp_exe}" -std=c++20 -Wall "{cpp_path}" -o "{exe_path}"'  # This line builds a single command string for the shell; quotes protect paths with spaces.

        try:  # This try block attempts to run the compiler and capture both its output and its success or failure status.
            completed = subprocess.run(  # This call actually launches g++ as a child process via the shell and waits for it to finish.
                compile_command_str,  # This argument is the full command string that the shell will execute.
                shell=True,  # This flag tells subprocess to run the command through the system shell so the command string is interpreted correctly.
                stdout=subprocess.PIPE,  # This argument captures the compiler's standard output so we can inspect it later.
                stderr=subprocess.STDOUT,  # This argument routes standard error into standard output so we do not miss any hidden error messages.
                text=True,  # This flag asks subprocess to decode the captured output as text strings instead of raw bytes.
            )  # This closing parenthesis ends the subprocess.run call.
        except FileNotFoundError:  # This except block catches the case where the shell could not find or run the compiler at the given path.
            elapsed_ms = int((time.time() - start_time) * 1000)  # This line computes how many milliseconds passed before we discovered the missing compiler.
            verification_result = {  # This dictionary records a failure result due to a missing compiler.
                "success": False,  # This field marks verification as failed because we could not even start g++.
                "errors": [  # This list contains a human-readable explanation of the problem and how to fix it.
                    "g++ compiler not found. Please install a C++ compiler (such as MinGW or Visual Studio Build Tools) and ensure 'g++' is on your PATH."
                ],  # This closing bracket ends the list of error messages for the missing-compiler case.
                "warnings": [],  # This empty list indicates that there were no compiler warnings before the failure.
                "compilation_time_ms": elapsed_ms,  # This field records how much time elapsed while attempting to start the compiler.
                "raw_stdout": "",  # This field is empty because no compiler output was produced.
                "raw_stderr": "",  # This field is also empty for the same reason.
            }  # This closing brace ends the verification_result dictionary for the missing-compiler case.
            state["verification_result"] = verification_result  # This line stores the failure result in the shared state.
            state["error_log"] = "g++ compiler not found. Please install a C++ compiler (such as MinGW or Visual Studio Build Tools) and ensure 'g++' is on your PATH."  # This line copies the exact error message into the error log so the modernizer can see it.
            print(f"❌ Verification FAILED: {state['error_log']}")  # This print informs you in the console that the compiler itself is missing.
            return state  # This return exits because there is no way to verify code without a working compiler.

    elapsed_ms = int((time.time() - start_time) * 1000)  # This line computes how long the compilation process took in milliseconds.
    stdout_text = (completed.stdout or "").strip()  # This line normalizes the compiler's standard output into a trimmed string.
    stderr_text = (completed.stderr or "").strip()  # This line normalizes the compiler's standard error output into a trimmed string.

    success = completed.returncode == 0  # This line interprets an exit code of zero as success and any non-zero value as a compilation failure.

    error_lines = stderr_text.splitlines() if stderr_text else []  # This line breaks the error output into individual lines for easier logging and display.
    warning_lines = stdout_text.splitlines() if stdout_text else []  # This line breaks the standard output into lines, which often contain warnings or notes.

    verification_result = {  # This dictionary captures the full outcome of the compilation step.
        "success": success,  # This field marks whether the compilation succeeded or failed based on the exit code.
        "errors": error_lines,  # This list records each line of compiler error output.
        "warnings": warning_lines,  # This list records each line of compiler standard output, which may include warnings or informational messages.
        "compilation_time_ms": elapsed_ms,  # This field records how many milliseconds the compiler took to run.
        "raw_stdout": stdout_text,  # This field keeps the full un-split standard output text for deeper inspection if needed.
        "raw_stderr": stderr_text,  # This field keeps the full un-split error output text for deeper inspection if needed.
    }  # This closing brace ends the verification_result dictionary for the normal compilation case.

    state["verification_result"] = verification_result  # This line stores the verification result in the shared workflow state so the router can inspect it.

    if verification_result["success"]:  # This condition checks whether compilation finished successfully without errors.
        print("✅ Verification PASSED")  # This print confirms in the console that the modernized code compiled successfully.
        state["error_log"] = ""  # This line clears the error log because there are no compiler errors to feed back into the model.
    else:  # This branch handles the case where g++ returned at least one error.
        print("❌ Verification FAILED")  # This print reports in the console that compilation failed.
        state["error_log"] = verification_result.get("raw_stderr", "")  # This line records the exact stderr text from g++ into the error log for the modernizer to use next time.
        print(f"Errors from g++:\n{state['error_log']}")  # This print shows the compiler errors so you can see exactly what went wrong.

    return state  # This return passes the updated state (including verification results) back into the LangGraph router.


def should_retry(state: ModernizationState) -> str:
    """
    Router function: Decides whether to retry modernization or proceed to END
    """
    if state["verification_result"].get("success"):
        print("\n→ Routing to END (SUCCESS)")
        return "end"
    elif state["attempt_count"] >= 3:
        print("\n→ Routing to END (MAX RETRIES REACHED)")
        return "end"
    else:
        print("\n→ Routing back to MODERNIZER (FAILED, RETRYING)")
        return "modernizer"


def build_workflow():
    """
    Builds and returns the LangGraph workflow
    """
    workflow = StateGraph(ModernizationState)
    
    # Add nodes
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("modernizer", modernizer_node)
    workflow.add_node("verifier", verifier_node)
    
    # Define edges
    workflow.add_edge("analyzer", "modernizer")
    workflow.add_edge("modernizer", "verifier")
    
    # Conditional edge from verifier based on result
    workflow.add_conditional_edges(
        "verifier",
        should_retry,
        {
            "modernizer": "modernizer",
            "end": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("analyzer")
    
    return workflow.compile()


def run_modernization_workflow(code: str, language: str = "python"):
    """
    Executes the modernization workflow with provided code
    """
    print("=" * 60)
    print("🚀 STARTING CODE MODERNIZATION WORKFLOW")
    print("=" * 60)
    
    # Initialize state
    initial_state = ModernizationState(
        code=code,
        language=language,
        analysis="",
        modernized_code="",
        verification_result={},
        error_log="",
        attempt_count=0
    )
    
    # Build and run the workflow
    graph = build_workflow()
    final_state = graph.invoke(initial_state)
    
    # Print final results
    print("\n" + "=" * 60)
    print("📊 MODERNIZATION COMPLETE")
    print("=" * 60)
    print(f"Language: {final_state['language']}")
    print(f"Total Attempts: {final_state['attempt_count']}")
    print(f"Verification Success: {final_state['verification_result'].get('success')}")
    print(f"\nFinal Modernized Code:\n{final_state['modernized_code']}")
    
    return final_state


if __name__ == "__main__":  # This block runs only when this file is executed as a script (e.g. python -m agents.workflow), not when imported.
    # Health check at the very beginning: ping the local Ollama server; if we cannot connect, exit with a clear message.
    if not check_ollama_health():  # This line runs the health check first; if it returns False, we do not start the workflow.
        exit(1)  # This line stops the script so the user can start Ollama and try again.

    # Resolve the path to test.cpp relative to the project root (parent of the agents folder) so it works from any working directory.
    _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This line gets the project root directory where test.cpp lives.
    _test_cpp_path = os.path.join(_base_dir, "test.cpp")  # This line builds the full path to test.cpp in the project root.

    try:  # This try block attempts to read the contents of test.cpp so we can run the full modernization loop on it.
        with open(_test_cpp_path, "r", encoding="utf-8") as f:  # This line opens test.cpp for reading as UTF-8 text.
            cpp_code = f.read()  # This line reads the entire file contents into a string for the workflow.
        print(f"📄 Loaded test.cpp ({len(cpp_code)} characters)")  # This print confirms the file was loaded and shows its size.
    except FileNotFoundError:  # This except block runs if test.cpp does not exist at the expected path.
        print("❌ test.cpp not found at", _test_cpp_path)  # This print tells the user where the script looked for the file.
        exit(1)  # This line stops the script so the user can add test.cpp or fix the path.

    # Once the connection is verified, run the full modernization loop on the C++ code from test.cpp.
    result = run_modernization_workflow(cpp_code, language="cpp")  # This line invokes the full workflow (analyzer -> modernizer -> verifier with retries) on the loaded code.
