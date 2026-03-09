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

from core.parser import extract_functions_from_cpp_file  # This import brings in our Tree-sitter-based helper to extract functions from a C++ file.
from core.graph import (  # This grouped import brings in helpers to build and analyze the function dependency graph.
    DependencyGraph,
    build_analysis_report,
)  # This closing parenthesis ends the grouped import from core.graph.
from core.differential_tester import run_differential_test  # This import brings in the differential tester used to check functional parity.


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
    analysis: str  # This field stores a JSON-formatted string describing analysis results such as discovered functions and graph metrics.
    dependency_map: dict[str, list[str]]  # This field stores a call-graph style dependency map: function name -> list of functions it calls.
    call_graph_data: dict[str, Any]  # This field stores a JSON-serializable view of the call graph (nodes and edges) for visualization and analysis.
    impact_map: dict[str, list[str]]  # This field stores which functions call which (alias of dependency_map for clarity in prompts).
    orphans: list[str]  # This field stores a list of functions that have no callers in the local call graph.
    analysis_report: str  # This field stores a concise human-readable summary of the codebase / translation unit structure.
    modernized_code: str  # This field holds the most recent modernized version of the code produced by the modernizer node.
    verification_result: dict  # This field records the outcome of compiling the modernized code, including success flag and any errors.
    error_log: str  # This field accumulates human-readable error messages that should be fed back into the model for retries.
    attempt_count: int  # This field tracks how many times the modernizer node has been run, which controls the retry loop.
    is_parity_passed: bool  # This field records whether the most recent differential test confirmed functional parity.
    is_functionally_equivalent: bool  # This field records whether the latest tester run declared the modernized code functionally equivalent.
    diff_output: str  # This field stores the unified diff from the tester when outputs differ.
    feedback_loop_count: int  # This field counts how many times we have routed back into the modernizer for self-healing.


def analyzer_node(state: ModernizationState) -> ModernizationState:
    """
    Node 1: Analyzer - Graph-aware analysis of legacy code.

    This node:
      - Uses CppParser to extract function definitions and their calls.
      - Builds a DependencyGraph of functions using networkx.
      - Identifies orphan functions (no callers) and circular recursions (cycles).
      - Produces both a machine-friendly JSON analysis and a human-readable analysis_report.
    """
    print("\n🔍 ANALYZER NODE")  # This print helps you see in the console when the analyzer node is running.
    print(f"Language: {state['language']}")  # This print shows which language label the workflow thinks it is processing.
    print(f"Input Code (first 200 chars):\n{state['code'][:200]}...")  # This print gives you a quick peek at the beginning of the input code for context.

    is_cpp = state["language"].lower() in {"cpp", "c++", "c++20"}  # This line checks whether the language label indicates C++, which is when we want to use the Tree-sitter parser.

    functions_info = []  # type: ignore[var-annotated]  # This variable will hold the list of functions discovered by the parser, if any.
    dependency_map: dict[str, list[str]] = {}  # This dictionary will map each function name to the list of functions it calls.
    parser_error: str = ""  # This string will store any error message encountered while trying to parse the code.
    orphans: list[str] = []  # This list will store functions with no callers.
    cycles: list[list[str]] = []  # This list will store any detected circular recursion cycles.
    analysis_report: str = ""  # This string will hold a concise human-readable summary of the code structure.
    call_graph_data: dict[str, Any] = {}  # This dictionary will store a JSON-serializable view of the call graph.

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

        try:  # This try block attempts to run the parser on the chosen C++ file path and build a dependency graph.
            functions_info = extract_functions_from_cpp_file(cpp_path)  # type: ignore[arg-type]  # This call extracts function definitions, calls, and byte spans.

            dep_graph = DependencyGraph(functions_info)  # This line constructs a DependencyGraph object from the function metadata.
            dependency_map = dep_graph.dependency_map  # This line reads the caller -> callees adjacency list.

            graph_metrics = dep_graph.analyze()  # This line analyzes orphans and cycles in the call graph.
            orphans = list(graph_metrics.get("orphans", []))  # This line records the set of orphan functions.
            cycles = list(graph_metrics.get("cycles", []))  # This line records the list of cycles.

            analysis_report = build_analysis_report(functions_info, dependency_map, orphans, cycles)  # This line builds a concise human-readable report.
            call_graph_data = dep_graph.to_dict()  # This line exports the call graph as simple nodes/edges data for visualization and prompts.
        except Exception as exc:  # This except block catches any parsing- or graph-related errors so they do not crash the workflow.
            parser_error = f"Analyzer failed to parse or analyze C++ file: {exc!r}"  # This line records a human-readable message explaining why analysis failed.
        finally:  # This block always runs, whether analysis succeeded or failed.
            if temp_file_path is not None and os.path.exists(temp_file_path):  # This check ensures we only delete the temporary file if we actually created one and it still exists.
                os.remove(temp_file_path)  # This line deletes the temporary file to avoid leaving unnecessary files on disk.

    analysis: dict[str, Any] = {  # This dictionary will hold the structured analysis information that we will turn into JSON.
        "language": state["language"],  # This entry records the language label that was used for this analysis.
        "functions": functions_info,  # This entry stores detailed per-function metadata including names, calls, and byte spans.
        "function_summary": {  # This nested dictionary describes the functions we discovered (if any).
            "count": len(functions_info),  # This entry records how many functions the parser found in the C++ file.
            "names": [fn.get("name", "") for fn in functions_info],  # This list comprehension collects just the function names from the parser output.
        },  # This closing brace ends the function_summary dictionary.
        "dependency_map": dependency_map,  # This entry captures the call-graph style function dependency map for the current translation unit.
        "call_graph_data": call_graph_data,  # This entry stores a simple nodes/edges representation of the call graph.
        "orphans": orphans,  # This entry records functions that have no callers.
        "cycles": cycles,  # This entry records any circular recursion cycles in the call graph.
        "analysis_report": analysis_report,  # This entry stores a concise human-readable description of the code structure.
        "parser_error": parser_error,  # This entry stores any error message from the parser, or an empty string if everything went fine.
    }  # This closing brace ends the analysis dictionary.

    state["analysis"] = json.dumps(analysis, indent=2, sort_keys=True)  # This line converts the analysis dictionary into a stable, human-readable JSON string and stores it in the state.
    state["dependency_map"] = dependency_map  # This line stores the raw dependency map separately so downstream nodes can consume it without re-parsing JSON.
    state["call_graph_data"] = call_graph_data  # This line stores the simple call graph data for downstream nodes and visualization.
    state["impact_map"] = dependency_map  # This line aliases the dependency_map as impact_map to emphasize its use in prompts.
    state["orphans"] = orphans  # This line stores the list of orphan functions so downstream nodes can choose to prune them.
    state["analysis_report"] = analysis_report  # This line stores the human-readable report so it can be included in prompts if desired.
    print(f"Analysis (JSON):\n{state['analysis']}")  # This print displays the analysis JSON in the console so you can see exactly what the analyzer found.

    return state  # This return passes the updated state object along to the next node in the workflow.


def pruner_node(state: ModernizationState) -> ModernizationState:
    """
    Node 2: Pruner - Removes orphan code (except main) before modernization.

    This node:
      - Reads the function metadata and orphan list from the deep analyzer.
      - Uses AST byte ranges to surgically remove orphan function definitions from the
        source text, except for the entry point 'main'.
      - Updates state["code"] with a pruned version to reduce the LLM's workload.
    """
    print("\n✂️  PRUNER NODE")  # This print marks the start of the pruning step in the console.

    is_cpp = state["language"].lower() in {"cpp", "c++", "c++20"}  # This line checks whether we are working with C++ code.
    if not is_cpp:  # This condition skips pruning for non-C++ languages.
        print("Pruner: skipping (non-C++ language).")
        return state  # This return leaves the state unchanged.

    if not state.get("analysis"):  # This condition ensures that we only run if the deep analyzer produced analysis.
        print("Pruner: no analysis found, skipping pruning.")
        return state  # This return leaves the state unchanged.

    try:
        analysis_obj = json.loads(state["analysis"])  # This line parses the JSON analysis string into a Python dictionary.
    except json.JSONDecodeError:
        print("Pruner: failed to parse analysis JSON, skipping pruning.")  # This print warns that analysis could not be decoded.
        return state  # This return leaves the state unchanged.

    functions_info = analysis_obj.get("functions") or []  # This line reads the detailed function metadata list.
    orphans = state.get("orphans") or analysis_obj.get("orphans") or []  # This line prefers the orphans recorded on the state but falls back to the analysis JSON.
    if not functions_info or not orphans:  # This condition checks whether there is anything to prune.
        print("Pruner: no functions or no orphans detected, skipping pruning.")  # This print explains why no pruning will happen.
        return state  # This return leaves the state unchanged.

    # We never prune the program entry point 'main', even if it has no callers.
    orphans_to_prune = {str(name) for name in orphans if str(name) != "main"}  # This set records orphan function names that are eligible for pruning.
    if not orphans_to_prune:
        print("Pruner: orphan list only contains 'main' or is empty, nothing to prune.")  # This print explains that we intentionally keep main.
        return state  # This return leaves the state unchanged.

    original_code = state["code"]  # This line reads the original source code text from the state.
    original_bytes = original_code.encode("utf-8")  # This line encodes the code as UTF-8 bytes so we can use Tree-sitter byte offsets.

    # Collect byte-span ranges for every function definition that should be removed.
    spans_to_remove: list[tuple[int, int]] = []  # This list will hold (start_byte, end_byte) pairs for orphan functions.
    for fn in functions_info:  # This loop inspects each function metadata entry.
        name = str(fn.get("name") or "")  # This line normalizes the function name.
        if name not in orphans_to_prune:  # This condition ensures we only consider orphan functions for pruning.
            continue
        start_byte = fn.get("start_byte")
        end_byte = fn.get("end_byte")
        if isinstance(start_byte, int) and isinstance(end_byte, int) and 0 <= start_byte <= end_byte <= len(original_bytes):  # This condition validates the byte span.
            spans_to_remove.append((start_byte, end_byte))  # This line records the valid span.

    if not spans_to_remove:  # This condition checks whether we have any valid spans to delete.
        print("Pruner: no valid byte spans found for orphan functions, skipping pruning.")  # This print explains why pruning will not proceed.
        return state  # This return leaves the state unchanged.

    spans_to_remove.sort(key=lambda pair: pair[0])  # This line sorts the spans by starting offset so we can rebuild the code in order.

    # Rebuild the code bytes, skipping over the regions marked for removal.
    new_chunks: list[bytes] = []  # This list will hold the surviving segments of the original code.
    cursor = 0  # This integer tracks our current position in the original byte array.
    for start_byte, end_byte in spans_to_remove:  # This loop processes each span to remove.
        if start_byte < cursor:  # This condition skips spans that overlap with or are fully contained in a previously removed region.
            continue
        if start_byte > cursor:  # This condition preserves the code between the previous cursor and the start of this span.
            new_chunks.append(original_bytes[cursor:start_byte])
        cursor = end_byte  # This line moves the cursor past the removed span.

    if cursor < len(original_bytes):  # This condition ensures we keep any trailing code after the last removed span.
        new_chunks.append(original_bytes[cursor:])

    pruned_bytes = b"".join(new_chunks)  # This line concatenates all preserved segments into a single byte string.
    pruned_code = pruned_bytes.decode("utf-8", errors="replace")  # This line decodes the pruned bytes back into text.

    state["code"] = pruned_code  # This line updates the state with the pruned source code so the modernizer works on a smaller, focused input.
    print(f"✂️  Pruning {len(spans_to_remove)} orphan function(s): {', '.join(sorted(orphans_to_prune))}")  # This print summarizes what was removed in a visually distinct way.

    return state  # This return passes the updated state object along to the next node in the workflow.


def modernizer_node(state: ModernizationState) -> ModernizationState:
    """
    Node 2: Modernizer - Rewrites code to modern standards
    Receives error_log if coming from verifier feedback
    """
    print("\n✏️  MODERNIZER NODE")  # This print marks the start of a modernization attempt in the console.
    print(f"Attempt: {state['attempt_count']}")  # This print shows how many modernization attempts have already been made.

    if state["error_log"]:  # This condition checks whether the verifier or tester reported any previous issues.
        print(f"Previous Feedback:\n{state['error_log']}")  # This print shows that feedback so you can see what the model is being asked to fix.

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
        analysis_obj = None
        if state["analysis"]:  # This condition checks whether the deep analyzer produced any structured information about functions.
            prompt_parts.append(  # This call adds a section containing the analyzer's JSON output so the model can see function structure and dependencies.
                "Here is a JSON analysis of the current code (functions, metadata, function dependency map, orphan functions, and cycles):\n"
                f"{state['analysis']}"
            )  # This comment clarifies that we are sharing a rich, graph-aware analysis with the model.
            try:
                analysis_obj = json.loads(state["analysis"])  # This line parses the analysis JSON so we can use function signatures and comments.
            except json.JSONDecodeError:
                analysis_obj = None  # This line safely falls back if the JSON cannot be parsed.
        if state.get("dependency_map"):  # This condition checks whether we have a non-empty dependency map to share explicitly.
            prompt_parts.append(  # This call adds a compact, focused view of the dependency map so the model can quickly see the local call graph.
                "Here is a concise JSON function dependency map for this translation unit:\n"
                f"{json.dumps(state['dependency_map'], indent=2, sort_keys=True)}"
            )  # This comment clarifies that we provide a focused call graph optimized for the model's consumption.
            # Derive caller information from the dependency_map so we can describe per-function context.
            dep_map = state["dependency_map"]
            callers_map: dict[str, list[str]] = {name: [] for name in dep_map.keys()}  # This dictionary will map each function to the list of functions that call it.
            for caller_name, callees in dep_map.items():  # This loop inverts the adjacency list to build the callers map.
                for callee_name in callees:
                    if callee_name in callers_map and caller_name not in callers_map[callee_name]:
                        callers_map[callee_name].append(caller_name)

            per_function_context_lines: list[str] = []  # This list will hold human-readable context lines for each function.
            signatures_by_name: dict[str, str] = {}
            comments_by_name: dict[str, str] = {}
            if isinstance(analysis_obj, dict):
                for fn_info in analysis_obj.get("functions", []):  # This loop builds lookup tables for signatures and comments by function name.
                    fn_name = str(fn_info.get("name") or "")
                    if not fn_name:
                        continue
                    if "signature" in fn_info:
                        signatures_by_name[fn_name] = str(fn_info.get("signature") or "")
                    if "comments" in fn_info:
                        comments_by_name[fn_name] = str(fn_info.get("comments") or "")

            for fn_name in sorted(dep_map.keys()):  # This loop builds a context sentence for each function in the translation unit.
                callers = sorted(callers_map.get(fn_name, []))
                callees = dep_map.get(fn_name, [])
                callers_str = ", ".join(callers) if callers else "none"
                callees_str = ", ".join(callees) if callees else "none"
                signature_text = signatures_by_name.get(fn_name, "")
                comments_text = comments_by_name.get(fn_name, "")
                signature_fragment = f" Signature: {signature_text}" if signature_text else ""
                comments_fragment = f" Documentation: {comments_text}" if comments_text else ""
                per_function_context_lines.append(
                    f"- You are modernizing function '{fn_name}'. It is called by [{callers_str}] and it calls [{callees_str}].{signature_fragment}{comments_fragment} Ensure the interface remains compatible with its callers and callees."
                )

            prompt_parts.append(
                "Function-level call graph context (for each function, who calls it and what it calls):\n"
                + "\n".join(per_function_context_lines)
            )  # This comment explains that we explicitly describe per-function context to the model.
        if state["error_log"]:  # This condition checks whether previous attempts produced any feedback that needs to be fixed.
            # Distinguish between compiler failures and parity (logic) failures so we can give targeted instructions.
            if state["verification_result"].get("success") and not state.get("is_functionally_equivalent", True):
                diff_text = state.get("diff_output") or state["error_log"]
                prompt_parts.append(  # This call adds a section explicitly describing the parity failure and unified diff output.
                    "The code compiled, but the output changed. Fix the logic to ensure functional parity with the original program.\n"
                    "Here is a unified diff between the original and modernized program outputs:\n"
                    f"{diff_text}"
                )
            else:
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


def tester_node(state: ModernizationState) -> ModernizationState:
    """
    Node 4: Tester - Runs a differential test to check functional parity.

    This node:
      - Only runs when compilation succeeded.
      - Uses run_differential_test to compare outputs of the original and modernized code.
      - Sets is_parity_passed and, on failure, records a unified diff in error_log so
        the modernizer can perform a self-healing pass focused on logic parity.
    """
    print("\n🧪 TESTER NODE")  # This print marks the start of the tester step in the console.

    # Reset parity / equivalence flags and previous diff output before running a new test.
    state["is_parity_passed"] = True  # This line pessimistically assumes parity will pass; we flip it to False on failure.
    state["is_functionally_equivalent"] = True  # This line assumes functional equivalence until the tester proves otherwise.
    state["diff_output"] = ""  # This line clears any previous diff output.

    is_cpp = state["language"].lower() in {"cpp", "c++", "c++20"}  # This line checks whether we are working with C++ code.
    if not is_cpp:
        print("Tester: skipping (non-C++ language).")
        return state  # This return leaves the state unchanged for non-C++ code.

    if not state["verification_result"].get("success"):  # This condition ensures we only run the tester when compilation succeeded.
        print("Tester: skipping because compilation failed.")  # This print clarifies why we do not run the differential test.
        state["is_parity_passed"] = False  # This line records that parity has not been confirmed.
        state["is_functionally_equivalent"] = False
        return state  # This return leaves error_log with compiler errors.

    if not state["modernized_code"].strip():  # This condition checks that we have modernized code to test.
        print("Tester: skipping because modernized_code is empty.")  # This print explains why the tester is being skipped.
        state["is_parity_passed"] = False
        state["is_functionally_equivalent"] = False
        return state

    # The differential tester expects a path to the original C++ file. In this workflow,
    # we assume the original is test.cpp at the project root (same as the __main__ block).
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This line gets the project root directory where test.cpp lives.
    original_cpp_path = os.path.join(base_dir, "test.cpp")  # This line builds the full path to test.cpp in the project root.

    if not os.path.isfile(original_cpp_path):
        print(f"Tester: original C++ file not found at {original_cpp_path}, skipping parity test.")  # This print warns that we cannot run the differential test.
        state["is_parity_passed"] = False
        state["is_functionally_equivalent"] = False
        return state

    print("🧪 Running differential test for functional parity...")  # This print signals that we are about to run the parity test.
    parity_ok, diff_text = run_differential_test(original_cpp_path, state["modernized_code"])  # This line runs the differential test and captures both status and diff.

    state["is_parity_passed"] = bool(parity_ok)  # This line records whether the latest parity test passed.
    state["is_functionally_equivalent"] = bool(parity_ok)  # This line mirrors the parity result into the functionally-equivalent flag.

    if parity_ok:
        print("✅ Parity Test PASSED (outputs match).")  # This print confirms that the modernized code matches the original program output.
        return state

    # On parity failure, store the unified diff so the modernizer can fix logic.
    print("❌ Parity Test FAILED (outputs differ).")  # This print reports that the outputs do not match.
    state["diff_output"] = diff_text  # This line stores the unified diff for the modernizer prompt.
    state["error_log"] = diff_text  # This line mirrors the diff into error_log for backward-compatible prompts and logging.

    return state  # This return passes the updated state (including parity info) back into the LangGraph router.


def should_retry(state: ModernizationState) -> str:
    """
    Router function: Decides whether to retry modernization or proceed to END
    """
    verification_success = bool(state["verification_result"].get("success"))  # This line normalizes the compilation success flag.
    parity_passed = bool(state.get("is_functionally_equivalent", False))  # This line reads whether the last differential test declared functional equivalence.
    feedback_loops = int(state.get("feedback_loop_count", 0))  # This line reads how many times we have already looped back into the modernizer.

    # Hard stop if we have already tried multiple self-healing passes.
    if state["attempt_count"] >= 3 or feedback_loops >= 3:
        print("\n🏁 Routing to END (MAX RETRIES REACHED)")  # This print signals that the self-healing loop has exhausted its budget.
        return "end"

    if not verification_success:
        # Compiler failure: route back to the modernizer with raw_stderr in error_log.
        state["feedback_loop_count"] = feedback_loops + 1  # This line increments the feedback loop counter for a compiler failure retry.
        print("\n🔄 Routing back to MODERNIZER (compiler failure, retrying)")  # This print explains that we are retrying due to compilation errors.
        return "modernizer"

    if not parity_passed:
        # Logic / parity failure: route back to the modernizer with unified_diff feedback.
        state["feedback_loop_count"] = feedback_loops + 1  # This line increments the feedback loop counter for a parity failure retry.
        print("\n🔄 Routing back to MODERNIZER (parity failure, self-healing)")  # This print explains that we are retrying due to a parity mismatch.
        return "modernizer"

    print("\n🏁 Routing to END (SUCCESS: compilation + parity)")  # This print signals that both compilation and parity passed.
    return "end"


def build_workflow():
    """
    Builds and returns the LangGraph workflow
    """
    workflow = StateGraph(ModernizationState)
    
    # Add nodes
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("pruner", pruner_node)
    workflow.add_node("modernizer", modernizer_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("tester", tester_node)
    
    # Define edges for the graph-first, self-healing pipeline
    workflow.add_edge("analyzer", "pruner")
    workflow.add_edge("pruner", "modernizer")
    workflow.add_edge("modernizer", "verifier")
    workflow.add_edge("verifier", "tester")
    
    # Conditional edge from tester based on compilation and parity results
    workflow.add_conditional_edges(
        "tester",
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
        dependency_map={},
        call_graph_data={},
        impact_map={},
        orphans=[],
        analysis_report="",
        modernized_code="",
        verification_result={},
        error_log="",
        attempt_count=0,
        is_parity_passed=False,
        is_functionally_equivalent=False,
        diff_output="",
        feedback_loop_count=0,
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