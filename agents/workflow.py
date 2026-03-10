"""
LangGraph Workflow for Code Modernization Engine
Implements a 3-node workflow: Analyzer -> Modernizer -> Verifier (with feedback loop)
"""

import sys
import os as os_module
sys.path.insert(0, os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__))))
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any
import json
import os
import re
import subprocess
import tempfile
import time
import requests
from langchain_ollama import ChatOllama
from core.parser import extract_functions_from_cpp_file
from core.graph import DependencyGraph, build_analysis_report
from core.differential_tester import run_differential_test


def check_ollama_health() -> bool:
    """
    Health check at the very beginning of the script: pings the local Ollama server (usually at http://localhost:11434).
    Returns True if the server responds; otherwise prints a clear message and returns False.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ OLLAMA CONNECTION VERIFIED")
            return True
        else:
            print("OLLAMA NOT RUNNING: Please start the Ollama application.")
            return False
    except requests.exceptions.RequestException:
        print("OLLAMA NOT RUNNING: Please start the Ollama application.")
        return False


class ModernizationState(TypedDict):
    """State object passed through the workflow"""
    code: str
    language: str
    analysis: str
    dependency_map: dict[str, list[str]]
    call_graph_data: dict[str, Any]
    impact_map: dict[str, list[str]]
    orphans: list[str]
    analysis_report: str
    modernized_code: str
    verification_result: dict
    error_log: str
    attempt_count: int
    is_parity_passed: bool
    is_functionally_equivalent: bool
    diff_output: str
    feedback_loop_count: int


def analyzer_node(state: ModernizationState) -> ModernizationState:
    """
    Node 1: Analyzer - Graph-aware analysis of legacy code.

    This node:
        - Uses CppParser to extract function definitions and their calls.
        - Builds a DependencyGraph of functions using networkx.
        - Identifies orphan functions (no callers) and circular recursions (cycles).
        - Produces both a machine-friendly JSON analysis and a human-readable analysis_report.
    """
    print("\n🔍 ANALYZER NODE")
    print(f"Language: {state['language']}")
    print(f"Input Code (first 200 chars):\n{state['code'][:200]}...")

    is_cpp = state["language"].lower() in {"cpp", "c++", "c++20"}
    functions_info = []
    dependency_map: dict[str, list[str]] = {}
    parser_error: str = ""
    orphans: list[str] = []
    cycles: list[list[str]] = []
    analysis_report: str = ""
    call_graph_data: dict[str, Any] = {}

    if is_cpp:
        code_value = state["code"]
        temp_file_path: str | None = None
        if os.path.isfile(code_value):
            cpp_path = code_value
        else:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".cpp",
                mode="w",
                encoding="utf-8",
            )
            temp_file_path = temp_file.name
            temp_file.write(code_value)
            temp_file.flush()
            temp_file.close()
            cpp_path = temp_file_path
        try:
            functions_info = extract_functions_from_cpp_file(cpp_path)
            dep_graph = DependencyGraph(functions_info)
            dependency_map = dep_graph.dependency_map
            graph_metrics = dep_graph.analyze()
            orphans = list(graph_metrics.get("orphans", []))
            cycles = list(graph_metrics.get("cycles", []))
            analysis_report = build_analysis_report(functions_info, dependency_map, orphans, cycles)
            call_graph_data = dep_graph.to_dict()
        except Exception as exc:
            parser_error = f"Analyzer failed to parse or analyze C++ file: {exc!r}"
        finally:
            if temp_file_path is not None and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    analysis: dict[str, Any] = {
        "language": state["language"],
        "functions": functions_info,
        "function_summary": {
            "count": len(functions_info),
            "names": [fn.get("name", "") for fn in functions_info],
        },
        "dependency_map": dependency_map,
        "call_graph_data": call_graph_data,
        "orphans": orphans,
        "cycles": cycles,
        "analysis_report": analysis_report,
        "parser_error": parser_error,
    }
    state["analysis"] = json.dumps(analysis, indent=2, sort_keys=True)
    state["dependency_map"] = dependency_map
    state["call_graph_data"] = call_graph_data
    state["impact_map"] = dependency_map
    state["orphans"] = orphans
    state["analysis_report"] = analysis_report
    print(f"Analysis (JSON):\n{state['analysis']}")
    return state


def pruner_node(state: ModernizationState) -> ModernizationState:
    """
    Node 2: Pruner - Removes orphan code (except main) before modernization.

    This node:
        - Reads the function metadata and orphan list from the deep analyzer.
        - Uses AST byte ranges to surgically remove orphan function definitions from the
          source text, except for the entry point 'main'.
        - Updates state["code"] with a pruned version to reduce the LLM's workload.
    """
    print("\n✂️  PRUNER NODE")
    is_cpp = state["language"].lower() in {"cpp", "c++", "c++20"}
    if not is_cpp:
        print("Pruner: skipping (non-C++ language).")
        return state
    if not state.get("analysis"):
        print("Pruner: no analysis found, skipping pruning.")
        return state

    try:
        analysis_obj = json.loads(state["analysis"])
    except json.JSONDecodeError:
        print("Pruner: failed to parse analysis JSON, skipping pruning.")
        return state
    functions_info = analysis_obj.get("functions") or []
    orphans = state.get("orphans") or analysis_obj.get("orphans") or []
    if not functions_info or not orphans:
        print("Pruner: no functions or no orphans detected, skipping pruning.")
        return state
    orphans_to_prune = {str(name) for name in orphans if str(name) != "main"}
    if not orphans_to_prune:
        print("Pruner: orphan list only contains 'main' or is empty, nothing to prune.")
        return state
    original_code = state["code"]
    original_bytes = original_code.encode("utf-8")
    spans_to_remove: list[tuple[int, int]] = []
    for fn in functions_info:
        name = str(fn.get("name") or "")
        if name not in orphans_to_prune:
            continue
        start_byte = fn.get("start_byte")
        end_byte = fn.get("end_byte")
        if isinstance(start_byte, int) and isinstance(end_byte, int) and 0 <= start_byte <= end_byte <= len(original_bytes):
            spans_to_remove.append((start_byte, end_byte))
    if not spans_to_remove:
        print("Pruner: no valid byte spans found for orphan functions, skipping pruning.")
        return state
    spans_to_remove.sort(key=lambda pair: pair[0])
    new_chunks: list[bytes] = []
    cursor = 0
    for start_byte, end_byte in spans_to_remove:
        if start_byte < cursor:
            continue
        if start_byte > cursor:
            new_chunks.append(original_bytes[cursor:start_byte])
        cursor = end_byte
    if cursor < len(original_bytes):
        new_chunks.append(original_bytes[cursor:])
    pruned_bytes = b"".join(new_chunks)
    pruned_code = pruned_bytes.decode("utf-8", errors="replace")
    state["code"] = pruned_code
    print(f"✂️  Pruning {len(spans_to_remove)} orphan function(s): {', '.join(sorted(orphans_to_prune))}")
    return state


def modernizer_node(state: ModernizationState) -> ModernizationState:
    """
    Node 2.5: Modernizer - Rewrites code to modern standards and handles LLM feedback.

    This node receives the current state (which may include an error_log from a
    previous verifier run) and attempts to modernize the C++ source via the
    local Ollama model. It also increments the attempt counter and stores the
    resulting modernized code back in the state.
    """
    print("\n✏️  MODERNIZER NODE")
    print(f"Attempt: {state['attempt_count']}")
    if state["error_log"]:
        print(f"Previous Feedback:\n{state['error_log']}")
    source_to_improve = state["modernized_code"] or state["code"]
    print(f"Source Code To Modernize (first 200 chars):\n{source_to_improve[:200]}...")
    is_cpp = state["language"].lower() in {"cpp", "c++", "c++20"}
    modernized = source_to_improve

    if is_cpp and ChatOllama is not None:
        prompt_parts: list[str] = []
        prompt_parts.append(
            "You are a Master C++20 Developer. "
            "Never use JavaScript keywords like 'function'. "
            "Always use proper C++ syntax (for example: 'int main()', 'auto', 'void', correct return types, and semicolons). "
            "Rewrite the given legacy C++ code into clean, idiomatic, and safe C++20 while preserving the original behavior."
        )
        analysis_obj = None
        if state["analysis"]:
            prompt_parts.append(
                "Here is a JSON analysis of the current code (functions, metadata, function dependency map, orphan functions, and cycles):\n"
                f"{state['analysis']}"
            )
            try:
                analysis_obj = json.loads(state["analysis"])
            except json.JSONDecodeError:
                analysis_obj = None
        if state.get("dependency_map"):
            prompt_parts.append(
                "Here is a concise JSON function dependency map for this translation unit:\n"
                f"{json.dumps(state['dependency_map'], indent=2, sort_keys=True)}"
            )
            dep_map = state["dependency_map"]
            callers_map: dict[str, list[str]] = {name: [] for name in dep_map.keys()}
            for caller_name, callees in dep_map.items():
                for callee_name in callees:
                    if callee_name in callers_map and caller_name not in callers_map[callee_name]:
                        callers_map[callee_name].append(caller_name)
            per_function_context_lines: list[str] = []
            signatures_by_name: dict[str, str] = {}
            comments_by_name: dict[str, str] = {}
            if isinstance(analysis_obj, dict):
                for fn_info in analysis_obj.get("functions", []):
                    fn_name = str(fn_info.get("name") or "")
                    if not fn_name:
                        continue
                    if "signature" in fn_info:
                        signatures_by_name[fn_name] = str(fn_info.get("signature") or "")
                    if "comments" in fn_info:
                        comments_by_name[fn_name] = str(fn_info.get("comments") or "")
            for fn_name in sorted(dep_map.keys()):
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
            )
        if state["error_log"]:
            if state["verification_result"].get("success") and not state.get("is_functionally_equivalent", True):
                diff_text = state.get("diff_output") or state["error_log"]
                prompt_parts.append(
                    "The code compiled, but the output changed. Fix the logic to ensure functional parity with the original program.\n"
                    "Here is a unified diff between the original and modernized program outputs:\n"
                    f"{diff_text}"
                )
            else:
                prompt_parts.append(
                    "Here are compilation errors from the last attempt. Fix these while modernizing:\n"
                    f"{state['error_log']}"
                )
        prompt_parts.append(
            "Here is the code to modernize into valid C++20:\n"
            "```cpp\n"
            f"{source_to_improve}\n"
            "```\n\n"
            "Respond with ONLY the full modernized C++20 source code, with no explanations or commentary outside the code."
        )
        full_prompt = "\n\n".join(prompt_parts)
        try:
            llm = ChatOllama(model='deepseek-coder:6.7b', temperature=0)
            response = llm.invoke(full_prompt)
            print(f'DEBUG: AI Output: {response.content}')
            raw_text = getattr(response, "content", str(response))
            cleaned = re.sub(r"```(?:[^\n]*)\n?", "", raw_text)
            modernized = cleaned.strip()
        except Exception as exc:
            error_message = f"Ollama call failed in modernizer: {exc!r}"
            print(error_message)
            if state["error_log"]:
                state["error_log"] += f"\n{error_message}"
            else:
                state["error_log"] = error_message
    else:
        modernized = source_to_improve.replace("var ", "auto ")

    state["modernized_code"] = modernized
    state["attempt_count"] += 1
    print(f"Modernized Code (first 200 chars):\n{modernized[:200]}...")

    return state




def verifier_node(state: ModernizationState) -> ModernizationState:
    """
    Node 3: Verifier - Validates modernized code via MCP compiler tool.

    Compiles the modernized code using g++ and records success or errors in the
    workflow state. Returns the updated state.
    """
    print("\n✅ VERIFIER NODE")
    print("Validating modernized code with real g++ compilation...")
    code_to_verify = state["modernized_code"]
    if not code_to_verify.strip():
        message = "CRITICAL: Ollama returned no code. Check if the model is loaded in RAM."
        print(message)
        verification_result = {
            "success": False,
            "errors": [message],
            "warnings": [],
            "compilation_time_ms": 0,
            "raw_stdout": "",
            "raw_stderr": "",
        }
        state["verification_result"] = verification_result
        state["error_log"] = message
        state["attempt_count"] = 3
        print(f"❌ Verification FAILED: {message}")
        return state
    start_time = time.time()
    with tempfile.TemporaryDirectory() as tmp_dir:
        cpp_path = os.path.join(tmp_dir, "modernized.cpp")
        exe_path = os.path.join(tmp_dir, "modernized.exe")
        with open(cpp_path, "w", encoding="utf-8") as cpp_file:
            cpp_file.write(code_to_verify)
            cpp_file.flush()
            os.fsync(cpp_file.fileno())
        if not os.path.exists(cpp_path):
            print("FILE MISSING!", cpp_path)
        gpp_exe = r"C:\msys64\mingw64\bin\g++.exe"
        compile_command_str = f'"{gpp_exe}" -std=c++20 -Wall "{cpp_path}" -o "{exe_path}"'
        try:
            completed = subprocess.run(
                compile_command_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError:
            elapsed_ms = int((time.time() - start_time) * 1000)
            verification_result = {
                "success": False,
                "errors": [
                    "g++ compiler not found. Please install a C++ compiler (such as MinGW or Visual Studio Build Tools) and ensure 'g++' is on your PATH."
                ],
                "warnings": [],
                "compilation_time_ms": elapsed_ms,
                "raw_stdout": "",
                "raw_stderr": "",
            }
            state["verification_result"] = verification_result
            state["error_log"] = "g++ compiler not found. Please install a C++ compiler (such as MinGW or Visual Studio Build Tools) and ensure 'g++' is on your PATH."
            print(f"❌ Verification FAILED: {state['error_log']}")
            return state
    elapsed_ms = int((time.time() - start_time) * 1000)
    stdout_text = (completed.stdout or "").strip()
    stderr_text = (completed.stderr or "").strip()
    success = completed.returncode == 0
    error_lines = stderr_text.splitlines() if stderr_text else []
    warning_lines = stdout_text.splitlines() if stdout_text else []

    verification_result = {
        "success": success,
        "errors": error_lines,
        "warnings": warning_lines,
        "compilation_time_ms": elapsed_ms,
        "raw_stdout": stdout_text,
        "raw_stderr": stderr_text,
    }

    state["verification_result"] = verification_result

    if verification_result["success"]:
        print("✅ Verification PASSED")
        state["error_log"] = ""
    else:
        print("❌ Verification FAILED")
        state["error_log"] = verification_result.get("raw_stderr", "")
        print(f"Errors from g++:\n{state['error_log']}")
    return state


def tester_node(state: ModernizationState) -> ModernizationState:
    """
    Node 4: Tester - Runs a differential test to check functional parity.

    This node:
      - Only runs when compilation succeeded.
      - Uses run_differential_test to compare outputs of the original and modernized code.
      - Sets is_parity_passed and, on failure, records a unified diff in error_log so
        the modernizer can perform a self-healing pass focused on logic parity.
    """
    print("\n🧪 TESTER NODE")
    state["is_parity_passed"] = True
    state["is_functionally_equivalent"] = True
    state["diff_output"] = ""
    is_cpp = state["language"].lower() in {"cpp", "c++", "c++20"}
    if not is_cpp:
        print("Tester: skipping (non-C++ language).")
        return state
    if not state["verification_result"].get("success"):
        print("Tester: skipping because compilation failed.")
        state["is_parity_passed"] = False
        state["is_functionally_equivalent"] = False
        return state
    if not state["modernized_code"].strip():
        print("Tester: skipping because modernized_code is empty.")
        state["is_parity_passed"] = False
        state["is_functionally_equivalent"] = False
        return state
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    original_cpp_path = os.path.join(base_dir, "test.cpp")
    if not os.path.isfile(original_cpp_path):
        print(f"Tester: original C++ file not found at {original_cpp_path}, skipping parity test.")
        state["is_parity_passed"] = False
        state["is_functionally_equivalent"] = False
        return state
    print("🧪 Running differential test for functional parity...")
    parity_ok, diff_text = run_differential_test(original_cpp_path, state["modernized_code"])
    state["is_parity_passed"] = bool(parity_ok)
    state["is_functionally_equivalent"] = bool(parity_ok)
    if parity_ok:
        print("✅ Parity Test PASSED (outputs match).")
        return state
    print("❌ Parity Test FAILED (outputs differ).")
    state["diff_output"] = diff_text
    state["error_log"] = diff_text
    return state


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

    print("\n🏁 Routing to END (SUCCESS: compilation + parity)")
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


if __name__ == "__main__":
    if not check_ollama_health():
        exit(1)
    _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _test_cpp_path = os.path.join(_base_dir, "test.cpp")
    try:
        with open(_test_cpp_path, "r", encoding="utf-8") as f:
            cpp_code = f.read()
        print(f"📄 Loaded test.cpp ({len(cpp_code)} characters)")
    except FileNotFoundError:
        print("❌ test.cpp not found at", _test_cpp_path)
        exit(1)
    result = run_modernization_workflow(cpp_code, language="cpp")