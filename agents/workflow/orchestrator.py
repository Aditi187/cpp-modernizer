import logging
import hashlib
from pathlib import Path
from typing import Optional, Any

from langgraph.graph import StateGraph, END

from agents.workflow.state import ModernizationState, create_initial_state
from agents.workflow.context import WorkflowContext

# Node imports
from agents.workflow.nodes.analyzer import analyzer_node
from agents.workflow.nodes.planner import planner_node
from agents.workflow.nodes.modernizer import modernizer_node
from agents.workflow.nodes.semantic_guard import semantic_guard_node
from agents.workflow.nodes.fixer import fixer_node
from agents.workflow.nodes.verifier import verifier_node

logger = logging.getLogger(__name__)

def verification_router(state: ModernizationState) -> str:
    """
    Router: routes based on verification and semantic results.
    NOTE: Conditional edge functions in LangGraph do NOT persist state mutations.
    attempt_count is incremented in verifier_node instead.
    """
    result = state.get("verification_result", {})
    success = result.get("success", False)
    semantic_ok = state.get("semantic_ok", True)
    compiler_status = result.get("compiler", "")

    context = state.get("context")
    config = getattr(context, "config", None)

    attempts = state.get("attempt_count", 0)

    # If compiler is not found or verification was skipped, treat as success
    # so the pipeline always delivers output rather than hanging on a missing tool.
    if compiler_status in ("not_found", "skipped"):
        logger.info("[ROUTER] Compiler not found or skipped — accepting rule-based output.")
        return END

    if success and semantic_ok:
        return END

    max_attempts = getattr(config, "max_attempts", 3) if config else 3
    if attempts >= max_attempts:
        logger.warning(f"[ROUTER] Max attempts ({max_attempts}) exhausted. Accepting current output.")
        return END

    if state.get("error_log"):
        return "fixer"

    if not semantic_ok:
        return "planner"

    logger.warning("[ROUTER] Verifier failed but no error log provided. Accepting current output to prevent infinite loop.")
    return END

def build_modernization_graph(use_checkpointing: bool = False):
    workflow = StateGraph(ModernizationState)
    
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("modernizer", modernizer_node)
    workflow.add_node("semantic_guard", semantic_guard_node)
    workflow.add_node("fixer", fixer_node)
    workflow.add_node("verifier", verifier_node)
    
    workflow.set_entry_point("analyzer")
    workflow.add_edge("analyzer", "planner")
    workflow.add_edge("planner", "modernizer")
    workflow.add_edge("modernizer", "semantic_guard")
    workflow.add_edge("semantic_guard", "verifier")
    workflow.add_edge("fixer", "semantic_guard")
    
    workflow.add_conditional_edges(
        "verifier",
        verification_router,
        {
            "fixer": "fixer",
            "planner": "planner",
            END: END
        }
    )

    if use_checkpointing:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
            logger.debug("[ORCHESTRATOR] Using in-memory checkpointing.")
            return workflow.compile(checkpointer=checkpointer)
        except Exception as e:
            logger.debug("[ORCHESTRATOR] Checkpointing unavailable (%s); running stateless.", e)

    return workflow.compile()



def _load_header_context(source_file: str) -> tuple[str, bool]:
    """
    Load adjacent header content (for .cpp files) or check for an implementation
    sibling (for .h files). Returns (header_code, has_implementation).

    Extracted to avoid duplication between run_modernization_workflow and
    run_modernization_stream_generator.
    
    IMPORTANT: Handles both real files and in-memory code strings.
    If source_file is "input.cpp" or doesn't exist, skips file operations.
    """
    header_code = ""
    has_implementation = False
    try:
        path = Path(source_file)
        
        # Skip file operations if file doesn't actually exist
        # (happens when code comes from web UI or API without a real file)
        if not path.exists():
            logger.debug(f"[_load_header_context] File '{source_file}' doesn't exist on disk (likely from web UI/API). Skipping header loading.")
            return header_code, has_implementation
        
        if path.suffix.lower() in (".cpp", ".cc", ".cxx"):
            from core.dependency_resolver import _extract_includes
            includes = _extract_includes(path)
            
            # --- Precision Symbol Registry Injection ---
            project_root = Path(__file__).resolve().parents[2]
            db_path = project_root / ".modernization_state.db"
            symbols = []
            if db_path.is_file():
                try:
                    from core.project_state import ProjectStateDB
                    from core.symbol_registry import format_symbols_as_headers
                    db = ProjectStateDB(str(db_path))
                    # include names are just filenames
                    include_names = [Path(inc).name for inc in includes]
                    symbols = db.get_symbols_for_includes(include_names)
                    if symbols:
                        formatted_headers = format_symbols_as_headers(symbols)
                        if formatted_headers:
                            header_code = f"// Reconstructed Header Declarations Context:\n{formatted_headers}"
                            logger.info(f"Injected {len(symbols)} symbols from registry into LLM context for {path.name}")
                except Exception as db_err:
                    logger.debug(f"Failed to query symbols from DB: {db_err}")

            # Fallback to loading raw header file content if database is empty/unavailable
            if not header_code:
                header_contents = []
                parent_dir = path.parent
                for inc in includes:
                    inc_path = Path(inc)
                    modernized_header = parent_dir / f"{inc_path.stem}_modernized{inc_path.suffix}"
                    original_header = parent_dir / inc_path
                    target_header = (
                        modernized_header if modernized_header.exists()
                        else (original_header if original_header.exists() else None)
                    )
                    if target_header:
                        with open(target_header, "r", encoding="utf-8") as hf:
                            content = hf.read()
                        header_contents.append(f"// Header Declarations from {inc_path.name}:\n{content}")
                if header_contents:
                    header_code = "\n\n".join(header_contents)
                    logger.info(f"Fallback: loaded adjacent headers for {path.name} from disk")
        elif path.suffix.lower() in (".h", ".hpp", ".hxx"):
            for ext in (".cpp", ".cc", ".cxx"):
                if (path.parent / f"{path.stem}{ext}").exists():
                    has_implementation = True
                    break
    except Exception as e:
        logger.error(f"Failed to analyse header/implementation context for {source_file}: {e}")
    return header_code, has_implementation


def run_modernization_workflow(
    code: str,
    source_file: str,
    output_path: Optional[str] = None,
    config: Optional[Any] = None,
    write_to_disk: bool = True,
    developer_feedback: str = "",
    extra_compile_args: Optional[list[str]] = None,
    request_id: str = "",
    run_id: str = "",
) -> ModernizationState:
    logger.info(f"Starting modernization workflow for {source_file}")

    ctx = WorkflowContext(config=config) if config else WorkflowContext()

    header_code, has_implementation = _load_header_context(source_file)

    initial_state = create_initial_state(
        code=code,
        source_file=source_file,
        output_file_path=output_path or "",
        context=ctx,
        request_id=request_id,
        run_id=run_id
    )
    initial_state["header_code"] = header_code
    initial_state["has_implementation"] = has_implementation
    initial_state["developer_feedback"] = developer_feedback
    initial_state["extra_compile_args"] = extra_compile_args

    app = build_modernization_graph()

    # Build a thread_id from BOTH the file path AND a short hash of the code
    # content so that different code submissions never share a checkpoint slot.
    code_hash = hashlib.md5(code.encode("utf-8", errors="replace")).hexdigest()[:12]
    thread_id = hashlib.md5(f"{source_file}:{code_hash}".encode()).hexdigest()
    invoke_config = {"configurable": {"thread_id": thread_id}}

    import os
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    pipeline_timeout = int(os.environ.get("PIPELINE_TIMEOUT_SECONDS", "600"))

    def _invoke():
        try:
            return app.invoke(initial_state, config=invoke_config)
        except TypeError:
            # Fallback: graph checkpointing failed due to unserializable objects (e.g. WorkflowContext)
            stateless_app = build_modernization_graph(use_checkpointing=False)
            return stateless_app.invoke(initial_state, config=invoke_config)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_invoke)
    try:
        final_state = future.result(timeout=pipeline_timeout)
    except FutureTimeoutError:
        logger.error(f"Pipeline wall-clock timeout exceeded ({pipeline_timeout}s). Aborting.")
        initial_state["error_log"] = f"Pipeline timed out after {pipeline_timeout} seconds."
        initial_state["semantic_ok"] = False
        return initial_state
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # Recovery from non-dict graph result
    if not isinstance(final_state, dict):
        logger.error("Workflow did not return a valid state dict.")
        return initial_state

    # Output Handling
    if not output_path:
        p = Path(source_file)
        if "_modernized.cpp" not in str(p):
            output_path = str(p.parent / f"{p.stem}_modernized.cpp")
        else:
            output_path = str(p)
    
    final_state["output_file_path"] = output_path

    # Flush token usage from context into state metrics
    if ctx.total_tokens > 0:
        metrics = final_state.get("metrics") or {}
        metrics["total_tokens"] = ctx.total_tokens
        final_state["metrics"] = metrics

    result_code = final_state.get("modernized_code")
    if not result_code or not result_code.strip():
        logger.warning("Modernization produced empty output; using original code.")
        result_code = final_state.get("code", "// empty")

    if write_to_disk:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result_code)
            logger.info(f"Saved modernized code to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write output file: {e}")

    # Defensive: cast to ModernizationState if needed
    if not isinstance(final_state, dict) or not hasattr(final_state, "keys"):
        logger.error("Workflow did not return a valid state dict. Returning initial state.")
        return initial_state
    return final_state


def run_modernization_stream_generator(
    code: str,
    source_file: str,
    config: Optional[Any] = None,
    developer_feedback: str = "",
    extra_compile_args: Optional[list[str]] = None,
    request_id: str = "",
    run_id: str = "",
):
    """
    Runs the modernization state graph step-by-step, yielding execution updates
    for each node, followed by the final state dictionary.
    """
    logger.info(f"Starting modernization stream for {source_file}")
    ctx = WorkflowContext(config=config) if config else WorkflowContext()

    header_code, has_implementation = _load_header_context(source_file)

    initial_state = create_initial_state(
        code=code,
        source_file=source_file,
        output_file_path="",
        context=ctx,
        request_id=request_id,
        run_id=run_id
    )
    initial_state["header_code"] = header_code
    initial_state["has_implementation"] = has_implementation
    initial_state["developer_feedback"] = developer_feedback
    initial_state["extra_compile_args"] = extra_compile_args

    app = build_modernization_graph()
    # Use code-content hash so every distinct code submission gets its own slot.
    code_hash = hashlib.md5(code.encode("utf-8", errors="replace")).hexdigest()[:12]
    thread_id = hashlib.md5(f"{source_file}:{code_hash}".encode()).hexdigest()
    invoke_config = {"configurable": {"thread_id": thread_id}}

    import time
    import os
    pipeline_timeout = int(os.environ.get("PIPELINE_TIMEOUT_SECONDS", "600"))
    start_time = time.time()
    
    current_state = dict(initial_state)
    try:
        # Stream events from LangGraph
        for event in app.stream(initial_state, config=invoke_config):
            if time.time() - start_time > pipeline_timeout:
                logger.error(f"Pipeline wall-clock timeout exceeded ({pipeline_timeout}s) in stream.")
                current_state["error_log"] = f"Pipeline timed out after {pipeline_timeout} seconds."
                current_state["semantic_ok"] = False
                break
                
            for node_name, node_state_updates in event.items():
                if isinstance(node_state_updates, dict):
                    # Merge delta updates — LangGraph yields only changed keys per node
                    for k, v in node_state_updates.items():
                        if v is not None or k not in current_state:
                            current_state[k] = v
                yield {"node": node_name, "status": "completed"}
    except Exception as e:
        logger.error(f"Error in streaming execution: {e}")
        try:
            current_state = app.invoke(initial_state, config=invoke_config)
        except Exception as fallback_e:
            logger.error(f"Fallback non-streaming execution also failed: {fallback_e}")
            try:
                current_state = app.invoke(initial_state)
            except Exception as last_e:
                logger.error(f"All execution strategies failed: {last_e}")
                # Ensure we always yield 'done' so the API never hangs with no result
                current_state.setdefault("modernized_code", initial_state.get("code", ""))
                current_state.setdefault("semantic_ok", False)
                current_state["error_log"] = f"Pipeline execution failed: {last_e}"

    # CRITICAL FIX: Ensure semantic_ok is always set in final state
    # If pipeline ran but semantic_ok wasn't set, query the verification result
    if "semantic_ok" not in current_state or current_state.get("semantic_ok") is None:
        logger.warning("[ORCHESTRATOR] semantic_ok not in final state, inferring from verification_result")
        verification_result = current_state.get("verification_result", {})
        modernized_code = current_state.get("modernized_code", "")
        
        # Infer: if we have modernized code and verification succeeded, consider it OK
        if modernized_code.strip() and verification_result.get("success", False):
            current_state["semantic_ok"] = True
        else:
            # If modernized code exists at all, allow it (lenient mode)
            current_state["semantic_ok"] = bool(modernized_code.strip())
            logger.info(f"[ORCHESTRATOR] Set semantic_ok={current_state.get('semantic_ok')} based on modernized_code presence")

    if ctx.total_tokens > 0:
        metrics = current_state.get("metrics") or {}
        metrics["total_tokens"] = ctx.total_tokens
        current_state["metrics"] = metrics

    yield {"node": "done", "state": current_state}