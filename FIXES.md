# Bug Report & Fix Log — Air-Gapped C++ Modernization Engine

This document describes every bug found across the codebase and the exact fix applied to each.

---

## BUG 1 (CRITICAL — Root Cause of "pipeline failed, no result returned")
**File:** `core/differential_tester.py`  
**Function:** `compile_cpp_source()`  
**Problem:** The function's docstring was placed **after** an early-return block (`if os.environ.get("SKIP_VERIFICATION")...`). In Python, a string literal after a `return` is dead code — it is evaluated and discarded. This made the function look correct but the docstring was unreachable, indicating the early-return was inserted without restructuring. More critically, this revealed the structural disorder that caused related bugs below.  
**Fix:** Moved the docstring to the top of the function, before the early-return guard.

---

## BUG 2 (CRITICAL — Infinite Retry Loop → "pipeline failed")
**File:** `agents/workflow/nodes/verifier.py`  
**Function:** `verifier_node()`  
**Problem:** When no C++ compiler is found, the verifier sets `success=False` and `error_log="No C++ compiler found."`. The `verification_router` in `orchestrator.py` then reads the non-empty `error_log` and routes to `"fixer"`. The fixer calls `compile_cpp_source()` again — which also fails (no compiler). This loop repeats until `max_attempts` is exhausted, then the pipeline ends with `modernized_code` present but `semantic_ok=False` and `success=False`.  
**Fix (verifier):** When compiler is not found, set `error_log = ""` so the router cannot route to fixer. Also set `compiler = "not_found"` in the verification result (already done).  
**Fix (router):** Added an explicit check: if `compiler in ("not_found", "skipped")`, route immediately to `END` instead of fixer.

---

## BUG 3 (CRITICAL — "Pipeline returned no result" in the UI)
**File:** `agents/workflow/orchestrator.py`  
**Function:** `run_modernization_stream_generator()`  
**Problem:** If `app.stream()` raises an exception AND both fallback `app.invoke()` calls also raise, the generator exits the try/except without ever yielding `{"node": "done", ...}`. The API's `stream_generator()` then sees `final_state = None` and yields `{"error": "Pipeline did not produce a final state."}`. The frontend receives this error SSE event but then falls to `if (!lastResult)` and throws a second generic error message, completely hiding the real cause.  
**Fix:** Added a final `except` catch that sets sensible defaults on `current_state` so the `yield {"node": "done", ...}` at the bottom always executes.

---

## BUG 4 (HIGH — Retry Loop Wastes All Attempts, Produces No Improvement)
**File:** `agents/workflow/nodes/modernizer.py`  
**Function:** `modernizer_node()`  
**Problem:** On every call (including retries), the modernizer reads `source = state.get("code", "")` — the **original legacy code**. When `semantic_guard` detects issues and injects `developer_feedback` into the state, the router sends the pipeline back through `planner → modernizer`. But the modernizer discards the previous modernized output and starts from scratch with the raw legacy code again. The developer feedback is present in the prompt but the LLM is re-doing all the work it already did, often producing the same result.  
**Fix:** On retry passes (`attempt_count > 0`), the modernizer now uses `state["modernized_code"]` (the best output so far) as its base, so each retry only needs to fix the specific remaining issues.

---

## BUG 5 (HIGH — `llm_success` Variable Used Before Assignment)
**File:** `agents/workflow/nodes/modernizer.py`  
**Function:** `modernizer_node()`  
**Problem:** `llm_success` is declared inside the `if _should_chunk(...)` block but is referenced later in the `pipeline_metadata` dict. If `_should_chunk()` is `False` and the full-file LLM path raises an exception before `llm_success = True` runs, the code falls to the `except` block and `modernized_llm = normalized_source`. Then `llm_success_after > llm_success_before` runs, but if `context.llm_calls_succeeded` wasn't incremented either, this line can produce incorrect results. More critically, if chunked mode returned empty and the full-file path was never entered, `llm_success` is `False` but `chunked_used` is `True` — causing attribution to be incorrectly shown as `"deterministic_rules_only"`.  
**Fix:** `llm_success = False` is initialized before all branches. The final attribution check now ORs with `chunked_used` so chunked successes are correctly attributed.

---

## BUG 6 (HIGH — Default Compile Standard Mismatch)
**File:** `core/differential_tester.py`  
**Function:** `_build_compile_command()`  
**Problem:** The default C++ standard when `cpp_standard=None` was `"-std=c++23"`. The project targets C++17 (as documented in README, `.env`, and all prompts). While `cpp_standard` is typically passed through from config, any code path that calls `_build_compile_command` without specifying a standard would compile with C++23 semantics, potentially masking C++17 compatibility issues in the output.  
**Fix:** Changed default from `"-std=c++23"` to `"-std=c++17"`.

---

## BUG 7 (HIGH — LangGraph Streaming State Loses Earlier Node Values)
**File:** `agents/workflow/orchestrator.py`  
**Function:** `run_modernization_stream_generator()`  
**Problem:** LangGraph's `app.stream()` yields events as `{node_name: {changed_keys: new_values}}` — only the **delta** of what that node modified. The existing code did `current_state.update(node_state_updates)` which is a shallow dict update. This is correct for keys that exist in the delta. However, if a node returns `{key: None}` to clear a value, `update()` would overwrite a valid value from an earlier node with `None`. The fix adds a guard: only update a key with `None` if it didn't previously exist.  
**Fix:** Replaced `current_state.update(...)` with a loop that skips `None` values for existing keys.

---

## BUG 8 (MEDIUM — Verifier UI Stage Stuck as "Active")
**File:** `web/app.js`  
**Function:** `runModernize()` SSE loop  
**Problem:** The pipeline UI has 5 stages: analyzer, planner, modernizer, semantic_guard, verifier. When a node completes, the SSE stream sends `{node: "node_name"}` and `advancePipeline()` marks that node complete and activates the next one. However, the **verifier** stage completes when the orchestrator sends `{node: "done", response: ...}` — NOT a verifier node event. So `advancePipeline("verifier")` is never called, leaving the verifier stage in "active" (spinning) state until `finishPipeline()` abruptly marks everything done/failed.  
**Fix:** When `d.node === 'done'` is received, explicitly call `setPipeline('verifier', 'completed')` before storing `lastResult`.

---

## BUG 9 (MEDIUM — Success=False Even When Modernization Succeeded)
**File:** `api.py`  
**Function:** Response building in both `/modernize` and `/modernize/stream`  
**Problem:** The API response `success` field was set directly to `state.get("semantic_ok", False)`. When the compiler is not found (`compiler="not_found"`), `semantic_ok` may be `True` (set by the semantic guard) but `verification_result.success` is `False`. Or, `semantic_ok` could be `False` just because the verifier never ran (no compiler). In both cases, the UI showed a red "Failed" status with perfectly good modernized code in the output panel.  
**Fix:** Success is now computed as: `semantic_ok OR (compiler_unavailable AND modernized_code is non-empty)`. If no compiler is available, the UI correctly shows success with a note that compilation was not verified.

---

## BUG 10 (MEDIUM — Fixer Always Uses `compile_only=True`)
**File:** `agents/workflow/nodes/fixer.py`  
**Function:** `attempt_compiler_error_autofix()`  
**Problem:** The fixer's internal compile check always used `compile_only=True`. For files that have a `main()` function, the verifier uses link mode (`compile_only=False`), so linker errors would be missed by the fixer's verification step. The fixer would think the fix succeeded and update `state["modernized_code"]`, then the verifier would fail again with the same linker error.  
**Fix:** The fixer now detects `main()` in the candidate code and uses `compile_only = not has_main`, matching the verifier's behavior.

---

## BUG 11 (MEDIUM — SQLite Timestamp Parsing Crashes Cache)
**File:** `agents/workflow/context.py`  
**Function:** `get_cached_llm_response()`  
**Problem:** The SQLite `CURRENT_TIMESTAMP` default can return timestamps in multiple formats depending on the SQLite version and build (e.g., `"2024-01-15 10:30:45"` vs `"2024-01-15 10:30:45.123456"`). The code used `datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")` which crashes with `ValueError` on fractional seconds, causing an unhandled exception that propagates up and breaks the LLM caching system entirely.  
**Fix:** Added multi-format parsing with a fallback that treats unparseable timestamps as expired (safe fail).

---

## BUG 12 (MEDIUM — Stream Error Hides Real Error Message)
**File:** `api.py` + `web/app.js`  
**Problem:** When the pipeline stream generator raises an exception, the API yields `{"error": "Pipeline stream error: <message>"}` and then `return`s. The frontend's SSE loop reads the `d.error` field and throws it. However, the thrown error propagated to the `catch (err)` block which calls `toast(Error: ${err.message})`. The issue is that the SSE parsing was in a `try/catch` that only called `console.warn` — so the error was swallowed, never reaching the outer `catch`, and the user saw the generic "Pipeline returned no result" message instead.  
**Fix:** The SSE parse block now re-throws real pipeline errors. The API also yields a `done_error` marker event so the frontend can cleanly terminate the read loop.

---

## BUG 13 (MEDIUM — HTTP Error Messages Are Cryptic)
**File:** `web/app.js`  
**Problem:** `throw new Error(\`HTTP ${resp.status}\`)` gives users no actionable information. A 401 means wrong token; a 429 means rate-limited. The generic message causes users to think the server is broken.  
**Fix:** Added specific messages for 401 (clears the stored token and prompts re-entry) and 429 (tells user to wait). A 401 also clears `sessionStorage` so the token prompt fires again on next run.

---

## BUG 14 (LOW — CLI `SKIP_VERIFICATION` Env Var Bleeds Between Batch Files)
**File:** `cli.py`  
**Function:** `process_single_file()`  
**Problem:** When `--skip-verify` is passed, the CLI sets `os.environ["SKIP_VERIFICATION"] = "1"`. This is a process-level mutation. If somehow `skip_verify=True` is passed for one file but not others (e.g. via programmatic use), the env var persists for all subsequent files in the same process.  
**Fix:** Added `os.environ.pop("SKIP_VERIFICATION", None)` when `skip_verify=False` to ensure a clean state for each file.

---

## BUG 15 (LOW — Fragile SQLite Checkpoint Import Chain)
**File:** `agents/workflow/orchestrator.py`  
**Function:** `build_modernization_graph()`  
**Problem:** The SQLite checkpointer import tried two paths in nested try/except blocks but was brittle — if the first import succeeded but `SqliteSaver` was renamed in that version, it would crash instead of trying the fallback.  
**Fix:** Replaced nested try/except with a loop over all known module paths, checking for `SqliteSaver` via `getattr`. Also added `PRAGMA journal_mode=WAL` for better multi-reader concurrency.

---

## BUG 16 (LOW — Missing Dependencies in `requirements.txt`)
**File:** `requirements.txt`  
**Problem:** Two dependencies were missing:  
- `langgraph-checkpoint-sqlite` — the orchestrator imports `SqliteSaver` but this package isn't listed  
- `python-multipart` — required by FastAPI for the `/modernize/file` upload endpoint  
Without these, a clean install fails at runtime, not at install time.  
**Fix:** Added both packages with appropriate version constraints.

---

## Summary Table

| # | Severity | File | Bug | Impact |
|---|----------|------|-----|--------|
| 1 | Critical | `differential_tester.py` | Misplaced docstring after early return | Code smell / dead docstring |
| 2 | Critical | `verifier.py` + `orchestrator.py` | No-compiler triggers infinite retry loop | Pipeline hangs, no output |
| 3 | Critical | `orchestrator.py` | Stream generator never yields "done" on total failure | "No result returned" error |
| 4 | High | `modernizer.py` | Retry reads original code, discards progress | All retries produce same broken output |
| 5 | High | `modernizer.py` | `llm_success` used before assignment | Wrong attribution in output |
| 6 | High | `differential_tester.py` | Default compile std = c++23 instead of c++17 | Masks C++17 compatibility errors |
| 7 | High | `orchestrator.py` | LangGraph delta merging loses `None` values | State corruption across nodes |
| 8 | Medium | `app.js` | Verifier UI stage stuck as "active" | Poor UX, confusing pipeline status |
| 9 | Medium | `api.py` | `success=False` when compiler missing | Red "failed" UI on good output |
| 10 | Medium | `fixer.py` | Fixer always uses `compile_only=True` | Linker errors not caught, false success |
| 11 | Medium | `context.py` | SQLite timestamp parsing crashes | LLM cache broken, all calls re-run |
| 12 | Medium | `api.py` + `app.js` | Stream error hides real error message | User sees "no result" not real error |
| 13 | Medium | `app.js` | Cryptic HTTP error codes | 401/429 not explained to user |
| 14 | Low | `cli.py` | `SKIP_VERIFICATION` bleeds between batch files | Verification silently skipped |
| 15 | Low | `orchestrator.py` | Fragile SQLite checkpoint import | Checkpointing silently disabled |
| 16 | Low | `requirements.txt` | Missing `langgraph-checkpoint-sqlite`, `python-multipart` | Clean install fails at runtime |
