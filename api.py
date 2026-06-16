"""
FastAPI REST API for the Air-Gapped C++ Modernization Engine.

Endpoints:
    GET  /health                        — liveness check
    GET  /info                          — engine version and model configuration
    GET  /api/model/health              — Ollama / LLM backend health check
    POST /modernize                     — modernize a C++ code string
    POST /modernize/stream              — real-time SSE streaming modernization
    POST /modernize/file                — modernize an uploaded .cpp file
    POST /api/dependencies              — extract #include dependency graph
    POST /api/static-analysis           — lightweight static analysis
    GET  /api/project/runs              — list all past modernization runs
    GET  /api/project/status/{run_id}   — get status + file list for a run
    GET  /api/project/audit/{run_id}    — full audit log for a run
    GET  /api/project/report/{run_id}   — download HTML modernization report
    GET  /api/project/stats             — overall project statistics

Run with:
    .venv\\Scripts\\python.exe -m uvicorn api:app --reload --port 8000
"""
from __future__ import annotations

import warnings
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality",
    category=UserWarning,
)


import os
import sys
import time
import difflib
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv

# Ensure project root is importable
_root = Path(__file__).parent.absolute()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

load_dotenv(dotenv_path=_root / ".env", override=False)

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request, APIRouter
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fastapi.security import HTTPBearer
import importlib.metadata

try:
    __version__ = importlib.metadata.version("air-gapped-codebase-modernization-engine")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.2.0"

import logging
import uuid
from core.logging_config import setup_structured_logging, request_id_var, run_id_var
setup_structured_logging(logging.INFO)

from agents.workflow.orchestrator import run_modernization_workflow
from core.differential_tester import compiler_available
from agents.workflow.metrics import calculate_modernization_score, get_safety_rating
from core.project_state import ProjectStateDB

# Project state DB (shared across requests, thread-safe WAL mode)
_DB_PATH = str(_root / ".modernization_state.db")
_project_db = ProjectStateDB(_DB_PATH)

# ---------------------------------------------------------------------------
app = FastAPI(
    title="C++ Modernization Engine API",
    description=(
        "Air-gapped LLM pipeline that transforms legacy C/C++ code "
        "into idiomatic C++17 using a multi-agent LangGraph workflow."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Serve the web UI static files at /web
try:
    app.mount("/web", StaticFiles(directory=str(_root / "web")), name="web")
except Exception:
    # In environments without StaticFiles support, continue without mounting
    pass

@app.on_event("startup")
async def check_compiler():
    from agents.workflow.config import WorkflowConfig
    cfg = WorkflowConfig.from_env()
    if not compiler_available(cfg.compiler_path):
        logger = logging.getLogger("ModernizationEngine")
        logger.warning("No C++ compiler found — verification will fail")

import hmac
import time
import json

security = HTTPBearer(auto_error=True)

# ── Multi-key API Authentication ───────────────────────────────────────────
# Supports two formats:
#   1. Simple: API_AUTH_TOKEN=my_secret_key
#   2. Multi-key: API_AUTH_TOKENS='{"team_a":"secret1", "team_b":"secret2"}'
#
# When using multi-key format, the API tracks which key submitted each job
# for multi-team deployments and audit purposes.

API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
API_AUTH_TOKENS_JSON = os.getenv("API_AUTH_TOKENS", "{}")

_valid_tokens: dict[str, str] = {}  # token -> team_name mapping

try:
    _tokens_map = json.loads(API_AUTH_TOKENS_JSON)
    if isinstance(_tokens_map, dict) and _tokens_map:
        # User provides {"team_a": "key1"} but we need {"key1": "team_a"}
        for team, token in _tokens_map.items():
            _valid_tokens[str(token)] = str(team)
except (json.JSONDecodeError, TypeError):
    pass

if API_AUTH_TOKEN:
    _valid_tokens[API_AUTH_TOKEN] = "default"
elif not _valid_tokens:
    raise RuntimeError(
        "No API authentication configured. Set either:\n"
        "  API_AUTH_TOKEN=<single_key> (simple)\n"
        "  API_AUTH_TOKENS='{\"team_a\": \"key1\", \"team_b\": \"key2\"}' (multi-key)"
    )

def verify_api_key(request: Request):
    """
    Verify the Bearer token or 'token' query param against configured keys.
    """
    # 1. Check query parameter first (useful for HTML links like /report)
    token_query = request.query_params.get("token")
    if token_query:
        provided = token_query
    else:
        # 2. Check Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        provided = auth_header.replace("Bearer ", "").strip()

    for token, team in _valid_tokens.items():
        if hmac.compare_digest(provided, token):
            return (provided, team)
            
    raise HTTPException(
        status_code=401,
        detail="Invalid authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )

# ── CORS ──────────────────────────────────────────────────────────────────
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],  # only what we need
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)

# ── Rate Limiter (slowapi + optional Redis backend) ───────────────────────
# When REDIS_URL is set and Redis is reachable, rate limits are shared across
# all uvicorn workers (correct for --workers N deployments).
# If Redis is unreachable or not configured, falls back to per-process memory
# with a startup warning — the app never crashes on startup due to missing Redis.
import logging as _logging
_limiter_log = _logging.getLogger("api.ratelimiter")

_RATE_LIMIT_STR = f"{os.getenv('RATE_LIMIT_PER_MIN', '60')}/minute"
_redis_url = os.getenv("REDIS_URL")

def _probe_redis(url: str) -> bool:
    """Attempt a lightweight PING to url. Returns True if Redis is reachable."""
    try:
        import redis as _redis_lib
        c = _redis_lib.from_url(url, socket_connect_timeout=2)
        c.ping()
        c.close()
        return True
    except Exception as exc:
        _limiter_log.warning(
            "Redis unreachable at %s (%s). "
            "Falling back to in-memory rate limiting. "
            "Rate limits will NOT be shared across uvicorn workers.",
            url, exc
        )
        return False

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    _use_redis = bool(_redis_url) and _probe_redis(_redis_url)
    if _use_redis:
        limiter = Limiter(key_func=get_remote_address, storage_uri=_redis_url)
        _limiter_log.info("Rate limiter: Redis backend at %s", _redis_url)
    else:
        limiter = Limiter(key_func=get_remote_address)  # in-process fallback
        if not _redis_url:
            _limiter_log.info(
                "Rate limiter: in-memory (set REDIS_URL for multi-worker correctness)"
            )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
except ImportError:
    _limiter_log.warning(
        "slowapi not installed; rate limiting disabled. Run: pip install slowapi redis"
    )

    class _NoopLimiter:
        def limit(self, _s):
            return lambda f: f

    limiter = _NoopLimiter()
    RateLimitExceeded = Exception

api_router = APIRouter()

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

import re as _re

_SAFE_FLAG_RE = _re.compile(r'^[-/_.A-Za-z0-9]+$')
_SAFE_FILENAME_RE = _re.compile(r'^[A-Za-z0-9_.\-]+$')

class ModernizeRequest(BaseModel):
    code: str = Field(..., description="Legacy C++ source code to modernize.")
    filename: str = Field("input.cpp", description="Logical filename (used for logging).")
    skip_verify: bool = Field(False, description="Skip compiler verification.")
    extra_flags: Optional[List[str]] = Field(None, description="Extra compiler flags.")

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    def model_post_init(self, __context):
        # Validate filename — no path traversal, no shell characters
        if not _SAFE_FILENAME_RE.match(self.filename):
            raise ValueError("filename contains invalid characters")
        # Validate extra_flags — only safe compiler-style flags
        if self.extra_flags:
            for flag in self.extra_flags:
                if not _SAFE_FLAG_RE.match(flag):
                    raise ValueError(f"Unsafe flag rejected: {flag!r}")
            # Hard cap on number of flags
            if len(self.extra_flags) > 16:
                raise ValueError("Too many extra_flags (max 16)")
        # Hard cap on code size
        if len(self.code) > 200 * 1024:
            raise ValueError("code exceeds maximum allowed size (200 KB)")


class TransformationDiff(BaseModel):
    added_lines: int
    removed_lines: int
    diff_preview: str  # first 40 lines of unified diff


class ModernizeResponse(BaseModel):
    success: bool
    modernized_code: str
    original_code: str
    diff: TransformationDiff
    score: float
    safety_rating: str
    attribution: Optional[str] = Field(
        None,
        description="How the result was produced: 'deterministic_rules_only', 'llm:<model_name>', or 'llm_verified_compile'"
    )
    tokens_used: Optional[int]
    compiler_status: str
    compiler_output: Optional[str] = Field(
        None,
        description="Full compiler stdout/stderr from verification step"
    )
    sanitizer_findings: Optional[List[str]] = Field(
        None,
        description="List of sanitizer findings (e.g., ASAN, UBSAN results)"
    )
    legacy_patterns_found: int
    processing_time_ms: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_diff(original: str, modernized: str) -> TransformationDiff:
    lines = list(difflib.unified_diff(
        original.splitlines(),
        modernized.splitlines(),
        fromfile="before",
        tofile="after",
        lineterm="",
    ))
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    preview = "\n".join(lines[:40])
    if len(lines) > 40:
        preview += f"\n... ({len(lines) - 40} more lines)"
    return TransformationDiff(added_lines=added, removed_lines=removed, diff_preview=preview)


def _run_pipeline(code: str, filename: str, skip_verify: bool, extra_flags: Optional[list[str]] = None, auth_team: Optional[str] = None) -> ModernizeResponse:
    from agents.workflow.config import WorkflowConfig
    cfg = WorkflowConfig.from_env()
    cfg.skip_verification = skip_verify  # thread-safe: per-request config object
    
    # Register run in database
    run_id = _project_db.start_run(total_files=1, config={"api_call": True, "skip_verification": skip_verify}, submitted_by=auth_team)
    _project_db.register_file(filename, run_id)
    _project_db.mark_running(filename)

    t0 = time.time()
    try:
        from core.logging_config import request_id_var
        state = run_modernization_workflow(
            code=code,
            source_file=filename,
            output_path="",
            config=cfg,
            write_to_disk=False,
            extra_compile_args=extra_flags,
            request_id=request_id_var.get(),
            run_id=run_id
        )
    except Exception as exc:
        _project_db.mark_failed(filename, str(exc))
        _project_db.finish_run(run_id)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    elapsed_ms = int((time.time() - t0) * 1000)

    original = state.get("code", code)
    modernized = state.get("modernized_code", code)
    metrics = state.get("metrics", {})
    score = calculate_modernization_score(state)
    attribution = state.get("pipeline_metadata", {}).get("attribution", "unknown")

    # Extract verification results
    verification_result = state.get("verification_result", {})
    raw_stdout = verification_result.get("raw_stdout", "") or ""
    raw_stderr = verification_result.get("raw_stderr", "") or ""
    compiler_output = (raw_stdout + "\n" + raw_stderr).strip() or verification_result.get("compiler_output")
    sanitizer_findings = verification_result.get("sanitizer_findings")

    # Persist audit record BEFORE returning
    _project_db.mark_done(
        filename,
        output_path="api_response",
        audit_entries=[
            {"rule": r, "attribution": attribution}
            for r in state.get("pipeline_metadata", {}).get("rules_applied", [])
        ],
        complexity=state.get("pipeline_metadata", {}).get("complexity_score", 0),
        llm_called=not state.get("pipeline_metadata", {}).get("llm_skipped", False),
        attribution=attribution,
        duration_ms=elapsed_ms,
    )
    _project_db.finish_run(run_id)

    # Consider success=True when semantic guard passed OR when compiler is unavailable
    # (no compiler != bad code; it means we can't verify, not that it's wrong)
    _compiler_status = verification_result.get("compiler", "unknown")
    _semantic_ok = state.get("semantic_ok", False)
    _compiler_unavailable = _compiler_status in ("not_found", "skipped")
    _has_output = bool(modernized and modernized.strip())
    _is_success = _semantic_ok or (_compiler_unavailable and _has_output)

    return ModernizeResponse(
        success=_is_success,
        modernized_code=modernized,
        original_code=original,
        diff=_build_diff(original, modernized),
        score=round(score, 3),
        safety_rating=get_safety_rating(score),
        attribution=attribution,
        tokens_used=metrics.get("total_tokens"),
        compiler_status=_compiler_status,
        compiler_output=compiler_output,
        sanitizer_findings=sanitizer_findings if sanitizer_findings else None,
        legacy_patterns_found=metrics.get("legacy_pattern_count", 0),
        processing_time_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@api_router.get("/health", tags=["System"])
def health():
    """Liveness check."""
    return {"status": "ok", "engine": "C++ Modernization Engine", "version": __version__}


@api_router.get("/health/compiler", tags=["System"])
def compiler_health():
    """Compiler backend health check."""
    from agents.workflow.config import WorkflowConfig
    cfg = WorkflowConfig.from_env()
    available = compiler_available(cfg.compiler_path)
    return {
        "status": "ok" if available else "error",
        "compiler_path": cfg.compiler_path or "default",
        "available": available
    }


@api_router.get("/api/model/health", tags=["System"])
def model_health():
    """Ollama / LLM backend health check."""
    import urllib.request
    import json
    ollama_url = os.getenv("MODERNIZER_ENDPOINT_BASE", "http://localhost:11434/v1")
    models_url = f"{ollama_url.rstrip('/')}/models"
    models = []
    status = "ok"
    backend = "ollama"
    try:
        req = urllib.request.Request(models_url)
        with urllib.request.urlopen(req, timeout=3) as r:
            data = json.loads(r.read().decode('utf-8'))
            if "data" in data:
                models = [m["id"] for m in data["data"]]
    except Exception:
        try:
            base_url = ollama_url.replace("/v1", "").rstrip("/")
            req = urllib.request.Request(f"{base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as r:
                data = json.loads(r.read().decode('utf-8'))
                if "models" in data:
                    models = [m["name"] for m in data["models"]]
        except Exception as e2:
            status = "error"
            backend = f"failed to connect: {e2}"
    return {
        "status": status,
        "backend": backend,
        "models": models
    }


@api_router.get("/info", tags=["System"])
def info():
    """Returns model routing and feature configuration."""
    import os
    return {
        "version": __version__,
        "analyzer_model": os.getenv("ANALYZER_MODEL", "deepseek-ai/deepseek-v3"),
        "modernizer_model": os.getenv("MODERNIZER_MODEL", "meta/llama-3.3-70b-instruct"),
        "target_standard": "C++17",
        "features": {
            "semantic_guard": True,
            "sqlite_checkpointing": True,
            "batch_processing": True,
            "skip_verify_mode": True,
            "token_tracking": True,
        },
    }


@api_router.post("/modernize", response_model=ModernizeResponse, tags=["Modernization"])
@limiter.limit(_RATE_LIMIT_STR)
async def modernize_code(request: Request, req: ModernizeRequest, creds=Depends(verify_api_key)):
    """
    Modernize a C++ code string.

    Send raw legacy C++ code and receive the modernized C++17 version
    along with a transformation diff and benchmark metrics.
    """
    if len(req.code) > 100 * 1024:
        raise HTTPException(status_code=413, detail="Payload too large (limit 100KB).")
    _, auth_team = creds  # thread-safe: extracted from per-request dependency
    import asyncio
    return await asyncio.to_thread(_run_pipeline, req.code, req.filename, req.skip_verify, req.extra_flags, auth_team)


@api_router.post("/modernize/stream", tags=["Modernization"])
@limiter.limit(_RATE_LIMIT_STR)
async def modernize_code_stream(request: Request, req: ModernizeRequest, creds=Depends(verify_api_key)):
    """
    Modernize C++ code, streaming updates about active node status in real-time,
    and yielding the final ModernizeResponse at the end.
    """

    from agents.workflow.config import WorkflowConfig
    config = WorkflowConfig.from_env()
    config.skip_verification = req.skip_verify
    _, auth_team = creds  # thread-safe: extracted from per-request dependency

    async def stream_generator():
        from agents.workflow.orchestrator import run_modernization_stream_generator
        import json
        t0 = time.time()
        final_state = None
        
        run_id = _project_db.start_run(total_files=1, config={"api_call": True, "stream": True}, submitted_by=auth_team)
        _project_db.register_file(req.filename, run_id)
        _project_db.mark_running(req.filename)
        
        from core.logging_config import run_id_var, request_id_var
        run_id_var.set(str(run_id))
        
        try:
            for event in run_modernization_stream_generator(
                code=req.code,
                source_file=req.filename,
                config=config,
                extra_compile_args=req.extra_flags,
                request_id=request_id_var.get(),
                run_id=str(run_id)
            ):
                if event["node"] == "done":
                    final_state = event["state"]
                else:
                    yield f"data: {json.dumps({'node': event['node'], 'status': 'completed'})}\n\n"
        except Exception as exc:
            logger.error("Stream pipeline error: %s", exc, exc_info=True)
            # Yield error then a done marker so the frontend receives a complete response
            yield f"data: {json.dumps({'error': f'Pipeline stream error: {str(exc)}'})}\n\n"
            # Also yield a done event with empty state so the frontend doesn't show
            # the generic "Pipeline returned no result" fallback error
            yield f"data: {json.dumps({'node': 'done_error', 'message': str(exc)})}\n\n"
            return

        if final_state:
            elapsed_ms = int((time.time() - t0) * 1000)
            original = final_state.get("code", req.code)
            modernized = final_state.get("modernized_code", req.code)
            metrics = final_state.get("metrics", {})
            score = calculate_modernization_score(final_state)
            attribution = final_state.get("pipeline_metadata", {}).get("attribution", "unknown")
            
            # Extract verification results
            verification_result = final_state.get("verification_result", {})
            raw_stdout = verification_result.get("raw_stdout", "") or ""
            raw_stderr = verification_result.get("raw_stderr", "") or ""
            compiler_output = (raw_stdout + "\n" + raw_stderr).strip() or verification_result.get("compiler_output")
            sanitizer_findings = verification_result.get("sanitizer_findings")
            
            _compiler_status = verification_result.get("compiler", "unknown")
            _semantic_ok = final_state.get("semantic_ok", False)
            _compiler_unavailable = _compiler_status in ("not_found", "skipped")
            _has_output = bool(modernized and modernized.strip())
            _is_success = _semantic_ok or (_compiler_unavailable and _has_output)

            resp = ModernizeResponse(
                success=_is_success,
                modernized_code=modernized,
                original_code=original,
                diff=_build_diff(original, modernized),
                score=round(score, 3),
                safety_rating=get_safety_rating(score),
                attribution=attribution,
                tokens_used=metrics.get("total_tokens"),
                compiler_status=_compiler_status,
                compiler_output=compiler_output,
                sanitizer_findings=sanitizer_findings if sanitizer_findings else None,
                legacy_patterns_found=metrics.get("legacy_pattern_count", 0),
                processing_time_ms=elapsed_ms,
            )
            
            _project_db.mark_done(
                req.filename,
                output_path="api_response",
                audit_entries=[{"rule": r, "attribution": attribution} for r in final_state.get("pipeline_metadata", {}).get("rules_applied", [])],
                complexity=final_state.get("pipeline_metadata", {}).get("complexity_score", 0),
                llm_called=not final_state.get("pipeline_metadata", {}).get("llm_skipped", False),
                attribution=attribution,
                duration_ms=elapsed_ms
            )
            _project_db.finish_run(run_id)

            # Yield the final complete result
            resp_dict = resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
            yield f"data: {json.dumps({'node': 'done', 'response': resp_dict})}\n\n"
        else:
            _project_db.mark_failed(req.filename, "Pipeline did not produce a final state")
            _project_db.finish_run(run_id)
            yield f"data: {json.dumps({'error': 'Pipeline did not produce a final state.'})}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Project endpoints (basic dashboard API)
# ---------------------------------------------------------------------------

from fastapi.responses import HTMLResponse


@api_router.post("/api/dependencies", tags=["Project"])
@limiter.limit(_RATE_LIMIT_STR)
def api_dependencies(request: Request, req: dict, creds=Depends(verify_api_key)):
    """Extract a simple dependency list from provided source code.

    This is a lightweight endpoint intended for the dashboard and tests.
    It performs a minimal include extraction and returns nodes/edges.
    """
    code = req.get("code", "") if isinstance(req, dict) else ""
    if len(code) > 100 * 1024:
        raise HTTPException(status_code=413, detail="Payload too large (limit 100KB).")
    import re
    includes = [m.group(1) for m in re.finditer(r'#\s*include\s*["<]([^">]+)[">]', code)]
    nodes = list(dict.fromkeys(includes)) if includes else ["input.cpp"]
    edges = []
    return {"nodes": nodes, "edges": edges}


@api_router.post("/api/static-analysis", tags=["Project"])
@limiter.limit(_RATE_LIMIT_STR)
def api_static_analysis(request: Request, req: dict, creds=Depends(verify_api_key)):
    """Perform a minimal static analysis pass returning a list of findings.

    This is intentionally lightweight for unit tests; real analyzers can be
    plugged in later.
    """
    code = req.get("code", "") if isinstance(req, dict) else ""
    if len(code) > 100 * 1024:
        raise HTTPException(status_code=413, detail="Payload too large (limit 100KB).")
    findings = []
    # Simple heuristic checks
    dangerous_signatures = ["malloc(", "strcpy(", "gets(", "system(", "sprintf("]
    for sig in dangerous_signatures:
        if sig in code:
            findings.append({"signature": sig})
    return {"findings": findings}


@api_router.get("/api/project/runs", tags=["Project"])
def list_runs(creds=Depends(verify_api_key)):
    """List recent modernization runs."""
    runs = _project_db.get_runs()
    return {"runs": runs}


@api_router.get("/api/project/status/{run_id}", tags=["Project"])
def run_status(run_id: int, creds=Depends(verify_api_key)):
    """Return run summary and file list for a specific run."""
    summary = _project_db.get_run_summary(run_id)
    if not summary["run"]:
        raise HTTPException(status_code=404, detail="Run not found")
    return summary


@api_router.get("/api/project/audit/{run_id}", tags=["Project"])
def run_audit(run_id: int, creds=Depends(verify_api_key)):
    """Return audit log for all files in a run."""
    summary = _project_db.get_run_summary(run_id)
    if not summary["run"]:
        raise HTTPException(status_code=404, detail="Run not found")
    audits = {}
    for f in summary["files"]:
        audits[f["path"]] = _project_db.get_audit_log(f["path"])
    return {"run": summary["run"], "audits": audits}


@api_router.get("/api/project/report/{run_id}", tags=["Project"], response_class=HTMLResponse)
def run_report(run_id: int, creds=Depends(verify_api_key)):
    """Return a simple HTML report for the run (generated on-the-fly)."""
    summary = _project_db.get_run_summary(run_id)
    if not summary["run"]:
        raise HTTPException(status_code=404, detail="Run not found")

    import html as html_lib
    # Build a minimal HTML report
    html = ["<html><head><meta charset='utf-8'><title>Run Report</title></head><body>"]
    html.append(f"<h1>Run {html_lib.escape(str(run_id))} Report</h1>")
    r = summary["run"]
    html.append(f"<p>Started: {html_lib.escape(str(r.get('started_at', '')))} - Finished: {html_lib.escape(str(r.get('finished_at', '')))}</p>")
    html.append("<table border='1' cellpadding='6' cellspacing='0'>")
    html.append("<tr><th>File</th><th>Status</th><th>Attribution</th><th>Duration ms</th></tr>")
    for f in summary["files"]:
        html.append("<tr>")
        html.append(f"<td>{html_lib.escape(str(f.get('path', '')))}</td>")
        html.append(f"<td>{html_lib.escape(str(f.get('status', '')))}</td>")
        html.append(f"<td>{html_lib.escape(str(f.get('attribution') or ''))}</td>")
        html.append(f"<td>{html_lib.escape(str(f.get('duration_ms') or 0))}</td>")
        html.append("</tr>")
    html.append("</table>")
    html.append("</body></html>")
    return "\n".join(html)



@api_router.post("/modernize/file", response_model=ModernizeResponse, tags=["Modernization"])
@limiter.limit(_RATE_LIMIT_STR)
async def modernize_file(
    request: Request,
    file: UploadFile = File(..., description="Upload a .cpp or .h source file."),
    skip_verify: bool = False,
    creds=Depends(verify_api_key)
):
    """
    Modernize an uploaded C++ source file.

    Upload a .cpp or .h file and receive the modernized output with diff.
    """
    if not file.filename or not file.filename.endswith((".cpp", ".h", ".cc", ".cxx")):
        raise HTTPException(status_code=422, detail="Only .cpp / .h / .cc / .cxx files are accepted.")

    contents = await file.read()
    if len(contents) > 100 * 1024:
        raise HTTPException(status_code=413, detail="File too large (limit 100KB).")
    try:
        code = contents.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=422, detail="File must be UTF-8 encoded.")

    _, auth_team = creds  # thread-safe: extracted from per-request dependency
    import asyncio
    return await asyncio.to_thread(_run_pipeline, code, file.filename, skip_verify, None, auth_team)





# ---------------------------------------------------------------------------
# Serve Web Dashboard (must be LAST — catches all unmatched routes)
# ---------------------------------------------------------------------------
app.include_router(api_router)

_web_dir = _root / "web"
if _web_dir.exists():
    app.mount("/", StaticFiles(directory=str(_web_dir), html=True), name="web")
