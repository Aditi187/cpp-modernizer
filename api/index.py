from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents.workflow.orchestrator import run_modernization_workflow

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Air-Gapped C++ Modernization Engine API",
    description="API for transforming legacy C++ into modern C++17.",
    version="0.1.0",
    root_path="/api"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class ModernizationRequest(BaseModel):
    code: str
    source_file: Optional[str] = "input.cpp"
    output_path: Optional[str] = None

@app.get("/")
async def root():
    return {
        "status": "online",
        "engine": "Modernization Engine API",
        "endpoints": {
            "modernize": "/modernize (POST)",
            "health": "/health (GET)"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/modernize")
async def modernize(request: ModernizationRequest):
    """
    Triggers the modernization workflow for the provided C++ code.
    """
    try:
        # We use a temporary or virtual path for the source file in serverless envs
        source_file = request.source_file or "api_input.cpp"
        
        # Run the workflow
        # Note: orchestrator writes to file, but we'll return the result code in JSON
        result_state = run_modernization_workflow(
            code=request.code,
            source_file=source_file,
            output_path=request.output_path
        )
        
        return {
            "success": True,
            "modernized_code": result_state.get("modernized_code", ""),
            "analysis": result_state.get("analysis_report", ""),
            "plan": result_state.get("plan_id", ""),
            "attempt_count": result_state.get("attempt_count", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
