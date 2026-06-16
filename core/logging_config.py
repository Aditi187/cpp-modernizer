import logging
import json
import traceback
from datetime import datetime
from contextvars import ContextVar

# Context variables to track request/run across threads/coroutines
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
run_id_var: ContextVar[str] = ContextVar("run_id", default="")

class StructuredJSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        
        req_id = request_id_var.get()
        if req_id:
            log_data["request_id"] = req_id
            
        r_id = run_id_var.get()
        if r_id:
            log_data["run_id"] = str(r_id)
            
        if record.exc_info:
            log_data["exception"] = "".join(traceback.format_exception(*record.exc_info))
            
        return json.dumps(log_data)

def setup_structured_logging(level=logging.INFO):
    """Replaces all existing logging handlers with a single JSON handler."""
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredJSONFormatter())
    logger.addHandler(handler)
