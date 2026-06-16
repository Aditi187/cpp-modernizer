"""
Shared code utility functions for the workflow.

Provides common operations used by multiple workflow nodes.
"""

import re
from typing import Optional

_CODE_FENCE_RE = re.compile(
    r"```(?:cpp|c\+\+|cxx|cc|hpp|h)?\s*\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

def extract_code(text: Optional[str], default: Optional[str] = None) -> str:
    """
    Extract the best C++ code block from LLM output.
    Preference order:
      1. Block containing 'main()'
      2. Block with the most #include and class/struct/function keywords
      3. Longest block by non-whitespace length
    If no code fences, return the whole text stripped.
    """
    if not text:
        return default if default is not None else ""
        
    blocks = list(_CODE_FENCE_RE.finditer(text))
    if not blocks:
        return default if default is not None else text.strip()
        
    # 1. Prefer block with main()
    for m in blocks:
        code = m.group(1)
        if 'main(' in code:
            return code.strip()
            
    # 2. Prefer block with most includes and class/struct/function
    def score_block(code_str):
        includes = code_str.count('#include')
        classes = code_str.count('class ') + code_str.count('struct ')
        functions = code_str.count('(')  # crude, but helps
        return includes * 3 + classes * 2 + functions
        
    best = max(blocks, key=lambda m: (score_block(m.group(1)), len(m.group(1).strip())))
    return best.group(1).strip()
