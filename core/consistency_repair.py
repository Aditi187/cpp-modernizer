"""
Semantic Consistency Repair Pass
================================
Fixes cross-cutting type mismatches that arise when independent
modernization rules transform types without propagating changes
to surrounding code.

Runs after both the rule engine and the LLM, before the semantic guard.

Addresses 9 categories of broken output:
  1. new std::string(x) assigned to std::string member
  2. delete on unique_ptr elements
  3. delete on std::string members
  4. unique_ptr returned where raw T* expected (missing .get())
  5. Callback called with std::string where const char* expected
  6. sscanf/fprintf with std::string without .c_str()
  7. free() on std::string members
  8. Dereferenced std::string (*s->member) in printf
"""

from __future__ import annotations

import re
import logging
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1: Discover member types from the current code
# ---------------------------------------------------------------------------

# Matches: std::string name;  or  std::string name{};  or  std::string name = ...;
_STRING_MEMBER_RE = re.compile(
    r'(?:std::string|string)\s+(\w+)\s*(?:[;={])',
)

# Matches: std::unique_ptr<Foo> member;
_UNIQUE_PTR_MEMBER_RE = re.compile(
    r'std::unique_ptr\s*<[^>]+>\s+(\w+)\s*[;{]',
)

# Matches: std::vector<std::unique_ptr<Foo>> member;
_VECTOR_UNIQUE_PTR_RE = re.compile(
    r'std::vector\s*<\s*std::unique_ptr\s*<([^>]+)>\s*>\s+(\w+)\s*[;{]',
)

# Matches struct/class member declarations to build a full type map
_MEMBER_DECL_RE = re.compile(
    r'^\s+((?:std::)?[\w:<>]+(?:\s*\*)?)\s+(\w+)\s*[;={]',
    re.MULTILINE,
)


def _discover_string_members(code: str) -> Set[str]:
    """Return set of member names that are std::string (value type)."""
    return {m.group(1) for m in _STRING_MEMBER_RE.finditer(code)}


def _discover_unique_ptr_containers(code: str) -> Set[str]:
    """Return set of member names that are vector<unique_ptr<T>>."""
    return {m.group(2) for m in _VECTOR_UNIQUE_PTR_RE.finditer(code)}


def _discover_unique_ptr_members(code: str) -> Set[str]:
    """Return set of member names that are unique_ptr<T>."""
    return {m.group(1) for m in _UNIQUE_PTR_MEMBER_RE.finditer(code)}


# ---------------------------------------------------------------------------
# Phase 2: Individual repair functions
# ---------------------------------------------------------------------------

def _fix_new_string_assignment(code: str, string_members: Set[str], repairs: List[str]) -> str:
    """Fix: s->member = new std::string(x);  →  s->member = x;
    
    Bug #1: When char* was converted to std::string, allocation sites
    still use 'new std::string(...)' which produces a std::string* 
    assigned to a std::string. Type mismatch / compile error.
    """
    pattern = re.compile(
        r'(\w+(?:->|\.)\w+)\s*=\s*new\s+std::string\s*\(([^)]*)\)\s*;'
    )
    
    def _replace(m: re.Match) -> str:
        member_expr = m.group(1)
        value = m.group(2).strip()
        # Extract the member name (last part after -> or .)
        parts = re.split(r'->|\.', member_expr)
        member_name = parts[-1] if parts else ""
        if member_name in string_members:
            repairs.append(f"Fix #1: '{member_expr} = new std::string(...)' → direct assignment (std::string is value type)")
            return f'{member_expr} = {value};'
        return m.group(0)
    
    return pattern.sub(_replace, code)


def _fix_delete_on_string_members(code: str, string_members: Set[str], repairs: List[str]) -> str:
    """Fix: delete s->member;  or  delete s[i]->member;  →  (removed)
    
    Bug #7: When char* member was converted to std::string, delete on
    that member is a compile error — std::string is a value type.
    Handles: delete obj->member;  delete obj[i]->member;  delete[] ...
    """
    for member in string_members:
        # Match obj->member, obj.member, obj[expr]->member, obj[expr].member
        pattern = re.compile(
            rf'^\s*delete\s*(?:\[\s*\])?\s*\w+(?:\[[^\]]*\])?(?:->|\.){re.escape(member)}\s*;\s*\n?',
            re.MULTILINE,
        )
        if pattern.search(code):
            repairs.append(f"Fix #7: Removed 'delete ...{member}' — std::string is a value type, auto-destructs")
            code = pattern.sub('', code)
    return code


def _fix_free_on_string_members(code: str, string_members: Set[str], repairs: List[str]) -> str:
    """Fix: free(s->member);  or  free(s[i]->member);  →  (removed)
    
    When char* member was converted to std::string, free() on
    that member is a compile error.
    Handles: free(obj->member);  free(obj[i]->member);
    """
    for member in string_members:
        # Match obj->member, obj.member, obj[expr]->member, obj[expr].member
        pattern = re.compile(
            rf'^\s*free\s*\(\s*\w+(?:\[[^\]]*\])?(?:->|\.){re.escape(member)}\s*\)\s*;\s*\n?',
            re.MULTILINE,
        )
        if pattern.search(code):
            repairs.append(f"Fix: Removed 'free(...{member})' — std::string manages its own memory")
            code = pattern.sub('', code)
    return code


def _fix_logger_fp_null(code: str, repairs: List[str]) -> str:
    """Fix Bug #3/4: Logger that never opens a file.
    
    If a class has 'fp(nullptr)' in constructor and 'if (file)' checks but
    never calls fopen, the logger does nothing. Also fix fprintf calls
    with std::string where .c_str() is needed.
    """
    # Fix: fprintf(fp, "%s\n", msg)  →  fprintf(fp, "%s\n", msg.c_str())
    # where msg is a std::string parameter
    # Look for fprintf/printf calls with std::string reference parameters
    # Pattern: void log(const std::string& msg) { ... fprintf(fp, ..., msg) ... }
    func_param_re = re.compile(
        r'void\s+\w+\s*\((?:[^)]*\bconst\s+std::string\s*&\s*(\w+)[^)]*)\)',
        re.DOTALL,
    )
    for pm in func_param_re.finditer(code):
        param = pm.group(1)
        # Fix fprintf/printf calls with that param
        for func in ('fprintf', 'printf', 'fputs'):
            cfunc_re = re.compile(
                rf'({func}\s*\([^;]*?)\b{re.escape(param)}\b(?!\.c_str\(\))([^;]*;)',
                re.DOTALL,
            )
            count = 0
            while cfunc_re.search(code):
                code = cfunc_re.sub(rf'\1{param}.c_str()\2', code, count=1)
                count += 1
                if count > 10:
                    break
            if count:
                repairs.append(f"Fix #3: Added .c_str() to string parameter '{param}' in {func} call")
    return code


def _fix_delete_on_unique_ptr_elements(code: str, uptr_containers: Set[str], repairs: List[str]) -> str:
    """Fix: for (auto& x : container) { delete x; }  →  remove delete
    
    Bug #2: When raw pointers were converted to unique_ptr, manual 
    delete causes double-free. unique_ptr destructor handles it.
    Also removes standalone 'delete variable;' when unique_ptr is present.
    """
    if not uptr_containers and 'unique_ptr' not in code:
        return code
    
    # Strategy: If we see unique_ptr anywhere, remove any standalone
    # 'delete <identifier>;' that isn't 'delete this;' or 'delete[]'
    # This is safe because if you have unique_ptr, you should never
    # be manually deleting what it owns.
    
    # Remove: delete varname;  (but not delete[] or delete this)
    pattern = re.compile(
        r'^\s*delete\s+(?!\[\s*\])(?!this\b)(\w+)\s*;\s*\n?',
        re.MULTILINE,
    )
    matches = pattern.findall(code)
    if matches:
        repairs.append(f"Fix #2: Removed {len(matches)} 'delete' statement(s) — unique_ptr handles deallocation automatically")
        code = pattern.sub('', code)
    
    return code


def _fix_unique_ptr_return(code: str, repairs: List[str]) -> str:
    """Fix: return s;  →  return s.get();  (inside range-for over unique_ptr)
    
    Bug #6: When a function returns T* but iterates over 
    vector<unique_ptr<T>>, returning the loop variable directly
    returns unique_ptr<T> instead of T*. Need .get().
    """
    if 'unique_ptr' not in code:
        return code
    
    # Detect pattern: for (auto/const auto& var : container) { ... return var; ... }
    # This is a simplified heuristic: look for 'return <identifier>;' inside
    # a block that also has a range-for with unique_ptr container
    
    # Find all range-for loop variables
    range_for_re = re.compile(
        r'for\s*\(\s*(?:const\s+)?auto\s*&?\s+(\w+)\s*:\s*(\w+)\s*\)'
    )
    
    for m in range_for_re.finditer(code):
        loop_var = m.group(1)
        container = m.group(2)
        
        # Check if the container is a vector<unique_ptr<...>>
        container_decl_re = re.compile(
            rf'std::vector\s*<\s*std::unique_ptr\s*<[^>]+>\s*>\s*(?:&\s*)?{re.escape(container)}\b'
        )
        if not container_decl_re.search(code):
            continue
        
        # Find 'return loop_var;' and replace with 'return loop_var.get();'
        return_re = re.compile(
            rf'(\breturn\s+){re.escape(loop_var)}\s*;'
        )
        if return_re.search(code):
            repairs.append(f"Fix #6: 'return {loop_var}' → 'return {loop_var}.get()' (unique_ptr element needs .get() for raw pointer return)")
            code = return_re.sub(rf'\g<1>{loop_var}.get();', code)
    
    return code


def _fix_cstr_in_c_functions(code: str, string_members: Set[str], repairs: List[str]) -> str:
    """Fix: fprintf(fp, "%s", s->member)  →  fprintf(fp, "%s", s->member.c_str())
    
    Bugs #8, #9: C functions (printf, fprintf, sscanf, sprintf, snprintf)
    require char* for %s. When members are now std::string, need .c_str().
    Also fixes: *s->member (dereferencing std::string is nonsensical).
    """
    if not string_members:
        return code
    
    c_funcs = r'(?:printf|fprintf|sprintf|snprintf|sscanf|fputs|puts)'
    
    for member in string_members:
        # Fix *obj->member → obj->member.c_str()  (Bug #9: deref on std::string)
        deref_pattern = re.compile(
            rf'\*\s*(\w+(?:->|\.){re.escape(member)})\b'
        )
        if deref_pattern.search(code):
            repairs.append(f"Fix #9: Removed dereference on '{member}' — std::string is not a pointer")
            code = deref_pattern.sub(rf'\1.c_str()', code)
        
        # Fix: c_func(..., obj->member, ...) → c_func(..., obj->member.c_str(), ...)
        # Only inside C-function calls, where member is used as a %s argument
        # Strategy: find occurrences of obj->member inside c_func() calls
        # and add .c_str() if not already present
        func_call_re = re.compile(
            rf'({c_funcs}\s*\([^;]*?)(\w+(?:->|\.){re.escape(member)})\b(?!\.c_str\(\))([^;]*;)',
            re.DOTALL,
        )
        count = 0
        while func_call_re.search(code):
            code = func_call_re.sub(r'\1\2.c_str()\3', code, count=1)
            count += 1
            if count > 20:  # safety valve
                break
        if count:
            repairs.append(f"Fix #8: Added .c_str() to '{member}' in {count} C-function call(s)")
    
    # Also fix standalone string variables (not member access) in C functions
    # Find local std::string variable declarations
    local_strings = set()
    for m in re.finditer(r'std::string\s+(\w+)\s*[;={]', code):
        local_strings.add(m.group(1))
    
    for var in local_strings:
        func_call_re = re.compile(
            rf'({c_funcs}\s*\([^;]*?)\b{re.escape(var)}\b(?!\.c_str\(\)|\s*\.)([^;]*;)',
            re.DOTALL,
        )
        count = 0
        while func_call_re.search(code):
            code = func_call_re.sub(rf'\1{var}.c_str()\2', code, count=1)
            count += 1
            if count > 20:
                break
        if count:
            repairs.append(f"Fix #8: Added .c_str() to local variable '{var}' in {count} C-function call(s)")
    
    return code


def _fix_callback_type_mismatch(code: str, string_members: Set[str], repairs: List[str]) -> str:
    """Fix: callback(id, user)  →  callback(id, user.c_str())
    
    Bug #5: When a callback signature expects const char* but the
    argument is now std::string, need .c_str() conversion.
    """
    # Detect function pointer typedefs with const char* parameters
    # typedef void (*Name)(int, const char*);
    # or: using Name = std::function<void(int, const char*)>;
    typedef_re = re.compile(
        r'(?:typedef\s+\w+\s*\(\s*\*\s*(\w+)\s*\)\s*\(([^)]*)\)\s*;|'
        r'using\s+(\w+)\s*=\s*(?:std::function\s*<\s*\w+\s*\(([^)]*)\)\s*>|void\s*\(\s*\*\s*\)\s*\(([^)]*)\))\s*;)'
    )
    
    # Collect all callback type names that have const char* in their signature
    callback_types: Set[str] = set()
    for m in typedef_re.finditer(code):
        name = m.group(1) or m.group(3)
        params = m.group(2) or m.group(4) or m.group(5) or ""
        if name and ('const char*' in params or 'const char *' in params):
            callback_types.add(name)
    
    if not callback_types:
        return code
    
    # Find all variable names (members or locals) that have a callback type
    # e.g., SessionCallback callback;  or  SessionCallback cb;
    callback_vars: Set[str] = set()
    for cb_type in callback_types:
        for m in re.finditer(rf'\b{re.escape(cb_type)}\s+(\w+)\s*[;=]', code):
            callback_vars.add(m.group(1))
        # Also add the type name itself (direct invocation by type name)
        callback_vars.add(cb_type)
    
    # For each callback variable invocation, fix string args
    for cb_var in callback_vars:
        call_re = re.compile(
            rf'\b{re.escape(cb_var)}\s*\(([^)]*)\)\s*;'
        )
        for call_m in call_re.finditer(code):
            args = call_m.group(1)
            new_args = []
            changed = False
            for arg in args.split(','):
                arg_stripped = arg.strip()
                # Extract the bare identifier (handle obj->member, obj.member, or plain var)
                bare = re.sub(r'->|\.', '.', arg_stripped).split('.')[-1] if arg_stripped else ""
                if bare in string_members and '.c_str()' not in arg_stripped:
                    new_args.append(arg.replace(arg_stripped, arg_stripped + '.c_str()'))
                    changed = True
                else:
                    new_args.append(arg)
            
            if changed:
                old_call = call_m.group(0)
                new_call = f'{cb_var}({", ".join(new_args)});'
                code = code.replace(old_call, new_call, 1)
                repairs.append(f"Fix #5: Added .c_str() to callback '{cb_var}' arguments — callback expects const char*")
    
    return code


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def repair_semantic_consistency(code: str) -> Tuple[str, List[str]]:
    """
    Post-processing pass that fixes cross-cutting type mismatches.
    
    Runs after all modernization rules and LLM output, before the
    semantic guard. Detects and corrects inconsistencies that arise 
    when independent transformations change types without propagating
    to surrounding code.
    
    Returns:
        (repaired_code, list_of_repair_descriptions)
    """
    repairs: List[str] = []
    
    # Discover what types exist in the current code
    string_members = _discover_string_members(code)
    uptr_containers = _discover_unique_ptr_containers(code)
    uptr_members = _discover_unique_ptr_members(code)
    
    logger.debug(
        "[CONSISTENCY] Discovered: string_members=%s, unique_ptr_containers=%s, unique_ptr_members=%s",
        string_members, uptr_containers, uptr_members,
    )
    
    # Apply repairs in dependency order
    # (string fixes first, then pointer fixes, then C-function fixes)
    
    # 1. Fix new std::string() assignments (Bug #1)
    code = _fix_new_string_assignment(code, string_members, repairs)
    
    # 2. Remove delete on string members (Bug #7)
    code = _fix_delete_on_string_members(code, string_members, repairs)
    
    # 3. Remove free() on string members
    code = _fix_free_on_string_members(code, string_members, repairs)
    
    # 4. Remove delete on unique_ptr elements (Bug #2)
    code = _fix_delete_on_unique_ptr_elements(code, uptr_containers, repairs)
    
    # 5. Fix unique_ptr returns (Bug #6)
    code = _fix_unique_ptr_return(code, repairs)
    
    # 6. Fix C-function calls with std::string args (Bugs #8, #9)
    code = _fix_cstr_in_c_functions(code, string_members, repairs)
    
    # 7. Fix callback type mismatches (Bug #5)
    code = _fix_callback_type_mismatch(code, string_members, repairs)
    
    # 8. Fix Logger/fprintf .c_str() for std::string parameters (Bug #3)
    code = _fix_logger_fp_null(code, repairs)
    
    # Clean up: remove blank lines left by deleted statements
    code = re.sub(r'\n{3,}', '\n\n', code)
    
    if repairs:
        logger.info("[CONSISTENCY] Applied %d semantic repairs:", len(repairs))
        for r in repairs:
            logger.info("  • %s", r)
    else:
        logger.info("[CONSISTENCY] No cross-cutting inconsistencies detected.")
    
    return code, repairs
