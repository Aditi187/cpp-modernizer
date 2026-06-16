from __future__ import annotations

"""
Expanded Rule-Based C++ Modernizer
===================================
Goal: Handle 90%+ of legacy C/C++ patterns deterministically — no LLM needed.
Only truly ambiguous semantic rewrites (complex ownership transfers, deep template
refactoring) fall through to the LLM.

Rule ordering matters:
  1. Headers (must come first so later rules can match modern headers)
  2. NULL / nullptr
  3. typedef → using
  4. #define constants → constexpr
  5. C-style casts → static_cast (non-malloc contexts)
  6. malloc/new/delete/free → smart pointers / RAII
  7. char* / strcpy / sprintf → std::string
  8. printf → std::cout (simple cases)
  9. Legacy containers → STL
  10. Loop modernization → range-based for
  11. throw() → noexcept
  12. auto keyword opportunities
  13. Header auto-injection
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule definition
# ---------------------------------------------------------------------------

class ModernizationRule:
    def __init__(
        self,
        pattern: str,
        replacement: str,
        description: str,
        ast_triggers: tuple[str, ...] = (),
        hint_only: bool = False,
        flags: int = re.MULTILINE,
        mask_strings: bool = True,
    ):
        self.pattern = re.compile(pattern, flags)
        self.replacement = replacement
        self.description = description
        self.ast_triggers = ast_triggers
        self.hint_only = hint_only
        self.mask_strings = mask_strings


# ---------------------------------------------------------------------------
# Comment/string masking (keeps newlines stable for span-based replacement)
# ---------------------------------------------------------------------------

_COMMENTS_AND_STRINGS_RE = re.compile(
    r'R"([^()\\\s]{0,16})\((?:(?!\)\1").)*\)\1"'  # raw string
    r'|//[^\n]*'                                    # line comment
    r'|/\*.*?\*/'                                   # block comment
    r'|"(?:\\.|[^"\\])*"'                           # string literal
    r"|'(?:\\.|[^'\\])*'",                          # char literal
    re.DOTALL,
)


def _mask_comments_and_strings(code: str) -> str:
    def _blank(m: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))
    return _COMMENTS_AND_STRINGS_RE.sub(_blank, code)


def _apply_rule(code: str, rule: ModernizationRule) -> Tuple[str, int]:
    """Apply a rule only on real code tokens (skipping comments/strings)."""
    masked = _mask_comments_and_strings(code) if rule.mask_strings else code
    matches = list(rule.pattern.finditer(masked))
    if not matches:
        return code, 0

    chunks: List[str] = []
    cursor = 0
    count = 0
    for m in matches:
        start, end = m.span()
        chunks.append(code[cursor:start])
        # Expand using the *original* code span (for capture groups)
        original_match = rule.pattern.match(code, start, end) or m
        try:
            chunks.append(original_match.expand(rule.replacement))
        except Exception:
            chunks.append(code[start:end])  # safe fallback
        cursor = end
        count += 1

    chunks.append(code[cursor:])
    return "".join(chunks), count


# ---------------------------------------------------------------------------
# Transformation rules  (ordered: most specific / safest first)
# ---------------------------------------------------------------------------

_RULES: List[ModernizationRule] = [

    # ── 1. C standard headers → C++ headers ──────────────────────────────
    *[
        ModernizationRule(
            pattern=rf"#\s*include\s*<{c_hdr}>",
            replacement=f"#include <{cpp_hdr}>",
            description=f"<{c_hdr}> → <{cpp_hdr}>",
        )
        for c_hdr, cpp_hdr in [
            ("assert.h", "cassert"), ("ctype.h",  "cctype"),
            ("errno.h",  "cerrno"),  ("float.h",  "cfloat"),
            ("limits.h", "climits"), ("locale.h", "clocale"),
            ("math.h",   "cmath"),   ("setjmp.h", "csetjmp"),
            ("signal.h", "csignal"), ("stdarg.h", "cstdarg"),
            ("stddef.h", "cstddef"), ("stdio.h",  "cstdio"),
            ("stdlib.h", "cstdlib"), ("string.h", "cstring"),
            ("time.h",   "ctime"),   ("wchar.h",  "cwchar"),
            ("wctype.h", "cwctype"), ("stdint.h", "cstdint"),
            ("inttypes.h", "cinttypes"),
        ]
    ],

    # ── 2. NULL → nullptr ────────────────────────────────────────────────
    ModernizationRule(
        pattern=r"\bNULL\b",
        replacement="nullptr",
        description="NULL → nullptr",
        ast_triggers=("null_macro",),
    ),

    # ── 3. typedef → using / struct ───────────────────────────────────────────────
    # Simple: typedef int MyInt;
    ModernizationRule(
        pattern=r"\btypedef\s+((?:(?!typedef|;|\{).)+?)\s+([A-Za-z_]\w*)\s*;",
        replacement=r"using \2 = \1;",
        description="typedef → using",
        flags=re.MULTILINE | re.DOTALL,
    ),
    # typedef struct { ... } Name;
    ModernizationRule(
        pattern=r"\btypedef\s+struct\s*(?:\w+\s*)?\{([^}]*)\}\s*([A-Za-z_]\w*)\s*;",
        replacement=r"struct \2 {\1};",
        description="typedef struct {...} Name; → struct Name {...};",
        flags=re.MULTILINE | re.DOTALL,
    ),

    # ── 4. #define constants → constexpr ─────────────────────────────────
    # Integer/float constants only (not function macros)
    ModernizationRule(
        pattern=r"^[ \t]*#\s*define\s+([A-Z][A-Z0-9_]*)\s+([0-9]+(?:\.[0-9]+)?(?:f|F|L|UL|ULL|LL)?)\b[ \t]*$",
        replacement=r"constexpr auto \1 = \2;",
        description="#define constant → constexpr",
    ),
    # String constant macros
    ModernizationRule(
        pattern=r'^[ \t]*#\s*define\s+([A-Z][A-Z0-9_]*)\s+("(?:[^"\\]|\\.)*")[ \t]*$',
        replacement=r'constexpr auto \1 = \2;',
        description="#define string constant → constexpr",
    ),

    # ── 5. throw() → noexcept ────────────────────────────────────────────
    ModernizationRule(
        pattern=r"\bthrow\s*\(\s*\)",
        replacement="noexcept",
        description="throw() → noexcept",
    ),

    # ── 6. std::auto_ptr → std::unique_ptr ───────────────────────────────
    ModernizationRule(
        pattern=r"\bstd\s*::\s*auto_ptr\b",
        replacement="std::unique_ptr",
        description="std::auto_ptr → std::unique_ptr",
        ast_triggers=("auto_ptr",),
    ),

    # (T*)malloc(sizeof(T))  →  new T()
    ModernizationRule(
        pattern=r"\(\s*([A-Za-z_]\w*)\s*\*\s*\)\s*malloc\s*\(\s*sizeof\s*\(\s*\1\s*\)\s*\)",
        replacement=r"new \1()",
        description="(T*)malloc(sizeof(T)) → new T()",
        ast_triggers=("malloc_usage",),
    ),
    # (T*)malloc(n * sizeof(T))  →  new T[n]
    ModernizationRule(
        pattern=r"\(\s*([A-Za-z_]\w*)\s*\*\s*\)\s*malloc\s*\(\s*([A-Za-z_]\w*)\s*\*\s*sizeof\s*\(\s*\1\s*\)\s*\)",
        replacement=r"new \1[\2]",
        description="(T*)malloc(n*sizeof(T)) → new T[n]",
        ast_triggers=("malloc_usage",),
    ),
    # calloc(n, sizeof(T)) → std::vector<T>(n, T{})
    ModernizationRule(
        pattern=r"calloc\s*\(\s*([A-Za-z0-9_]+)\s*,\s*sizeof\s*\(\s*([A-Za-z_]\w*)\s*\)\s*\)",
        replacement=r"std::vector<\2>(\1)",
        description="calloc(n, sizeof(T)) → std::vector<T>(n)",
        ast_triggers=("malloc_usage",),
    ),
    # free(ptr) → delete ptr
    ModernizationRule(
        pattern=r"\bfree\s*\(\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*|->[A-Za-z_]\w*)*)\s*\)\s*;",
        replacement=r"delete \1;",
        description="free() → delete",
        ast_triggers=("free_usage",),
    ),
    # (These rules were removed because T* = make_unique<T>() is invalid C++)
    # new T[n] → std::vector<T>(n)
    ModernizationRule(
        pattern=r"\bnew\s+([A-Za-z_]\w*)\s*\[([^\]]+)\]",
        replacement=r"std::vector<\1>(\2)",
        description="new T[n] → std::vector<T>(n)",
        hint_only=True,
    ),

    # ── 8. C-style casts → static_cast (safe, non-malloc contexts) ───────
    # (int)x, (double)x, (float)x, (long)x, (unsigned)x
    ModernizationRule(
        pattern=r"\(\s*(int|long|unsigned\s+int|unsigned\s+long|float|double|size_t|uint32_t|uint64_t|int32_t|int64_t|char)\s*\)\s*([A-Za-z_]\w*)\b",
        replacement=r"static_cast<\1>(\2)",
        description="C-style primitive cast → static_cast",
        ast_triggers=("c_style_cast",),
    ),

    # ── 9. strcpy / strcat / strlen / strcmp → std::string methods ────────
    # strcpy(dest, src) → dest = src  (when dest is std::string)
    ModernizationRule(
        pattern=r"\bstrcpy\s*\(\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*,\s*([^)]+)\)",
        replacement=r"\1 = \2",
        description="strcpy → assignment",
        ast_triggers=("strcpy_usage",),
    ),
    # strcat(dest, src) → dest += src
    ModernizationRule(
        pattern=r"\bstrcat\s*\(\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*,\s*([^)]+)\)",
        replacement=r"\1 += \2",
        description="strcat → += operator",
        ast_triggers=("strcat_usage",),
    ),
    # strcmp(a, b) == 0 → a == b
    ModernizationRule(
        pattern=r"\bstrcmp\s*\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\)\s*==\s*0",
        replacement=r"\1 == \2",
        description="strcmp(a,b)==0 → a==b",
    ),
    # strcmp(a, b) != 0 → a != b
    ModernizationRule(
        pattern=r"\bstrcmp\s*\(\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\)\s*!=\s*0",
        replacement=r"\1 != \2",
        description="strcmp(a,b)!=0 → a!=b",
    ),
    # strlen(s) → s.size() (when s is a variable, not a literal)
    ModernizationRule(
        pattern=r"\bstrlen\s*\(\s*([A-Za-z_]\w*)\s*\)",
        replacement=r"\1.size()",
        description="strlen(s) → s.size()",
        ast_triggers=("strlen_usage",),
    ),

    # ── 10. memset(ptr, 0, sizeof(*ptr)) → *ptr = {} ─────────────────────
    ModernizationRule(
        pattern=r"\bmemset\s*\(\s*&\s*([A-Za-z_]\w*)\s*,\s*0\s*,\s*sizeof\s*\(\s*\1\s*\)\s*\)",
        replacement=r"\1 = {}",
        description="memset(&v, 0, sizeof(v)) → v = {}",
    ),

    # ── 11. printf simple cases → std::cout ──────────────────────────────
    # printf("text\n") → std::cout << "text\n";
    ModernizationRule(
        pattern=r'\bprintf\s*\(\s*("(?:[^"\\]|\\.)*\\n")\s*\)\s*;',
        replacement=r"std::cout << \1;",
        description='printf("text\\n") → std::cout',
        ast_triggers=("printf_usage",),
        mask_strings=False,
    ),
    # printf("text\n", var) → std::cout << "text\n" << var;  (single arg)
    ModernizationRule(
        pattern=r'\bprintf\s*\(\s*("[^"]*%[sd][^"]*")\s*,\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*|->[A-Za-z_]\w*)*)\s*\)\s*;',
        replacement=r"std::cout << \2 << std::endl;  // TODO: format string may need review",
        description="printf(fmt, var) → std::cout (single arg)",
        ast_triggers=("printf_usage",),
        hint_only=True,
        mask_strings=False,
    ),

    # ── 12. Ranged-based for loop ─────────────────────────────────────────
    # for (int i = 0; i < vec.size(); i++) → for (auto& elem : vec) [hint]
    ModernizationRule(
        pattern=r"\bfor\s*\(\s*(?:int|size_t|auto)\s+(\w+)\s*=\s*0\s*;\s*\1\s*<\s*(\w+)\.size\(\)\s*;\s*\+\+\1\s*\)",
        replacement=r"for (auto& \1_elem : \2)  // was: for(int \1=0; \1<\2.size(); \1++)",
        description="index loop → range-based for (hint)",
        hint_only=True,  # structural change needs human review
    ),

    # ── 13. 0 → nullptr in pointer contexts (bool/pointer assignments) ────
    # Type* ptr = 0; → Type* ptr = nullptr;
    ModernizationRule(
        pattern=r"(\b[A-Za-z_]\w*\s*\*\s*[A-Za-z_]\w*\s*=\s*)0\s*;",
        replacement=r"\1nullptr;",
        description="pointer = 0 → pointer = nullptr",
    ),

    # ── 14. Redundant duplicate statements ───────────────────────────────
    ModernizationRule(
        pattern=r"^([ \t]*[^\n{};]+;)\n\1$",
        replacement=r"\1",
        description="Remove duplicate consecutive statement",
        flags=re.MULTILINE,
    ),

    # ── 15. Register keyword removal (removed in C++17) ──────────────────
    ModernizationRule(
        pattern=r"\bregister\s+",
        replacement="",
        description="Remove 'register' keyword (removed in C++17)",
    ),

    # ── 16. volatile/restrict cleanup hints ───────────────────────────────
    # (hint only — needs review for thread-safety context)

    # ── 17. #pragma once (add if header guard detected) ──────────────────
    # (handled in _ensure_pragma_once below, not as a rule)

    # ── 18. INT_MAX / DBL_MAX → std::numeric_limits ──────────────────────
    ModernizationRule(
        pattern=r"\bINT_MAX\b",
        replacement="std::numeric_limits<int>::max()",
        description="INT_MAX → std::numeric_limits<int>::max()",
    ),
    ModernizationRule(
        pattern=r"\bINT_MIN\b",
        replacement="std::numeric_limits<int>::min()",
        description="INT_MIN → std::numeric_limits<int>::min()",
    ),
    ModernizationRule(
        pattern=r"\bDBL_MAX\b",
        replacement="std::numeric_limits<double>::max()",
        description="DBL_MAX → std::numeric_limits<double>::max()",
    ),
    ModernizationRule(
        pattern=r"\bFLT_MAX\b",
        replacement="std::numeric_limits<float>::max()",
        description="FLT_MAX → std::numeric_limits<float>::max()",
    ),
    ModernizationRule(
        pattern=r"\bSIZE_MAX\b",
        replacement=r"std::numeric_limits<std::size_t>::max()",
        description="SIZE_MAX → std::numeric_limits<size_t>::max()",
    ),

    # ── 19. UINT_MAX / LONG_MAX / LLONG_MAX ──────────────────────────────
    ModernizationRule(
        pattern=r"\bUINT_MAX\b",
        replacement="std::numeric_limits<unsigned int>::max()",
        description="UINT_MAX → std::numeric_limits<unsigned int>::max()",
    ),
    ModernizationRule(
        pattern=r"\bLONG_MAX\b",
        replacement="std::numeric_limits<long>::max()",
        description="LONG_MAX → std::numeric_limits<long>::max()",
    ),
    ModernizationRule(
        pattern=r"\bLLONG_MAX\b",
        replacement="std::numeric_limits<long long>::max()",
        description="LLONG_MAX → std::numeric_limits<long long>::max()",
    ),

    # ── 20. std::endl → '\n' for performance ─────────────────────────────
    # std::endl flushes the buffer — almost always unnecessary
    ModernizationRule(
        pattern=r"<<\s*std::endl\b",
        replacement=r"<< '\\n'",
        description="std::endl → '\\n' (avoids unnecessary flush)",
    ),

    # ── 21. exit(0)/exit(1) in main → return 0/return 1 ──────────────────
    ModernizationRule(
        pattern=r"\bexit\s*\(\s*0\s*\)\s*;",
        replacement="return 0;",
        description="exit(0) → return 0",
    ),
    ModernizationRule(
        pattern=r"\bexit\s*\(\s*1\s*\)\s*;",
        replacement="return 1;",
        description="exit(1) → return 1",
    ),

    # ── 22. (void) cast removal (suppressing unused-param warnings) ───────
    # Modern way: just comment out or use [[maybe_unused]]
    ModernizationRule(
        pattern=r"\(void\)\s*([A-Za-z_]\w*)\s*;",
        replacement=r"[[maybe_unused]] auto& \1_unused = \1;  // suppress unused warning",
        description="(void)param → [[maybe_unused]]",
        hint_only=True,
    ),

    # ── 23. char[] string literals → std::string ─────────────────────────
    # char buf[] = "literal"; → std::string buf = "literal";
    ModernizationRule(
        pattern=r'\bchar\s+(\w+)\s*\[\s*\]\s*=\s*("(?:[^"\\]|\\.)*")\s*;',
        replacement=r'std::string \1 = \2;',
        description='char[] = "..." → std::string',
        ast_triggers=("char_array_string",),
    ),

    # ── 24. char* func args → const std::string& (hint) ─────────────────
    # const char* name → const std::string& name  (when no pointer arithmetic)
    ModernizationRule(
        pattern=r"\bconst\s+char\s*\*\s*(\w+)\b",
        replacement=r"const std::string& \1",
        description="const char* arg → const std::string& arg",
        ast_triggers=("char_pointer",),
    ),

    # ── 25. Trailing return type for long signatures ───────────────────────
    # Already done by C++11 — just flag with hint when return type matches
    # (Complex: skip, handle via LLM gate)

    # ── 26. bool literals: TRUE/FALSE → true/false ───────────────────────
    ModernizationRule(
        pattern=r"\bTRUE\b",
        replacement="true",
        description="TRUE → true",
    ),
    ModernizationRule(
        pattern=r"\bFALSE\b",
        replacement="false",
        description="FALSE → false",
    ),

    # ── 27. BOOL typedef → bool ──────────────────────────────────────────
    ModernizationRule(
        pattern=r"\bBOOL\b",
        replacement="bool",
        description="BOOL → bool",
    ),

    # ── 28. int i; for() → auto (range-based already handled) ────────────
    # Catch common iterator declarations: vector<T>::iterator → auto
    ModernizationRule(
        pattern=r"\bstd::vector<(\w+)>::iterator\b",
        replacement=r"auto",
        description="std::vector<T>::iterator → auto",
        hint_only=True,
    ),
    ModernizationRule(
        pattern=r"\bstd::map<([^>]+)>::iterator\b",
        replacement=r"auto",
        description="std::map<T,U>::iterator → auto",
        hint_only=True,
    ),
    ModernizationRule(
        pattern=r"\bstd::list<(\w+)>::iterator\b",
        replacement=r"auto",
        description="std::list<T>::iterator → auto",
        hint_only=True,
    ),

    # ── 29. sprintf → (comment hint only — needs review) ─────────────────
    ModernizationRule(
        pattern=r"\bsprintf\s*\(",
        replacement=r"snprintf(  // TODO: consider std::format (C++20) or ostringstream",
        description="sprintf → snprintf (safer)",
        ast_triggers=("sprintf_usage",),
    ),

    # ── 30. gets() → std::getline ────────────────────────────────────────
    ModernizationRule(
        pattern=r"\bgets\s*\(\s*([A-Za-z_]\w*)\s*\)\s*;",
        replacement=r"std::getline(std::cin, \1);",
        description="gets() → std::getline (secure)",
        ast_triggers=("gets_usage",),
    ),

    # ── 31. atoi/atof/atol → std::stoi/stof/stol ─────────────────────────
    ModernizationRule(
        pattern=r"\batoi\s*\(",
        replacement=r"std::stoi(",
        description="atoi → std::stoi",
        ast_triggers=("atoi_usage",),
    ),
    ModernizationRule(
        pattern=r"\batof\s*\(",
        replacement=r"std::stof(",
        description="atof → std::stof",
        ast_triggers=("atof_usage",),
    ),
    ModernizationRule(
        pattern=r"\batol\s*\(",
        replacement=r"std::stol(",
        description="atol → std::stol",
        ast_triggers=("atol_usage",),
    ),

    # ── 32. abs() → std::abs() ───────────────────────────────────────────
    ModernizationRule(
        pattern=r"(?<![:\w])abs\s*\(",
        replacement=r"std::abs(",
        description="abs() → std::abs() (avoids implicit C abs)",
    ),

    # ── 33. Explicit bool: if (ptr != NULL) → if (ptr) ──────────────────
    ModernizationRule(
        pattern=r"if\s*\(\s*(\w+)\s*!=\s*nullptr\s*\)",
        replacement=r"if (\1)",
        description="if (ptr != nullptr) → if (ptr)",
    ),
    ModernizationRule(
        pattern=r"if\s*\(\s*(\w+)\s*==\s*nullptr\s*\)",
        replacement=r"if (!\1)",
        description="if (ptr == nullptr) → if (!ptr)",
    ),

    # ── 34. while(1) → while(true) ───────────────────────────────────────
    ModernizationRule(
        pattern=r"\bwhile\s*\(\s*1\s*\)",
        replacement="while (true)",
        description="while(1) → while(true)",
    ),
    ModernizationRule(
        pattern=r"\bfor\s*\(\s*;\s*1\s*;\s*\)",
        replacement="for (;;)  // infinite loop",
        description="for(;1;) → for(;;)",
    ),

    # ── 35. Naked new[] → std::vector ────────────────────────────────────
    # int* arr = new int[10]; → std::vector<int> arr(10);
    ModernizationRule(
        pattern=r"\b(int|double|float|char|long|unsigned)\s*\*\s*(\w+)\s*=\s*new\s+\1\s*\[([^\]]+)\]\s*;",
        replacement=r"std::vector<\1> \2(\3);",
        description="T* = new T[n] → std::vector<T>(n)",
        hint_only=True,
    ),

    # ── 36. delete[] → (removed when array replaced by vector) ───────────
    ModernizationRule(
        pattern=r"\bdelete\s*\[\s*\]\s*(\w+)\s*;",
        replacement=r"// \1 is now a std::vector — delete[] removed",
        description="delete[] var → comment (vector manages lifetime)",
        hint_only=True,
    ),
]


# ---------------------------------------------------------------------------
# Smart header injection
# ---------------------------------------------------------------------------

_HEADER_REQUIREMENTS: Dict[str, List[str]] = {
    "memory":          [r"\bstd::unique_ptr\b", r"\bstd::shared_ptr\b",
                        r"\bstd::make_unique\b", r"\bstd::make_shared\b",
                        r"\bstd::weak_ptr\b"],
    "vector":          [r"\bstd::vector\b"],
    "string":          [r"\bstd::string\b"],
    "string_view":     [r"\bstd::string_view\b"],
    "array":           [r"\bstd::array\b"],
    "optional":        [r"\bstd::optional\b", r"\bstd::nullopt\b"],
    "variant":         [r"\bstd::variant\b"],
    "any":             [r"\bstd::any\b"],
    "span":            [r"\bstd::span\b"],
    "algorithm":       [r"\bstd::sort\b", r"\bstd::find\b", r"\bstd::min\b",
                        r"\bstd::max\b", r"\bstd::copy\b", r"\bstd::swap\b",
                        r"\bstd::remove_if\b", r"\bstd::find_if\b",
                        r"\bstd::transform\b", r"\bstd::accumulate\b",
                        r"\bstd::for_each\b"],
    "numeric":         [r"\bstd::accumulate\b", r"\bstd::iota\b"],
    "functional":      [r"\bstd::function\b", r"\bstd::bind\b"],
    "chrono":          [r"\bstd::chrono\b"],
    "map":             [r"\bstd::map\b"],
    "unordered_map":   [r"\bstd::unordered_map\b"],
    "set":             [r"\bstd::set\b"],
    "unordered_set":   [r"\bstd::unordered_set\b"],
    "tuple":           [r"\bstd::tuple\b", r"\bstd::make_tuple\b"],
    "utility":         [r"\bstd::pair\b", r"\bstd::make_pair\b",
                        r"\bstd::move\b", r"\bstd::forward\b"],
    "iostream":        [r"\bstd::cout\b", r"\bstd::cin\b", r"\bstd::cerr\b",
                        r"\bstd::endl\b"],
    "sstream":         [r"\bstd::stringstream\b", r"\bstd::ostringstream\b",
                        r"\bstd::istringstream\b"],
    "limits":          [r"\bstd::numeric_limits\b"],
    "type_traits":     [r"\bstd::is_same\b", r"\bstd::enable_if\b",
                        r"\bstd::decay\b", r"\bstd::remove_const\b"],
    "mutex":           [r"\bstd::mutex\b", r"\bstd::lock_guard\b",
                        r"\bstd::unique_lock\b"],
    "thread":          [r"\bstd::thread\b"],
    "atomic":          [r"\bstd::atomic\b"],
}


def _ensure_headers(code: str) -> str:
    """Inject missing #include directives for any std symbols used."""
    masked = _mask_comments_and_strings(code)
    missing: List[str] = []

    for header, patterns in _HEADER_REQUIREMENTS.items():
        include_re = re.compile(
            rf"#\s*include\s*<\s*{re.escape(header)}\s*>", re.IGNORECASE
        )
        if include_re.search(code):
            continue
        for pat in patterns:
            if re.search(pat, masked):
                missing.append(header)
                break

    if not missing:
        return code

    lines = code.splitlines(keepends=True)
    last_include_idx = -1
    for idx, line in enumerate(lines):
        if re.match(r"^\s*#\s*include\b", line):
            last_include_idx = idx

    insert_idx = last_include_idx + 1 if last_include_idx != -1 else 0
    new_includes = "".join(f"#include <{h}>\n" for h in sorted(missing))

    if insert_idx > 0 and not lines[insert_idx - 1].endswith("\n"):
        lines[insert_idx - 1] += "\n"

    lines.insert(insert_idx, new_includes)
    return "".join(lines)


def _update_sibling_includes(code: str, file_path: Optional[str]) -> str:
    """Rewrite local #include "X.h" to #include "X_modernized.h" for
    sibling files that have already been modernized (multi-file support)."""
    if not file_path:
        return code

    from pathlib import Path
    try:
        parent = Path(file_path).parent
        inc_re = re.compile(r'#\s*include\s*"([^"]+)"')

        def _replace(m: re.Match[str]) -> str:
            inc = m.group(1)
            inc_path = Path(inc)
            if inc_path.stem.endswith("_modernized"):
                return m.group(0)
            modernized = parent / f"{inc_path.stem}_modernized{inc_path.suffix}"
            if modernized.exists():
                return f'#include "{inc_path.stem}_modernized{inc_path.suffix}"'
            return m.group(0)

        return inc_re.sub(_replace, code)
    except Exception:
        return code


# ---------------------------------------------------------------------------
# Complexity scorer — decides if LLM is worth calling
# ---------------------------------------------------------------------------

_COMPLEX_PATTERNS = [
    r"\bmalloc\b",          # remaining mallocs (not simple sizeof pattern)
    r"\bfree\s*\(",         # manual memory dealloc
    r"\bnew\s+\w",          # remaining raw new
    r"\bdelete\s+",         # raw delete
    r"char\s*\*",           # raw char pointer
    r"\bstruct\s+\w+\s*{",  # struct that might become class
    r"\bsprintf\b",         # format string complexity
    r"(?<!\w)realloc\b",    # realloc
    r"\bva_list\b",         # variadic args
]

_COMPLEX_RE = [re.compile(p) for p in _COMPLEX_PATTERNS]


def complexity_score(code: str) -> int:
    """Count how many complex legacy patterns remain after rule application.
    0 = fully handled by rules. High = LLM useful."""
    masked = _mask_comments_and_strings(code)
    return sum(len(r.findall(masked)) for r in _COMPLEX_RE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_modernization_rules(
    code: str,
    detected_patterns: Optional[Dict[str, int]] = None,
) -> Tuple[str, List[str]]:
    """Apply all deterministic modernization rules.
    Returns (transformed_code, list_of_applied_descriptions)."""
    updated = code
    applied: List[str] = []
    active = detected_patterns or {}

    for rule in _RULES:
        if rule.hint_only:
            # Count matches but don't transform
            masked = _mask_comments_and_strings(updated)
            n = sum(1 for _ in rule.pattern.finditer(masked))
            if n:
                applied.append(f"[HINT] {rule.description} ({n} occurrences)")
            continue

        if rule.ast_triggers:
            if active:
                if not any(int(active.get(t, 0) or 0) > 0 for t in rule.ast_triggers):
                    continue
            else:
                masked = _mask_comments_and_strings(updated)
                n = sum(1 for _ in rule.pattern.finditer(masked))
                if n:
                    applied.append(f"[HINT] {rule.description} ({n} occurrences - requires AST)")
                continue

        try:
            updated, n = _apply_rule(updated, rule)
            if n:
                applied.append(f"{rule.description} ({n}×)")
                logger.info("[RuleModernizer] Applied '%s' (%d×)", rule.description, n)
        except Exception as exc:
            logger.warning("[RuleModernizer] Rule '%s' failed: %s", rule.description, exc)

    return updated, applied


class RuleModernizer:
    """Drop-in replacement for the old RuleModernizer.
    Also exposes `needs_llm()` to let callers skip the LLM for simple files."""

    def __init__(self) -> None:
        self._ast = None
        try:
            from .ast_modernizer import ASTModernizer
            inst = ASTModernizer()
            if inst.available:
                self._ast = inst
        except Exception:
            pass

    def needs_llm(self, code_after_rules: str, threshold: int = 3) -> bool:
        """Return True if the code still has complex patterns needing LLM help."""
        return complexity_score(code_after_rules) >= threshold

    def modernize_text(self, text: str, file_path: Optional[str] = None, cpp_standard: Optional[str] = None) -> str:
        """Apply rule-based modernization, update sibling includes, inject headers."""
        if self._ast is not None:
            try:
                updated = self._ast.modernize_text(text, cpp_standard=cpp_standard)
            except Exception:
                updated, _ = apply_modernization_rules(text)
        else:
            updated, _ = apply_modernization_rules(text)

        updated = _update_sibling_includes(updated, file_path)
        updated = _ensure_headers(updated)
        return updated

    def modernize_with_report(
        self,
        text: str,
        file_path: Optional[str] = None,
        detected_patterns: Optional[Dict[str, int]] = None,
        cpp_standard: Optional[str] = None,
    ) -> Tuple[str, List[str], bool]:
        """Modernize and return (code, applied_rules, needs_llm).
        Callers can use needs_llm to decide whether to invoke the LLM."""
        if self._ast is not None:
            try:
                updated = self._ast.modernize_text(text, cpp_standard=cpp_standard)
                applied = ["AST-based modernization applied"]
            except Exception:
                updated, applied = apply_modernization_rules(text, detected_patterns)
        else:
            updated, applied = apply_modernization_rules(text, detected_patterns)

        updated = _update_sibling_includes(updated, file_path)
        updated = _ensure_headers(updated)
        llm_needed = self.needs_llm(updated)
        return updated, applied, llm_needed
