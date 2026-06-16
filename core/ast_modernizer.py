import os
import re

try:
    from clang import cindex
except ImportError:
    cindex = None


class ASTModernizer:
    """Lightweight AST‑based modernizer.

    Currently supports:
    * Replace `NULL` with `nullptr`.
    * Convert simple `typedef` declarations to `using`.
    * Transform C‑style `#define` constants to `constexpr auto` (skips include
      guards and functional macros).
    * Map old C headers to their C++ counterparts (18 headers).
    """

    _HEADER_MAP = {
        # Standard C → C++ header mappings
        "<assert.h>": "<cassert>",
        "<ctype.h>": "<cctype>",
        "<errno.h>": "<cerrno>",
        "<float.h>": "<cfloat>",
        "<limits.h>": "<climits>",
        "<locale.h>": "<clocale>",
        "<math.h>": "<cmath>",
        "<setjmp.h>": "<csetjmp>",
        "<signal.h>": "<csignal>",
        "<stdarg.h>": "<cstdarg>",
        "<stddef.h>": "<cstddef>",
        "<stdio.h>": "<cstdio>",
        "<stdlib.h>": "<cstdlib>",
        "<string.h>": "<cstring>",
        "<time.h>": "<ctime>",
        "<wchar.h>": "<cwchar>",
        "<wctype.h>": "<cwctype>",
        "<stdint.h>": "<cstdint>",
    }

    # Pattern to detect include-guard style names (all caps, underscores, ends _H or _HPP)
    _INCLUDE_GUARD_RE = re.compile(
        r"^[A-Z][A-Z0-9_]*(?:_H|_HPP|_INCLUDED|_GUARD)$"
    )

    def __init__(self):
        self._index = None
        self._system_includes = []
        if cindex is not None:
            try:
                self._index = cindex.Index.create()
                self._system_includes = self._discover_system_include_paths()
            except Exception:
                self._index = None

    @classmethod
    def _discover_system_include_paths(cls) -> list[str]:
        """Query host compiler (g++/clang++) for default system include paths."""
        import subprocess
        paths = []
        compilers = []
        env_compiler = os.environ.get("COMPILER_PATH")
        if env_compiler:
            compilers.append(env_compiler)
        compilers.extend(["g++", "clang++", "gpp"])

        for compiler in compilers:
            try:
                proc = subprocess.run(
                    [compiler, "-E", "-x", "c++", "-v", "-"],
                    input="",
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                stderr = proc.stderr
                started = False
                for line in stderr.splitlines():
                    line_strip = line.strip()
                    if "#include <...> search starts here:" in line_strip:
                        started = True
                        continue
                    if "End of search list." in line_strip:
                        started = False
                        break
                    if started and line_strip:
                        path_dir = os.path.abspath(line_strip)
                        if os.path.isdir(path_dir):
                            paths.append(path_dir)
                if paths:
                    break
            except Exception:
                continue
        return paths

    @property
    def available(self) -> bool:
        """True if clang bindings are functional."""
        return self._index is not None

    def _apply_nullptr(self, code: str, tokens) -> str:
        edits = []
        for token in tokens:
            if token.spelling == "NULL" and token.kind == cindex.TokenKind.IDENTIFIER:
                edits.append((token.extent.start.offset, token.extent.end.offset, "nullptr"))
        return self._apply_edits(code, edits)

    def _apply_typedef(self, code: str, tokens) -> str:
        # Look for pattern: typedef <type...> <identifier> ;
        edits = []
        i = 0
        while i < len(tokens):
            if tokens[i].spelling == "typedef":
                # collect tokens until ';'
                j = i + 1
                while j < len(tokens) and tokens[j].spelling != ";":
                    j += 1
                if j < len(tokens):
                    # tokens[i+1:j] contains type and name
                    # Find last identifier before ';'
                    name_token = None
                    type_start = tokens[i + 1].extent.start.offset
                    for k in range(i + 1, j):
                        if tokens[k].kind == cindex.TokenKind.IDENTIFIER:
                            name_token = tokens[k]
                    if name_token:
                        type_end = name_token.extent.start.offset
                        type_str = code[type_start:type_end].strip()
                        name = name_token.spelling
                        replacement = f"using {name} = {type_str};"
                        start = tokens[i].extent.start.offset
                        end = tokens[j].extent.end.offset
                        edits.append((start, end, replacement))
                i = j
            i += 1
        return self._apply_edits(code, edits)

    @classmethod
    def _is_include_guard_name(cls, name: str) -> bool:
        """Return True if *name* looks like a header include guard."""
        return bool(cls._INCLUDE_GUARD_RE.match(name))

    def _apply_define(self, code: str, tokens) -> str:
        """Convert simple #define NAME VALUE → constexpr auto NAME = VALUE;

        Safety: Skips include guards and functional macros (those with parentheses
        immediately after the name, e.g. ``#define MAX(a,b)``).
        """
        edits = []
        i = 0
        while i < len(tokens) - 2:
            if tokens[i].spelling == "#" and tokens[i + 1].spelling == "define":
                name_tok = tokens[i + 2]
                name = name_tok.spelling

                # Skip include guards (e.g. MY_HEADER_H)
                if self._is_include_guard_name(name):
                    i += 3
                    continue

                # Capture everything up to end of line as value
                line_end = code.find('\n', name_tok.extent.end.offset)
                if line_end == -1:
                    line_end = len(code)
                value = code[name_tok.extent.end.offset:line_end].strip()

                # Skip functional macros: value starts with '(' immediately
                # (e.g. #define MAX(a,b) ...)
                if value.startswith("("):
                    i += 3
                    continue

                # Skip empty defines (guards like #define FOO)
                if not value:
                    i += 3
                    continue

                # Skip values that are not simple constants (contain commas, complex expressions)
                # Only convert numeric literals, string literals, or simple identifiers
                if not re.match(r'^(?:[0-9]+(?:\.[0-9]+)?[fFlLuU]*|0[xX][0-9a-fA-F]+[uUlL]*|"[^"]*"|[A-Za-z_]\w*)$', value):
                    i += 3
                    continue

                replacement = f"constexpr auto {name} = {value};"
                start = tokens[i].extent.start.offset
                end = line_end
                edits.append((start, end, replacement))
                i = i + 3
            else:
                i += 1
        return self._apply_edits(code, edits)

    def _apply_includes(self, code: str, tokens) -> str:
        edits = []
        i = 0
        while i < len(tokens) - 2:
            if tokens[i].spelling == "#" and tokens[i + 1].spelling == "include":
                header_tok = tokens[i + 2]
                hdr = header_tok.spelling
                new_hdr = self._HEADER_MAP.get(hdr)
                if new_hdr:
                    start = header_tok.extent.start.offset
                    end = header_tok.extent.end.offset
                    edits.append((start, end, new_hdr))
                i += 3
            else:
                i += 1
        return self._apply_edits(code, edits)

    @staticmethod
    def _apply_edits(code: str, edits) -> str:
        if not edits:
            return code
        # Apply in reverse order to keep offsets valid
        new_code = code
        for start, end, repl in sorted(edits, reverse=True):
            new_code = new_code[:start] + repl + new_code[end:]
        return new_code

    def modernize_text(self, code: str, cpp_standard: Optional[str] = None) -> str:
        """Apply all AST‑based transformations.

        If clang is unavailable the original source is returned unchanged.
        """
        if not self.available:
            return code
        # Parse the temporary file to obtain tokens
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(delete=False, suffix=".cpp", mode="w", encoding="utf-8") as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        try:
            std_flag = f"-std={cpp_standard}" if cpp_standard else "-std=c++23"
            parse_args = [std_flag]
            for p in self._system_includes:
                parse_args.extend(["-isystem", p])
            tu = self._index.parse(tmp_path, args=parse_args, options=0)
            tokens = list(tu.get_tokens(extent=tu.cursor.extent))
            new_code = self._apply_nullptr(code, tokens)
            new_code = self._apply_typedef(new_code, tokens)
            new_code = self._apply_define(new_code, tokens)
            new_code = self._apply_includes(new_code, tokens)
            return new_code
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
