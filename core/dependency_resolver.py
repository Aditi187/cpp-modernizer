import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# Matches `#include "file.h"` or `#include <file.h>`
_INCLUDE_RE = re.compile(r'#\s*include\s*["<]([^">]+)[">]')


def _extract_includes(file_path: Path) -> List[str]:
    """Read a file and extract all include targets, ignoring comments."""
    includes = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Simple comment masking to prevent parsing commented includes
        content_clean = re.sub(r"//[^\n]*|/\*.*?\*/", "", content, flags=re.DOTALL)

        for line in content_clean.splitlines():
            line = line.strip()
            if line.startswith("#"):
                match = _INCLUDE_RE.match(line)
                if match:
                    includes.append(match.group(1))
    except Exception as e:
        logger.error(f"Error reading includes from {file_path}: {e}")
    return includes


import json


class CompileCommandsReader:
    """
    Parses compile_commands.json to extract file-specific include paths.
    """

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.file_to_includes: Dict[Path, List[Path]] = {}
        self._load()

    def _load(self) -> None:
        candidates = [
            self.workspace_root / "compile_commands.json",
            self.workspace_root / "build" / "compile_commands.json",
            Path.cwd() / "compile_commands.json",
            Path.cwd() / "build" / "compile_commands.json",
        ]
        db_file = None
        for c in candidates:
            if c.is_file():
                db_file = c
                break
        if not db_file:
            return
        try:
            with open(db_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                file_val = entry.get("file")
                if not file_val:
                    continue
                base_dir = Path(entry.get("directory", ""))
                file_path = Path(file_val)
                if not file_path.is_absolute() and base_dir:
                    file_path = (base_dir / file_path).resolve()
                else:
                    file_path = file_path.resolve()

                args = entry.get("arguments") or entry.get("command")
                if isinstance(args, str):
                    args = args.split()
                if not isinstance(args, list):
                    continue

                inc_dirs = []
                i = 0
                while i < len(args):
                    arg = args[i]
                    if arg == "-I" and i + 1 < len(args):
                        inc_path = Path(args[i+1])
                        if not inc_path.is_absolute() and base_dir:
                            inc_path = (base_dir / inc_path).resolve()
                        else:
                            inc_path = inc_path.resolve()
                        inc_dirs.append(inc_path)
                        i += 2
                    elif arg.startswith("-I"):
                        path_str = arg[2:].strip()
                        if path_str:
                            inc_path = Path(path_str)
                            if not inc_path.is_absolute() and base_dir:
                                inc_path = (base_dir / inc_path).resolve()
                            else:
                                inc_path = inc_path.resolve()
                            inc_dirs.append(inc_path)
                        i += 1
                    elif arg == "-isystem" and i + 1 < len(args):
                        inc_path = Path(args[i+1])
                        if not inc_path.is_absolute() and base_dir:
                            inc_path = (base_dir / inc_path).resolve()
                        else:
                            inc_path = inc_path.resolve()
                        inc_dirs.append(inc_path)
                        i += 2
                    else:
                        i += 1
                self.file_to_includes[file_path] = inc_dirs
        except Exception as e:
            logger.warning(f"Error parsing compile_commands.json: {e}")

    def get_include_dirs(self, file_path: Path) -> List[Path]:
        return self.file_to_includes.get(file_path.resolve(), [])


def resolve_dependencies(
    files: List[Path], workspace_root: Path
) -> Tuple[List[Path], Dict[Path, Set[Path]]]:
    """Build a dependency graph and return the topologically sorted files.

    Returns:
        - Sorted list of files (from leaves/independent files to root dependencies)
        - Dependency mapping (node -> set of direct dependencies within the target list)
    """
    file_set = {f.resolve() for f in files}
    file_map = {f.name: f.resolve() for f in files}  # Name lookup for local header matching

    # Load compile commands include directories
    commands_reader = CompileCommandsReader(workspace_root)

    # Map: Node -> Set of files it directly depends on
    dependencies: Dict[Path, Set[Path]] = {f: set() for f in file_set}

    for f in file_set:
        extracted = _extract_includes(f)
        compile_dirs = commands_reader.get_include_dirs(f)
        for inc in extracted:
            inc_path = Path(inc)
            resolved = None

            # 1. Resolve relative to compile commands include dirs
            for d in compile_dirs:
                candidate = (d / inc_path).resolve()
                if candidate in file_set:
                    resolved = candidate
                    break

            # 2. Resolve relative to the current file's parent
            if not resolved:
                candidate1 = (f.parent / inc_path).resolve()
                if candidate1 in file_set:
                    resolved = candidate1

            # 3. Resolve relative to the workspace root
            if not resolved:
                candidate2 = (workspace_root / inc_path).resolve()
                if candidate2 in file_set:
                    resolved = candidate2

            # 4. Resolve using simple name match
            if not resolved:
                candidate3 = file_map.get(inc_path.name)
                if candidate3 in file_set:
                    resolved = candidate3

            if resolved and resolved != f:
                dependencies[f].add(resolved)

    # Topological sort (Kahn's algorithm or DFS detection)
    # Map: Node -> Set of incoming edges (nodes depending on it)
    dependents: Dict[Path, Set[Path]] = {f: set() for f in file_set}
    in_degree = {f: 0 for f in file_set}

    for u, deps in dependencies.items():
        for v in deps:
            dependents[v].add(u)
            in_degree[u] += 1

    # Queue of files with zero in-degree (independent files/leaves)
    queue = [u for u, deg in in_degree.items() if deg == 0]
    sorted_files = []

    # Sort queue initially to ensure deterministic ordering
    queue.sort(key=lambda p: p.name)

    while queue:
        u = queue.pop(0)
        sorted_files.append(u)
        for v in dependents[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
        # Re-sort queue to maintain deterministic order
        queue.sort(key=lambda p: p.name)

    # Detect cycles
    if len(sorted_files) < len(file_set):
        unresolved = file_set - set(sorted_files)
        logger.warning(
            f"Circular dependency detected among files: {[f.name for f in unresolved]}. "
            "Breaking dependency chain to process remaining files."
        )
        # Append unresolved files arbitrarily to ensure no file is lost
        sorted_files.extend(sorted(list(unresolved), key=lambda p: p.name))

    return sorted_files, dependencies
