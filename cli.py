import warnings
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality",
    category=UserWarning,
)

import argparse
import sys
import os
import logging
import difflib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

from dotenv import load_dotenv

# Add project root to path for relative imports
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env before workflow imports to ensure config is available
load_dotenv(dotenv_path=project_root / ".env", override=False)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TaskProgressColumn,
    TimeElapsedColumn, TimeRemainingColumn, TextColumn, MofNCompleteColumn
)
from rich import box

from agents.workflow.orchestrator import run_modernization_workflow
from agents.workflow.metrics import calculate_modernization_score, get_safety_rating
from core.dependency_resolver import resolve_dependencies
from core.project_state import ProjectStateDB

console = Console()

import importlib.metadata
try:
    VERSION = importlib.metadata.version("air-gapped-codebase-modernization-engine")
except importlib.metadata.PackageNotFoundError:
    VERSION = "0.2.0"

_DB_PATH = str(project_root / ".modernization_state.db")


def setup_logging(debug: bool = False) -> logging.Logger:
    """Configures structured logging for the modernization engine."""
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    return logging.getLogger("ModernizationEngine")


def print_industrial_report(state: Dict[str, Any], source_file: str = "") -> None:
    """Prints a professional modernization benchmark report using Rich."""
    metrics = state.get("metrics", {})
    semantic_report = state.get("semantic_report", {})
    attempts = state.get("attempt_count", 0)
    score = calculate_modernization_score(state)
    safety = get_safety_rating(score)
    meta = state.get("pipeline_metadata", {})

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="white")

    status_color = "green" if state.get("semantic_ok") else "yellow"
    status_text = "SUCCESS" if state.get("semantic_ok") else "PARTIAL_SUCCESS"

    compile_ok = state.get("verification_result", {}).get("success", False)
    compiler_used = state.get("verification_result", {}).get("compiler", "skipped")
    compile_status = "[green]STABLE[/]" if compile_ok else "[dim]SKIPPED[/]"

    tokens = metrics.get("total_tokens", None)
    token_display = str(tokens) if tokens else "[dim]N/A[/]"

    llm_skipped = meta.get("llm_skipped", False)
    attribution = meta.get("attribution", "unknown")
    complexity = meta.get("complexity_score", "?")
    rules_applied = meta.get("rules_applied", [])

    table.add_row("Source File", str(source_file or state.get("original_file_path", "")))
    table.add_row("Target Standard", state.get("language", "C++17").upper())
    table.add_row("Status", f"[{status_color}]{status_text}[/]")
    table.add_row("Processed Functions", str(metrics.get("function_count", 0)))
    table.add_row("Legacy Patterns Fixed", str(metrics.get("legacy_pattern_count", 0)))
    table.add_row("Fix Iterations", str(attempts))
    table.add_row("Semantic Guard", "[green]PASSED[/]" if state.get("semantic_ok") else "[yellow]WARNING[/]")
    table.add_row("Modernization Score", f"[bold green]{score:.2f}[/]")
    table.add_row("Safety Rating", f"[bold]{safety}[/]")
    table.add_row("Complexity Score", str(complexity))
    table.add_row("LLM Called", "[yellow]YES[/]" if not llm_skipped else "[green]NO — rules only[/]")
    table.add_row("Attribution", attribution)
    table.add_row("Rules Applied", str(len(rules_applied)))
    table.add_row("Token Efficiency", token_display)
    table.add_row("Compilation Status", compile_status)

    console.print()
    console.print(Panel(table, title="[bold white]INDUSTRIAL MODERNIZATION BENCHMARK REPORT[/]",
                        border_style="bright_blue", padding=(1, 2)))

    # Before / After diff panel
    original = state.get("code", "")
    modernized = state.get("modernized_code", "")
    if original and modernized and original.strip() != modernized.strip():
        diff_lines = list(difflib.unified_diff(
            original.splitlines(),
            modernized.splitlines(),
            fromfile="legacy (before)",
            tofile="modernized (after)",
            lineterm="",
        ))
        shown = diff_lines[:60]
        if len(diff_lines) > 60:
            shown.append(f"... ({len(diff_lines) - 60} more lines)")

        diff_text = ""
        for line in shown:
            if line.startswith("+") and not line.startswith("+++"):
                diff_text += f"[green]{line}[/]\n"
            elif line.startswith("-") and not line.startswith("---"):
                diff_text += f"[red]{line}[/]\n"
            elif line.startswith("@@"):
                diff_text += f"[cyan]{line}[/]\n"
            else:
                diff_text += f"{line}\n"

        console.print(Panel(
            diff_text.rstrip(),
            title="[bold white]TRANSFORMATION DIFF (before -> after)[/]",
            border_style="green",
            padding=(1, 2),
        ))
    console.print()


def process_single_file(
    input_path: Path,
    output_path: Optional[str],
    skip_verify: bool,
    logger: logging.Logger,
    interactive: bool = False,
    dry_run: bool = False,
    db: Optional[ProjectStateDB] = None,
    run_id: Optional[int] = None,
) -> tuple[bool, dict]:
    """Modernize one file. Returns (success, audit_info)."""
    # Set env var for this invocation; WorkflowConfig.from_env() reads it
    # Note: for batch runs, this is process-wide, which is intentional
    if skip_verify:
        os.environ["SKIP_VERIFICATION"] = "1"
    else:
        # Ensure previous runs don't bleed into this one
        os.environ.pop("SKIP_VERIFICATION", None)

    t_start = time.monotonic()

    # ── Skip unchanged files (resume logic) ──────────────────────────────
    if db and db.is_up_to_date(str(input_path)):
        console.print(f"  [dim]>>  {input_path.name} - unchanged, skipping[/]")
        if db:
            db.mark_skipped(str(input_path))
        return True, {"skipped": True}

    if db:
        db.register_file(str(input_path), run_id)
        db.mark_running(str(input_path))

    if dry_run:
        # Dry-run: apply rules only, report what would change, don't write
        try:
            from core.rule_modernizer import RuleModernizer, complexity_score
            code = input_path.read_text(encoding="utf-8", errors="replace")
            rm = RuleModernizer()
            result, applied, needs_llm = rm.modernize_with_report(code)
            score = complexity_score(result)
            console.print(
                f"  [cyan]DRY-RUN[/] {input_path.name}: "
                f"{len(applied)} rule(s), complexity={score}, "
                f"LLM would be: {'[yellow]YES[/]' if needs_llm else '[green]NO[/]'}"
            )
            for r in applied:
                console.print(f"    [dim]• {r}[/]")
        except Exception as e:
            console.print(f"  [red]DRY-RUN ERROR[/] {input_path.name}: {e}")
        return True, {"dry_run": True}

    try:
        code = input_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error(f"Failed to read {input_path}: {e}")
        if db:
            db.mark_failed(str(input_path), str(e))
        return False, {"error": str(e)}

    developer_feedback = ""
    while True:
        try:
            write_to_disk = not interactive
            final_state = run_modernization_workflow(
                code=code,
                source_file=str(input_path),
                output_path=output_path,
                write_to_disk=write_to_disk,
                developer_feedback=developer_feedback,
                run_id=str(run_id) if run_id else ""
            )
            print_industrial_report(final_state, source_file=str(input_path))

            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            meta = final_state.get("pipeline_metadata", {})
            llm_called = not meta.get("llm_skipped", False)
            attribution = meta.get("attribution", "unknown")
            complexity = meta.get("complexity_score", 0)
            rules_applied = meta.get("rules_applied", [])

            if interactive:
                choice = ""
                while choice not in ("y", "yes", "n", "no", "r", "retry"):
                    console.print(
                        f"[bold yellow]Accept changes for {input_path.name}? "
                        f"[y]es / [n]o / [r]etry: [/]", end=""
                    )
                    try:
                        choice = input().strip().lower()
                    except EOFError:
                        choice = "n"

                if choice in ("y", "yes"):
                    out_file = output_path or final_state.get("output_file_path")
                    if not out_file:
                        p = Path(input_path)
                        out_file = str(p.parent / f"{p.stem}_modernized{p.suffix}")
                    modernized_code = final_state.get("modernized_code", "")
                    # Atomic write: temp → rename
                    tmp = Path(out_file).with_suffix(".tmp")
                    tmp.write_text(modernized_code, encoding="utf-8")
                    tmp.replace(Path(out_file))
                    if db:
                        db.mark_done(
                            str(input_path), out_file,
                            audit_entries=[{"rule": r, "attribution": attribution} for r in rules_applied],
                            complexity=complexity, llm_called=llm_called,
                            attribution=attribution, duration_ms=elapsed_ms,
                        )
                    return True, {"output": out_file, "llm_called": llm_called, "attribution": attribution}

                elif choice in ("n", "no"):
                    if db:
                        db.mark_failed(str(input_path), "user_rejected")
                    return False, {"rejected": True}

                elif choice in ("r", "retry"):
                    console.print("[bold cyan]Enter feedback for the LLM: [/]", end="")
                    try:
                        developer_feedback = input().strip()
                    except EOFError:
                        developer_feedback = ""
                    continue
            else:
                out = output_path or final_state.get("output_file_path")
                if db and out:
                    db.mark_done(
                        str(input_path), str(out),
                        audit_entries=[{"rule": r, "attribution": attribution} for r in rules_applied],
                        complexity=complexity, llm_called=llm_called,
                        attribution=attribution, duration_ms=elapsed_ms,
                    )
                return True, {"output": out, "llm_called": llm_called, "attribution": attribution}

        except Exception as e:
            logger.exception(f"Engine failed for {input_path}: {e}")
            if db:
                db.mark_failed(str(input_path), str(e))
            return False, {"error": str(e)}


def collect_cpp_files(directory: Path) -> List[Path]:
    """Recursively collect all C++ source files, excluding _modernized outputs."""
    files = []
    skip_dirs = {".venv", ".git", "build", "dist", "__pycache__", "mingw64", "mingw"}
    for pattern in ("**/*.cpp", "**/*.h", "**/*.cc", "**/*.cxx", "**/*.hpp"):
        for p in directory.glob(pattern):
            # Skip excluded directories
            if any(d in p.parts for d in skip_dirs):
                continue
            if "_modernized" not in p.name:
                files.append(p)
    return sorted(files)


def _print_project_summary(db: ProjectStateDB, run_id: int, elapsed: float) -> None:
    """Print final project-level summary table."""
    summary = db.get_run_summary(run_id)
    files = summary.get("files", [])

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("File", style="white", no_wrap=False)
    table.add_column("Status", justify="center")
    table.add_column("LLM", justify="center")
    table.add_column("Complexity", justify="right")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Attribution", style="dim")

    status_icons = {
        "done": "[green]✓ done[/]",
        "failed": "[red]✗ failed[/]",
        "skipped": "[dim]>> skipped[/]",
        "pending": "[yellow]… pending[/]",
    }

    for f in sorted(files, key=lambda x: x.get("path", "")):
        status = f.get("status", "?")
        llm = "[yellow]YES[/]" if f.get("llm_called") else "[green]NO[/]"
        table.add_row(
            Path(f["path"]).name if f.get("path") else "?",
            status_icons.get(status, status),
            llm,
            str(f.get("complexity", "")),
            str(f.get("duration_ms", "")),
            str(f.get("attribution", ""))[:40],
        )

    run = summary.get("run", {})
    console.print()
    console.print(table)
    console.print()
    console.print(Panel(
        f"[green]✓ Passed:[/]  {run.get('passed', 0)}    "
        f"[red]✗ Failed:[/]  {run.get('failed', 0)}    "
        f"[dim]>> Skipped:[/] {run.get('skipped', 0)}    "
        f"[cyan]LLM calls:[/] {run.get('llm_calls', 0)}    "
        f"[dim]Total time: {elapsed:.1f}s[/]",
        title="[bold white]PROJECT SUMMARY[/]",
        border_style="bright_blue",
    ))


def main() -> None:
    """Industry-grade CLI for the Air-Gapped C++ Modernization Engine."""
    parser = argparse.ArgumentParser(
        description=(
            "Air-Gapped C++ Modernization Engine v" + VERSION + "\n"
            "Transforms legacy C/C++ codebases into verified, idiomatic C++17.\n"
            "Deterministic-first: rules handle 40%+ of files in <1s. "
            "LLM only called for genuinely complex patterns."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", help="Path to a C++ source file OR directory to batch-process.")
    parser.add_argument("-o", "--output", help="Custom output path (single-file mode only).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable detailed debug logging.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip compiler verification.")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers for batch mode (default: 2).")
    parser.add_argument("-i", "--interactive", action="store_true", help="Human-in-the-loop review mode.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing files (instant, no LLM).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip files already successfully processed and unchanged.")
    parser.add_argument("--rules-only", action="store_true",
                        help="Disable LLM entirely. Rules-based transformation only (instant).")
    parser.add_argument("--cpp-standard", choices=["c++14", "c++17", "c++20", "c++23"], default="c++17",
                        help="Target C++ standard (default: c++17).")
    parser.add_argument("--audit", type=str, metavar="FILE",
                        help="Write structured audit log to FILE (JSONL format).")

    parser.add_argument("--stats", action="store_true",
                        help="Show project statistics from previous runs and exit.")
    parser.add_argument("--version", action="version", version=f"modernization-engine {VERSION}")

    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    # Propagate C++ standard via environment variable
    os.environ["CPP_STANDARD"] = args.cpp_standard

    input_path = Path(args.input)
    if not args.stats and not input_path.exists():
        console.print(f"[red]✗ Input not found:[/] {args.input}")
        sys.exit(1)

    # Apply rules-only mode
    if args.rules_only:
        os.environ["USE_LLM"] = "false"
        console.print("[dim]Rules-only mode active — LLM disabled[/]")

    # Project state DB
    db = ProjectStateDB(_DB_PATH)

    # ── Stats mode ────────────────────────────────────────────────────────
    if args.stats:
        stats = db.get_statistics()
        table = Table(title="Project Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        for k, v in stats.items():
            table.add_row(k.replace("_", " ").title(), str(round(v, 1) if isinstance(v, float) else v))
        console.print(table)
        sys.exit(0)

    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()

    # ── SINGLE FILE MODE ──────────────────────────────────────────────────
    if input_path.is_file():
        run_id = db.start_run(total_files=1)
        output_file = args.output or str(output_dir / f"{input_path.stem}_modernized{input_path.suffix}")
        ok, audit = process_single_file(
            input_path, output_file, args.skip_verify, logger,
            interactive=args.interactive, dry_run=args.dry_run,
            db=db, run_id=run_id
        )
        db.finish_run(run_id)
        if args.audit:
            db.export_audit_jsonl(run_id, args.audit)
            console.print(f"[dim]Audit log written to {args.audit}[/]")

        sys.exit(0 if ok else 1)

    # ── BATCH DIRECTORY MODE ──────────────────────────────────────────────
    if not input_path.is_dir():
        console.print(f"[red]✗ Input must be a file or directory:[/] {args.input}")
        sys.exit(1)

    cpp_files = collect_cpp_files(input_path)
    if not cpp_files:
        console.print(f"[yellow]No C++ files found in {input_path}[/]")
        sys.exit(0)

    # Count how many can be skipped (resume mode)
    resumable = 0
    if args.resume:
        resumable = sum(1 for f in cpp_files if db.is_up_to_date(str(f)))
        console.print(
            f"[dim]Resume mode: {resumable}/{len(cpp_files)} files unchanged — will skip[/]"
        )

    # Resolve DAG dependency order
    sorted_files, deps = resolve_dependencies(cpp_files, workspace_root=input_path)

    run_id = db.start_run(
        total_files=len(sorted_files),
        config={"workers": args.workers, "skip_verify": args.skip_verify,
                "rules_only": args.rules_only, "dry_run": args.dry_run,
                "cpp_standard": args.cpp_standard}
    )

    console.print(Panel(
        f"[bold]Processing {len(sorted_files)} files[/] "
        f"([cyan]{len(cpp_files)}[/] discovered, [dim]DAG-ordered[/])\n"
        f"Workers: {args.workers} | "
        f"LLM: {'[red]DISABLED[/]' if args.rules_only else '[green]enabled[/]'} | "
        f"Standard: [cyan]{args.cpp_standard.upper()}[/] | "
        f"Resume: {'[green]on[/]' if args.resume else '[dim]off[/]'}",
        title="[bold white]AIR-GAPPED C++ MODERNIZATION ENGINE[/]",
        border_style="bright_blue",
    ))

    completed: Set[Path] = set()
    pending: Set[Path] = set(sorted_files)
    running_set: Set[Path] = set()
    results: Dict[str, bool] = {}

    def _process(p: Path) -> tuple:
        out = str(output_dir / f"{p.stem}_modernized{p.suffix}")
        ok, audit_info = process_single_file(
            p, out, args.skip_verify, logger,
            interactive=args.interactive, dry_run=args.dry_run,
            db=db, run_id=run_id
        )
        return str(p), ok, audit_info

    workers = 1 if args.interactive else args.workers
    import concurrent.futures

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("[cyan]Modernizing...", total=len(sorted_files))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures_map = {}

            while pending or running_set:
                # Schedule all ready files (deps satisfied)
                ready = [
                    f for f in sorted(pending, key=lambda x: x.name)
                    if deps.get(f, set()).issubset(completed) and f not in running_set
                ]
                # Break deadlock
                if not ready and not running_set and pending:
                    ready.append(sorted(pending, key=lambda x: x.name)[0])

                for f in ready:
                    pending.discard(f)
                    running_set.add(f)
                    future = executor.submit(_process, f)
                    futures_map[future] = f

                if futures_map:
                    done_futures, _ = concurrent.futures.wait(
                        futures_map.keys(),
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for fut in done_futures:
                        f = futures_map.pop(fut)
                        running_set.discard(f)
                        try:
                            path_str, ok, audit_info = fut.result()
                            results[path_str] = ok
                            completed.add(Path(path_str))
                            skipped = audit_info.get("skipped") or audit_info.get("dry_run")
                            icon = ">>" if skipped else ("✓" if ok else "✗")
                            color = "dim" if skipped else ("green" if ok else "red")
                            progress.console.print(
                                f"  [{color}]{icon}[/] [white]{Path(path_str).name}[/] "
                                f"[dim]{'skipped' if skipped else ('LLM' if audit_info.get('llm_called') else 'rules-only')}[/]"
                            )
                        except Exception as e:
                            logger.error(f"Error processing {f}: {e}")
                            results[str(f)] = False
                            completed.add(f)
                            db.mark_failed(str(f), str(e))
                        progress.advance(task)

    db.finish_run(run_id)
    elapsed = time.monotonic() - t_start

    # Write audit log if requested
    if args.audit:
        db.export_audit_jsonl(run_id, args.audit)
        console.print(f"[dim]Audit log written to {args.audit}[/]")



    _print_project_summary(db, run_id, elapsed)

    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    sys.exit(0 if failed == 0 else 1)



if __name__ == "__main__":
    main()
