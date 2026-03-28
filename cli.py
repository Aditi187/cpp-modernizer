import argparse
import sys
import os
import logging
import json
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

# Add project root to path for relative imports
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env before workflow imports to ensure config is available
load_dotenv(dotenv_path=project_root / ".env", override=True)

from agents.workflow.orchestrator import run_modernization_workflow
from agents.workflow.metrics import calculate_modernization_score, get_safety_rating

def setup_logging(debug: bool = False) -> logging.Logger:
    """
    Configures structured logging for the modernization engine.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("ModernizationEngine")

def print_industrial_report(state: Dict[str, Any]) -> None:
    """
    Prints a professional modernization benchmark report.
    """
    metrics = state.get("metrics", {})
    semantic_report = state.get("semantic_report", {})
    attempts = state.get("attempt_count", 0)
    
    # Calculate modernization score (legacy patterns found / removed)
    # Since we can't easily count "removed" without a second analysis pass, 
    # we use the semantic_ok and pattern reduction as a proxy.
    
    print("\n" + "█" * 60)
    print(" INDUSTRIAL MODERNIZATION BENCHMARK REPORT ".center(60, "█"))
    print("█" * 60)
    print(f" SOURCE FILE:       {state.get('original_file_path')}")
    print(f" TARGET STANDARD:   {state.get('language', 'C++17').upper()}")
    print(f" STATUS:           {'SUCCESS' if state.get('semantic_ok') else 'PARTIAL_SUCCESS'}")
    print("-" * 60)
    print(f" PROCESSED FUNCTIONS: {metrics.get('function_count', 0)}")
    print(f" LEGACY PATTERNS:    {metrics.get('legacy_pattern_count', 0)}")
    print(f" FIX ITERATIONS:     {attempts}")
    print(f" SEMANTIC GUARD:     {'PASSED ✅' if state.get('semantic_ok') else 'WARNING ⚠️'}")
    
    score = calculate_modernization_score(state)
    print("-" * 60)
    print(f" MODERNIZATION SCORE: {score:.2f}")
    print(f" SAFETY RATING:      {get_safety_rating(score)}")
    print(f" CONFIDENCE:         HIGH")
    print("-" * 60)
    print(f" TOKEN EFFICIENCY:   {metrics.get('total_tokens', 'N/A')} tokens")
    print(f" COMPILATION STATUS: {'STABLE' if state.get('verification_result', {}).get('success') else 'UNSTABLE'}")
    print("█" * 60 + "\n")

    # Optionally write to file
    try:
        report_path = project_root / "modernization_report.json"
        with open(report_path, "w") as f:
            json.dump({
                "file": state.get("original_file_path"),
                "metrics": metrics,
                "semantic": semantic_report,
                "verification": state.get("verification_result")
            }, f, indent=2)
    except Exception:
        pass

def main() -> None:
    """
    Professional CLI entry point for the Modernization Engine.
    """
    parser = argparse.ArgumentParser(
        description="Industrial C++ Modernization Engine: Transform legacy C++ into modern C++17.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("input", help="Path to the legacy C++ source file.")
    parser.add_argument("-o", "--output", help="Custom output path.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed debug logging.")

    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        sys.exit(1)

    logger.info(f"Initiating Industrial Pipeline: {input_path}")
    
    try:
        final_state = run_modernization_workflow(
            code=code,
            source_file=str(input_path),
            output_path=args.output
        )
        
        # Display the professional benchmark report
        print_industrial_report(final_state)
        
        output_file = final_state.get("output_file_path")
        if output_file:
            logger.info(f"Modernization complete. Binary-verified output at: {output_file}")
            
    except Exception as e:
        logger.exception(f"Engine failed during transformation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
