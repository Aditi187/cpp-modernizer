from __future__ import annotations

import argparse
import os

from agents.function_modernizer import FunctionModernizer
from core.openai_bridge import OpenAIBridge
from core.logger import get_logger
from core.parser import CppParser

logger = get_logger(__name__)


def _select_bridge():
    return OpenAIBridge.from_env(log_fn=logger.info)


def _cmd_modernize(args: argparse.Namespace) -> int:
    if args.enable_rag is not None:
        os.environ["ENABLE_RAG"] = "1" if args.enable_rag else "0"
    if args.reflection_iters is not None:
        os.environ["REFLECTION_MAX_ITERS"] = str(max(0, int(args.reflection_iters)))

    parser = CppParser()
    bridge = _select_bridge()
    modernizer = FunctionModernizer(parser=parser, llm=bridge)

    modernizer.modernize_file(args.file)
    modernizer.print_report()
    logger.info("Modernization completed for %s", args.file)
    return 0


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Air-Gapped modernization CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    modernize = subparsers.add_parser("modernize", help="Modernize one C++ source file in place")
    modernize.add_argument("--file", required=True, help="Path to C++ file")
    modernize.add_argument(
        "--enable-rag",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable retrieval augmented prompts",
    )
    modernize.add_argument(
        "--reflection-iters",
        type=int,
        default=None,
        help="Max reflection improvement iterations",
    )
    modernize.set_defaults(handler=_cmd_modernize)

    return parser


def main() -> int:
    parser = build_cli_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
