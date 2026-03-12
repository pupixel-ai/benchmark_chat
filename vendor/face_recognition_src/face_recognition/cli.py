from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .config import PipelineConfig
from .pipeline import PipelineRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="face-recognition")
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import", help="Import and cluster a directory of photos")
    import_parser.add_argument("--input-dir", required=True, help="Directory containing photos")
    import_parser.add_argument("--db-path", required=True, help="Path to SQLite metadata.db")
    import_parser.add_argument("--index-path", required=True, help="Path to FAISS index file")
    import_parser.add_argument("--log-dir", help="Optional log directory")
    import_parser.add_argument("--batch-size", type=int, default=100, help="Flush threshold")
    import_parser.add_argument(
        "--batch-retry-limit",
        type=int,
        default=2,
        help="Retry count for transient batch persistence failures",
    )
    import_parser.add_argument(
        "--batch-retry-backoff-seconds",
        type=float,
        default=1.0,
        help="Base backoff in seconds between batch retries",
    )
    import_parser.add_argument("--max-side", type=int, default=1920, help="Max image dimension")
    import_parser.add_argument(
        "--det-threshold",
        type=float,
        default=0.60,
        help="Face detection confidence threshold",
    )
    import_parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.50,
        help="Cosine similarity threshold for clustering",
    )
    import_parser.add_argument(
        "--providers",
        nargs="+",
        default=("CPUExecutionProvider",),
        help="ONNX Runtime providers in priority order",
    )
    import_parser.add_argument(
        "--preflight-validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable lightweight preflight checks such as zero-byte file detection",
    )
    import_parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing SQLite/index state when present",
    )
    import_parser.add_argument(
        "--model-name",
        default="buffalo_l",
        help="InsightFace model pack name",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "import":
        parser.error("unsupported command")

    config = PipelineConfig.from_args(
        input_dir=args.input_dir,
        db_path=args.db_path,
        index_path=args.index_path,
        log_dir=args.log_dir,
        batch_size=args.batch_size,
        batch_retry_limit=args.batch_retry_limit,
        batch_retry_backoff_seconds=args.batch_retry_backoff_seconds,
        max_side=args.max_side,
        det_threshold=args.det_threshold,
        sim_threshold=args.sim_threshold,
        preflight_validate=args.preflight_validate,
        resume=args.resume,
        providers=args.providers,
        model_name=args.model_name,
    )

    runner = PipelineRunner(config)
    report = runner.run_once()
    print(report.summary_text())
    return 1 if report.run_status == "failed" else 0
