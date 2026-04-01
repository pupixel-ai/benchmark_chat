from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from config import FEISHU_LONG_CONNECTION_LOG_LEVEL, PROJECT_ROOT
from services.reflection.feishu_long_connection import run_feishu_long_connection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Feishu long connection callback worker.")
    parser.add_argument("--project-root", default=PROJECT_ROOT, help="Project root for reflection asset access.")
    parser.add_argument("--log-level", default=FEISHU_LONG_CONNECTION_LOG_LEVEL, help="Feishu SDK log level, e.g. INFO or DEBUG.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_feishu_long_connection(
        project_root=args.project_root,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
