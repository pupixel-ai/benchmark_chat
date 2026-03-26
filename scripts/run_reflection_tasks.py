from __future__ import annotations

import argparse
import json

from config import PROJECT_ROOT, USER_NAME
from services.reflection import run_reflection_task_generation


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate offline reflection patterns and local review tasks.")
    parser.add_argument("--user-name", default=USER_NAME, help="Reflection user name. Defaults to MEMORY_USER_NAME.")
    args = parser.parse_args()

    result = run_reflection_task_generation(
        project_root=PROJECT_ROOT,
        user_name=args.user_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
