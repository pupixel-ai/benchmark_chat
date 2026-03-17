from __future__ import annotations

from backend.artifact_store import ArtifactCatalogStore, build_task_asset_manifest
from backend.db import SessionLocal
from backend.models import TaskRecord
from services.asset_store import TaskAssetStore


def main() -> None:
    asset_store = TaskAssetStore()
    catalog = ArtifactCatalogStore()
    updated = 0

    with SessionLocal() as session:
        records = session.query(TaskRecord).all()

    for record in records:
        if not record.user_id:
            continue
        manifest = build_task_asset_manifest(record.task_id, record.task_dir, asset_store)
        catalog.replace_task_artifacts(record.task_id, record.user_id, manifest)
        updated += 1

    print(f"Backfilled artifact catalog for {updated} tasks")


if __name__ == "__main__":
    main()
