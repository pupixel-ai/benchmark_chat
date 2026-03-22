from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.pipeline_service import MemoryPipelineService
from tests.test_pipeline_memory import (
    FakeAssetStore,
    FakeFaceRecognition,
    FakeFaceReviewStore,
    FakeLLMProcessor,
    ScaleReplayFakeImageProcessor,
    ScaleReplayFakeVLMAnalyzer,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        task_dir = Path(tmpdir)
        uploads_dir = task_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        for event_idx in range(120):
            for photo_idx in range(5):
                (uploads_dir / f"camera_event{event_idx:03d}_p{photo_idx:02d}.jpg").write_bytes(b"stub")
        for ref_idx in range(400):
            (uploads_dir / f"style_reference_{ref_idx:03d}.png").write_bytes(b"stub")

        with patch("services.pipeline_service.ImageProcessor", ScaleReplayFakeImageProcessor), patch(
            "services.pipeline_service.FaceRecognition", FakeFaceRecognition
        ), patch("services.pipeline_service.VLMAnalyzer", ScaleReplayFakeVLMAnalyzer), patch(
            "services.pipeline_service.LLMProcessor", FakeLLMProcessor
        ):
            service = MemoryPipelineService(
                task_id="scale_replay_v0321_2",
                task_dir=str(task_dir),
                asset_store=FakeAssetStore(),
                user_id="scale_replay_user",
                face_review_store=FakeFaceReviewStore(),
                task_version="v0321.2",
            )
            result = service.run(max_photos=1000, use_cache=False)

        memory = result["memory"]
        report = {
            "version": result["version"],
            "pipeline_family": memory["pipeline_family"],
            "event_count": result["summary"]["event_count"],
            "event_window_count": memory["summary"]["event_window_count"],
            "relationship_count": result["summary"]["relationship_count"],
            "reference_media_signal_count": memory["summary"]["reference_media_signal_count"],
            "over_segmentation_anomaly": memory["summary"]["over_segmentation_anomaly"],
            "profile_bucket_count": memory["summary"]["profile_bucket_count"],
            "staging_db_path": memory["artifacts"]["staging_db_path"],
            "reference_media_path": memory["artifacts"]["reference_media_path"],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
