from __future__ import annotations

import time
from typing import Any, List

from .config import PipelineConfig
from .models import DetectedFace, EngineResult
from .utils import l2_normalize


class FaceEngine:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._app = None
        self._recognition_model = None
        self._face_align = None
        self._applied_providers: tuple[str, ...] = ()

    def detect_and_embed(self, image: object) -> EngineResult:
        self._ensure_loaded()
        if self._app is None or self._recognition_model is None or self._face_align is None:
            raise RuntimeError("insightface is not available in the current environment")

        detection_start = time.perf_counter()
        bboxes, kpss = self._app.det_model.detect(image, max_num=0)
        detection_seconds = time.perf_counter() - detection_start

        faces: List[DetectedFace] = []
        embedding_seconds = 0.0
        if bboxes is None or len(bboxes) == 0:
            return EngineResult(faces=faces, detection_seconds=detection_seconds)

        for index, bbox in enumerate(bboxes):
            score = float(bbox[4]) if len(bbox) > 4 else 1.0
            if score < self.config.det_threshold:
                continue
            kps = kpss[index] if kpss is not None else None
            embedding_start = time.perf_counter()
            if kps is None:
                continue
            # InsightFace exposes different norm_crop signatures across releases;
            # passing the landmarks positionally works across those variants.
            aligned = self._face_align.norm_crop(image, kps, image_size=112)
            raw_embedding = self._recognition_model.get_feat(aligned)
            embedding_seconds += time.perf_counter() - embedding_start

            try:
                vector = raw_embedding.flatten().tolist()
            except AttributeError:
                vector = list(raw_embedding)

            faces.append(
                DetectedFace(
                    bbox=[float(value) for value in bbox[:4]],
                    score=score,
                    embedding=l2_normalize(vector),
                    kps=kps.tolist() if hasattr(kps, "tolist") else kps,
                )
            )

        return EngineResult(
            faces=faces,
            detection_seconds=detection_seconds,
            embedding_seconds=embedding_seconds,
        )

    def applied_providers(self) -> tuple[str, ...]:
        return self._applied_providers or tuple(self.config.providers)

    def _ensure_loaded(self) -> None:
        if self._app is not None and self._recognition_model is not None:
            return

        try:
            from insightface.app import FaceAnalysis  # type: ignore
            from insightface.utils import face_align  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "insightface is required to run the real face engine"
            ) from exc

        app = FaceAnalysis(name=self.config.model_name, providers=list(self.config.providers))
        app.prepare(ctx_id=self.config.ctx_id, det_size=self.config.det_size)

        recognition_model = None
        for model in getattr(app, "models", {}).values():
            task_name = getattr(model, "taskname", "")
            if task_name == "recognition" and hasattr(model, "get_feat"):
                recognition_model = model
                break

        if recognition_model is None:
            raise RuntimeError("unable to locate an InsightFace recognition model")

        providers = tuple(self.config.providers)
        session = getattr(recognition_model, "session", None)
        if session is not None and hasattr(session, "get_providers"):
            try:
                providers = tuple(session.get_providers())
            except Exception:
                providers = tuple(self.config.providers)

        self._app = app
        self._recognition_model = recognition_model
        self._face_align = face_align
        self._applied_providers = providers
