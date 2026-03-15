from __future__ import annotations

import argparse
from pathlib import Path

from config import FACE_DET_THRESHOLD, FACE_MAX_SIDE, FACE_MODEL_NAME, FACE_PROVIDERS
from services.face_landmarks import FaceLandmarkAnalyzer
from services.face_recognition import _load_face_recognition_modules
from utils import cosine_similarity


def build_engine():
    PipelineConfig, FaceEngine, _, load_image, resize_image = _load_face_recognition_modules()
    config = PipelineConfig.from_args(
        input_dir=".",
        db_path="runtime/diagnostics/face_pair.db",
        index_path="runtime/diagnostics/face_pair.index",
        max_side=FACE_MAX_SIDE,
        det_threshold=FACE_DET_THRESHOLD,
        sim_threshold=0.0,
        providers=FACE_PROVIDERS,
        model_name=FACE_MODEL_NAME,
    )
    return FaceEngine(config), load_image, resize_image


def scale_bbox(bbox, scale_x: float, scale_y: float, width: int, height: int) -> dict:
    x1, y1, x2, y2 = bbox[:4]
    scaled_x1 = max(0, min(width, int(round(x1 * scale_x))))
    scaled_y1 = max(0, min(height, int(round(y1 * scale_y))))
    scaled_x2 = max(0, min(width, int(round(x2 * scale_x))))
    scaled_y2 = max(0, min(height, int(round(y2 * scale_y))))
    return {
        "bbox": [scaled_x1, scaled_y1, scaled_x2, scaled_y2],
        "bbox_xywh": {
            "x": scaled_x1,
            "y": scaled_y1,
            "w": max(0, scaled_x2 - scaled_x1),
            "h": max(0, scaled_y2 - scaled_y1),
        },
    }


def detect_faces(image_path: str, analyzer: FaceLandmarkAnalyzer):
    engine, load_image, resize_image = build_engine()
    raw = load_image(Path(image_path))
    resized = resize_image(raw, FACE_MAX_SIDE)
    result = engine.detect_and_embed(resized.pixels)
    scale_x = raw.width / resized.width
    scale_y = raw.height / resized.height

    faces = []
    for index, face in enumerate(result.faces):
        bbox = scale_bbox(face.bbox, scale_x, scale_y, raw.width, raw.height)
        pose = analyzer.analyze_face(
            raw.pixels,
            bbox["bbox_xywh"],
            insight_pose=face.insight_pose,
            insight_landmark_2d_106=face.insight_landmark_2d_106,
            insight_landmark_3d_68=face.insight_landmark_3d_68,
        )
        center_x = bbox["bbox_xywh"]["x"] + bbox["bbox_xywh"]["w"] / 2.0
        faces.append(
            {
                "index": index,
                "center_x": center_x,
                "bbox": bbox["bbox"],
                "score": float(face.score),
                "embedding": face.embedding,
                "pose_bucket": pose.get("pose_bucket"),
                "pose_yaw": pose.get("pose_yaw"),
                "landmark_detected": pose.get("landmark_detected"),
                "landmark_source": pose.get("landmark_source"),
            }
        )
    faces.sort(key=lambda item: item["center_x"])
    return faces


def main() -> None:
    parser = argparse.ArgumentParser(description="诊断两张图片中的人脸相似度与 pose 信息")
    parser.add_argument("image_a")
    parser.add_argument("image_b")
    args = parser.parse_args()

    analyzer = FaceLandmarkAnalyzer()
    faces_a = detect_faces(args.image_a, analyzer)
    faces_b = detect_faces(args.image_b, analyzer)

    print(f"image_a={args.image_a}")
    for face in faces_a:
        print(
            f"  A[{face['index']}] bbox={face['bbox']} score={face['score']:.3f} "
            f"pose={face['pose_bucket']} yaw={face['pose_yaw']} source={face['landmark_source']}"
        )

    print(f"image_b={args.image_b}")
    for face in faces_b:
        print(
            f"  B[{face['index']}] bbox={face['bbox']} score={face['score']:.3f} "
            f"pose={face['pose_bucket']} yaw={face['pose_yaw']} source={face['landmark_source']}"
        )

    best_pair = None
    for face_a in faces_a:
        for face_b in faces_b:
            similarity = cosine_similarity(face_a["embedding"], face_b["embedding"])
            if best_pair is None or similarity > best_pair["similarity"]:
                best_pair = {
                    "a_index": face_a["index"],
                    "b_index": face_b["index"],
                    "similarity": similarity,
                    "a_pose": face_a["pose_bucket"],
                    "b_pose": face_b["pose_bucket"],
                }

    if best_pair is None:
        print("未找到可比较的人脸。")
        return

    print(
        "best_pair="
        f"A[{best_pair['a_index']}] vs B[{best_pair['b_index']}] "
        f"similarity={best_pair['similarity']:.4f} "
        f"poses={best_pair['a_pose']} -> {best_pair['b_pose']}"
    )


if __name__ == "__main__":
    main()
