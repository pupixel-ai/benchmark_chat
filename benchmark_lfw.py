#!/usr/bin/env python3
"""
LFW 10-fold 基准测试。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import ssl
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.datasets import fetch_lfw_pairs

from config import FACE_MATCH_TOP_K, FACE_MODEL_NAME, LFW_BENCHMARK_DIR, RUNTIME_DIR
from vendor.face_recognition_src.face_recognition.config import PipelineConfig
from vendor.face_recognition_src.face_recognition.engine import FaceEngine
from vendor.face_recognition_src.face_recognition.image_io import load_image


DEFAULT_CACHE = Path(RUNTIME_DIR) / "lfw_cache"
DEFAULT_LFW_ROOT = DEFAULT_CACHE / "lfw_home" / "lfw_funneled"
DEFAULT_PAIRS_FILE = DEFAULT_CACHE / "lfw_home" / "pairs.txt"


@dataclass(frozen=True)
class PairSample:
    left: Path
    right: Path
    is_same: int


def ensure_lfw_dataset(data_home: Path, lfw_root: Path, pairs_file: Path, download_if_missing: bool) -> None:
    if lfw_root.exists() and pairs_file.exists():
        return
    if not download_if_missing:
        raise FileNotFoundError(f"LFW 数据集缺失: {lfw_root}")

    ssl._create_default_https_context = ssl._create_unverified_context
    fetch_lfw_pairs(
        subset="10_folds",
        color=True,
        resize=1.0,
        data_home=str(data_home),
        download_if_missing=True,
    )


def read_pairs(pairs_file: Path, lfw_root: Path) -> tuple[list[PairSample], int, int]:
    lines = pairs_file.read_text(encoding="utf-8").splitlines()
    header = lines[0].split()
    fold_count = int(header[0])
    pairs_per_type = int(header[1])
    pairs: List[PairSample] = []

    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) == 3:
            person, left_idx, right_idx = parts
            left = lfw_root / person / f"{person}_{int(left_idx):04d}.jpg"
            right = lfw_root / person / f"{person}_{int(right_idx):04d}.jpg"
            is_same = 1
        else:
            left_person, left_idx, right_person, right_idx = parts
            left = lfw_root / left_person / f"{left_person}_{int(left_idx):04d}.jpg"
            right = lfw_root / right_person / f"{right_person}_{int(right_idx):04d}.jpg"
            is_same = 0
        pairs.append(PairSample(left=left, right=right, is_same=is_same))

    return pairs, fold_count, pairs_per_type * 2


def load_embedding_cache(cache_path: Path) -> Dict[str, object]:
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        return {
            "embeddings": payload.get("embeddings", {}),
            "meta": payload.get("meta", {}),
        }

    bootstrap_path = Path(RUNTIME_DIR) / "lfw_dev_eval_cache.pkl"
    if bootstrap_path.exists():
        with bootstrap_path.open("rb") as handle:
            payload = pickle.load(handle)
        return {
            "embeddings": payload.get("embeddings", {}),
            "meta": payload.get("meta", {}),
        }

    return {"embeddings": {}, "meta": {}}


def compute_embeddings(image_paths: Sequence[Path], cache_path: Path) -> Dict[str, object]:
    payload = load_embedding_cache(cache_path)
    embeddings: Dict[str, object] = payload["embeddings"]
    meta: Dict[str, Dict[str, object]] = payload["meta"]

    config = PipelineConfig.from_args(
        input_dir=".",
        db_path=str(Path(RUNTIME_DIR) / "lfw_benchmark.db"),
        index_path=str(Path(RUNTIME_DIR) / "lfw_benchmark.index"),
        model_name=FACE_MODEL_NAME,
    )
    engine = FaceEngine(config)

    total = len(image_paths)
    started_at = time.perf_counter()
    processed = 0

    for path in image_paths:
        key = str(path)
        if key in embeddings:
            processed += 1
            continue

        try:
            image = load_image(path)
            result = engine.detect_and_embed(image.pixels)
            if result.faces:
                face = max(result.faces, key=lambda item: float(item.score))
                embeddings[key] = np.asarray(face.embedding, dtype=np.float32)
                meta[key] = {
                    "face_count": len(result.faces),
                    "score": float(face.score),
                    "error": None,
                }
            else:
                embeddings[key] = None
                meta[key] = {"face_count": 0, "score": None, "error": "no_face"}
        except Exception as exc:  # pragma: no cover - depends on dataset decode/runtime
            embeddings[key] = None
            meta[key] = {"face_count": 0, "score": None, "error": f"{type(exc).__name__}: {exc}"}

        processed += 1
        if processed % 200 == 0 or processed == total:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as handle:
                pickle.dump({"embeddings": embeddings, "meta": meta}, handle)
            elapsed = time.perf_counter() - started_at
            rate = processed / elapsed if elapsed else 0.0
            eta = (total - processed) / rate if rate else 0.0
            print(f"progress {processed}/{total} elapsed={elapsed:.1f}s eta={eta:.1f}s")

    with cache_path.open("wb") as handle:
        pickle.dump({"embeddings": embeddings, "meta": meta}, handle)
    return {"embeddings": embeddings, "meta": meta}


def pair_scores(samples: Sequence[PairSample], embeddings: Dict[str, object], meta: Dict[str, Dict[str, object]]) -> tuple[np.ndarray, np.ndarray, int]:
    scores = []
    labels = []
    missing = 0
    for sample in samples:
        left = embeddings.get(str(sample.left))
        right = embeddings.get(str(sample.right))
        if left is None or right is None:
            scores.append(-2.0)
            missing += 1
        else:
            scores.append(float(np.dot(left, right)))
        labels.append(sample.is_same)
    return np.asarray(scores), np.asarray(labels), missing


def accuracy_stats(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    predictions = (scores >= threshold).astype(np.int32)
    positives = max(1, int((labels == 1).sum()))
    negatives = max(1, int((labels == 0).sum()))
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    return {
        "accuracy": float((predictions == labels).mean()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "far": fp / negatives,
        "frr": fn / positives,
    }


def choose_threshold(scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> Dict[str, float]:
    best = None
    for threshold in thresholds:
        stats = accuracy_stats(scores, labels, float(threshold))
        candidate = (-stats["far"], stats["accuracy"], float(threshold))
        if best is None or candidate > best["rank"]:
            best = {
                "rank": candidate,
                "threshold": float(threshold),
                "accuracy": stats["accuracy"],
                "far": stats["far"],
                "frr": stats["frr"],
            }
    assert best is not None
    return best


def run_benchmark(output_dir: Path, lfw_root: Path, pairs_file: Path, download_if_missing: bool) -> Dict[str, object]:
    ensure_lfw_dataset(DEFAULT_CACHE, lfw_root, pairs_file, download_if_missing)
    samples, fold_count, fold_size = read_pairs(pairs_file, lfw_root)
    image_paths = sorted({sample.left for sample in samples} | {sample.right for sample in samples})
    cache_path = output_dir / "embeddings.pkl"
    payload = compute_embeddings(image_paths, cache_path)
    embeddings = payload["embeddings"]
    meta = payload["meta"]

    thresholds = np.round(np.arange(0.30, 0.751, 0.01), 2)
    folds = []
    total_fp = total_fn = total_tp = total_tn = total_missing = 0
    per_fold_thresholds = []

    for fold_index in range(fold_count):
        start = fold_index * fold_size
        end = start + fold_size
        test_samples = samples[start:end]
        train_samples = samples[:start] + samples[end:]
        train_scores, train_labels, _ = pair_scores(train_samples, embeddings, meta)
        test_scores, test_labels, missing_pairs = pair_scores(test_samples, embeddings, meta)

        chosen = choose_threshold(train_scores, train_labels, thresholds)
        stats = accuracy_stats(test_scores, test_labels, chosen["threshold"])
        folds.append(
            {
                "fold": fold_index + 1,
                "threshold": chosen["threshold"],
                "accuracy": round(stats["accuracy"], 6),
                "far": round(stats["far"], 6),
                "frr": round(stats["frr"], 6),
                "missing_pairs": missing_pairs,
            }
        )
        per_fold_thresholds.append(chosen["threshold"])
        total_tp += stats["tp"]
        total_tn += stats["tn"]
        total_fp += stats["fp"]
        total_fn += stats["fn"]
        total_missing += missing_pairs

    mean_accuracy = float(np.mean([fold["accuracy"] for fold in folds]))
    std_accuracy = float(np.std([fold["accuracy"] for fold in folds]))
    mean_threshold = float(np.mean(per_fold_thresholds))
    detection_failures = len([1 for item in meta.values() if item.get("error") == "no_face"])
    image_errors = len([1 for item in meta.values() if item.get("error") not in (None, "no_face")])

    result = {
        "generated_at": datetime.now().isoformat(),
        "dataset": "LFW 10-fold",
        "model_name": FACE_MODEL_NAME,
        "match_top_k": FACE_MATCH_TOP_K,
        "recommended_threshold": round(mean_threshold, 4),
        "strong_threshold": round(mean_threshold, 4),
        "mean_threshold": round(mean_threshold, 4),
        "scan_range": {"min": 0.30, "max": 0.75, "step": 0.01},
        "mean_accuracy": round(mean_accuracy, 6),
        "std_accuracy": round(std_accuracy, 6),
        "false_accept_rate": round(total_fp / max(1, total_fp + total_tn), 6),
        "false_reject_rate": round(total_fn / max(1, total_fn + total_tp), 6),
        "detection_failures": detection_failures,
        "image_level_errors": image_errors,
        "missing_pairs": total_missing,
        "folds": folds,
    }
    return result


def write_outputs(output_dir: Path, result: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_json = output_dir / "latest.json"
    summary_md = output_dir / "summary.md"
    latest_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# LFW 10-fold Benchmark",
        "",
        f"- Generated: {result['generated_at']}",
        f"- Model: {result['model_name']}",
        f"- Recommended strong threshold: {result['recommended_threshold']}",
        f"- Mean accuracy: {result['mean_accuracy']:.4%}",
        f"- Std accuracy: {result['std_accuracy']:.4%}",
        f"- FAR: {result['false_accept_rate']:.4%}",
        f"- FRR: {result['false_reject_rate']:.4%}",
        f"- Detection failures: {result['detection_failures']}",
        f"- Missing pairs: {result['missing_pairs']}",
        "",
        "| Fold | Threshold | Accuracy | FAR | FRR | Missing Pairs |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for fold in result["folds"]:
        lines.append(
            f"| {fold['fold']} | {fold['threshold']:.2f} | {fold['accuracy']:.4%} | "
            f"{fold['far']:.4%} | {fold['frr']:.4%} | {fold['missing_pairs']} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LFW 10-fold benchmark")
    parser.add_argument("--output-dir", default=LFW_BENCHMARK_DIR, help="输出目录")
    parser.add_argument("--lfw-root", default=str(DEFAULT_LFW_ROOT), help="LFW 图片目录")
    parser.add_argument("--pairs-file", default=str(DEFAULT_PAIRS_FILE), help="pairs.txt 路径")
    parser.add_argument("--download-if-missing", action="store_true", help="缺失时自动下载")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    lfw_root = Path(args.lfw_root)
    pairs_file = Path(args.pairs_file)

    result = run_benchmark(
        output_dir=output_dir,
        lfw_root=lfw_root,
        pairs_file=pairs_file,
        download_if_missing=args.download_if_missing,
    )
    write_outputs(output_dir, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
