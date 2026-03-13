"""
工具函数
"""
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Any

ORIENTATION_TAG = 274


def normalized_exif_bytes(image) -> bytes | None:
    try:
        exif = image.getexif()
    except Exception:
        return None

    if not exif:
        return None

    if ORIENTATION_TAG in exif:
        exif[ORIENTATION_TAG] = 1
    return exif.tobytes()


def compress_image(image_path: str, output_path: str, max_size: int = 1536, quality: int = 85) -> str:
    """
    压缩图片

    Args:
        image_path: 原图片路径
        output_path: 输出路径
        max_size: 最大边长
        quality: JPEG质量

    Returns:
        压缩后的图片路径
    """
    from config import MAX_IMAGE_SIZE, JPEG_QUALITY

    from PIL import Image, ImageOps

    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    exif = normalized_exif_bytes(img)

    # 计算缩放比例
    ratio = min(MAX_IMAGE_SIZE / img.size[0], MAX_IMAGE_SIZE / img.size[1])

    if ratio < 1:
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.LANCZOS)

    if "A" in img.getbands():
        img = img.convert("RGBA")
    else:
        img = img.convert("RGB")

    if output_path.lower().endswith(".webp"):
        save_kwargs = {"quality": quality, "method": 6}
        if exif:
            save_kwargs["exif"] = exif
        img.save(output_path, "WEBP", **save_kwargs)
    else:
        save_kwargs = {"quality": JPEG_QUALITY, "optimize": True}
        if exif:
            save_kwargs["exif"] = exif
        img.save(output_path, "JPEG", **save_kwargs)

    return output_path


def smart_deduplicate(photos: List, time_window_seconds: int = 60) -> List:
    """
    智能去重：在时间窗口内选择代表性照片

    Args:
        photos: 照片列表
        time_window_seconds: 时间窗口（秒）

    Returns:
        去重后的照片列表
    """
    if len(photos) <= 1:
        return photos

    # 按时间排序
    sorted_photos = sorted(photos, key=lambda p: p.timestamp)

    # 分组
    groups = []
    current_group = [sorted_photos[0]]

    for photo in sorted_photos[1:]:
        time_diff = (photo.timestamp - current_group[0].timestamp).total_seconds()

        if time_diff <= time_window_seconds:
            current_group.append(photo)
        else:
            groups.append(current_group)
            current_group = [photo]

    groups.append(current_group)

    # 每组选择代表性照片
    selected = []

    for group in groups:
        if len(group) == 1:
            selected.append(group[0])
        else:
            # 选择人脸最多、最清晰的照片
            best = max(group, key=lambda p: len(p.faces))
            selected.append(best)

    return selected


def calculate_distance(loc1: Dict, loc2: Dict) -> float:
    """
    计算两个地点之间的距离（km）

    Args:
        loc1: {lat, lng}
        loc2: {lat, lng}

    Returns:
        距离（km）
    """
    if not loc1 or not loc2:
        return 0

    if "lat" not in loc1 or "lng" not in loc1:
        return 0

    if "lat" not in loc2 or "lng" not in loc2:
        return 0

    from geopy.distance import geodesic

    point1 = (loc1["lat"], loc1["lng"])
    point2 = (loc2["lat"], loc2["lng"])

    return geodesic(point1, point2).kilometers


def time_overlap(range1: str, range2: str) -> bool:
    """
    检查两个时间范围是否重叠

    Args:
        range1: "14:00-15:30"
        range2: "15:00-16:00"

    Returns:
        是否重叠
    """
    def parse_range(time_range: str):
        parts = time_range.split("-")
        if len(parts) != 2:
            return None, None

        def parse_time(t: str):
            try:
                h, m = map(int, t.split(":"))
                return h * 60 + m
            except:
                return 0

        return parse_time(parts[0]), parse_time(parts[1])

    start1, end1 = parse_range(range1)
    start2, end2 = parse_range(range2)

    if start1 is None or end1 is None or start2 is None or end2 is None:
        return False

    return not (end1 <= start2 or end2 <= start1)


def is_weekend(timestamp: datetime) -> bool:
    """
    判断是否是周末

    Args:
        timestamp: 时间戳

    Returns:
        是否是周末
    """
    return timestamp.weekday() >= 5  # 5=周六, 6=周日


def parse_duration(duration_str: str) -> float:
    """
    解析持续时间字符串为小时数

    Args:
        duration_str: "1.5小时" 或 "90分钟"

    Returns:
        小时数
    """
    try:
        if "小时" in duration_str:
            return float(duration_str.replace("小时", ""))
        elif "分钟" in duration_str:
            return float(duration_str.replace("分钟", "")) / 60
        else:
            return float(duration_str)
    except:
        return 0


def save_json(data: Dict, path: str):
    """保存JSON文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict:
    """加载JSON文件"""
    if not os.path.exists(path):
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_timestamp(dt: datetime) -> str:
    """格式化时间戳"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def cosine_similarity(vec1, vec2) -> float:
    """
    计算余弦相似度

    Args:
        vec1: 向量1
        vec2: 向量2

    Returns:
        相似度（0-1）
    """
    import numpy as np

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def file_sha256(path: str) -> str:
    """计算文件 SHA256。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
