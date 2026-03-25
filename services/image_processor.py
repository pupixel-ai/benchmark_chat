"""
图片预处理模块
"""
import os
import re
import tempfile
from datetime import datetime
from typing import Callable, Dict, List, Tuple, Optional
from zipfile import ZipFile
from PIL import Image, ExifTags, ImageDraw, ImageFont, ImageOps
from pillow_heif import register_heif_opener
import exifread
from models import Photo
from config import CACHE_DIR, MAX_IMAGE_SIZE, JPEG_QUALITY, AMAP_API_KEY, PROJECT_ROOT
from utils import compress_image, file_sha256, normalized_exif_bytes

# 注册HEIC格式支持
register_heif_opener()


DEDUP_BURST_WINDOW_SECONDS = 5
DEDUP_HASH_SIZE = 8
DEDUP_MAX_DISTANCE = 4
FILENAME_DATETIME_PATTERNS = (
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})(?:\d{3})?(?!\d)"),
    re.compile(
        r"(?<!\d)(20\d{2})[-_.](\d{2})[-_.](\d{2})(?:[ T_:-]+|[ _-]at[ _-])(\d{2})[-_.:](\d{2})[-_.:](\d{2})(?!\d)"
    ),
)


class ImageProcessor:
    """图片处理器"""

    def __init__(self, cache_dir: str = CACHE_DIR, amap_api_key: str = AMAP_API_KEY):
        self.cache_dir = cache_dir
        self.amap_api_key = amap_api_key
        self.last_dedupe_report: Dict[str, object] = {}
        self.compress_dir = os.path.join(cache_dir, "compressed_images")
        self.jpeg_dir = os.path.join(cache_dir, "jpeg_images")  # 全尺寸JPEG，用于人脸识别
        self.boxed_dir = os.path.join(cache_dir, "boxed_images")  # 带人脸框的图片
        self.face_dir = os.path.join(cache_dir, "face_crops")  # 人脸裁剪图
        os.makedirs(self.compress_dir, exist_ok=True)
        os.makedirs(self.jpeg_dir, exist_ok=True)
        os.makedirs(self.boxed_dir, exist_ok=True)
        os.makedirs(self.face_dir, exist_ok=True)

    def list_supported_photos(self, photo_dir: str) -> List[str]:
        supported_formats = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.livp', '.webp'}
        photo_files = []

        for filename in sorted(os.listdir(photo_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_formats:
                photo_files.append(filename)

        return photo_files

    def load_photos(self, photo_dir: str, max_photos: int = None) -> List[Photo]:
        """
        加载照片目录中的所有照片

        Args:
            photo_dir: 照片目录路径
            max_photos: 最多加载多少张

        Returns:
            照片列表
        """
        photos, _ = self.load_photos_with_errors(photo_dir, max_photos)
        return photos

    def load_photos_with_errors(
        self,
        photo_dir: str,
        max_photos: int = None,
    ) -> Tuple[List[Photo], List[Dict]]:
        """
        加载照片目录中的所有照片，并返回坏图/读取失败列表。

        Returns:
            (照片列表, 错误列表)
        """
        photos = []
        errors = []
        supported_files = self.list_supported_photos(photo_dir)
        selected_files = supported_files[:max_photos] if max_photos else supported_files

        for index, filename in enumerate(selected_files, start=1):
            path = os.path.join(photo_dir, filename)
            photo_id = f"photo_{index:03d}"

            try:
                # 读取EXIF信息
                exif = self._read_exif(path)

                photo = Photo(
                    photo_id=photo_id,
                    filename=filename,
                    path=path,
                    timestamp=exif.get("datetime", datetime.now()),
                    location=exif.get("location", {}),
                    source_hash=file_sha256(path),
                )

                photos.append(photo)

            except Exception as e:
                print(f"警告：无法读取照片 {filename}: {e}")
                errors.append({
                    "image_id": photo_id,
                    "filename": filename,
                    "path": path,
                    "step": "load",
                    "error": str(e),
                })

        # 按时间排序
        photos.sort(key=lambda p: p.timestamp)

        return photos, errors

    def convert_to_jpeg(
        self,
        photos: List[Photo],
        *,
        normalize_live_photos: bool = False,
        progress_callback: Callable[[int, int, Photo], None] | None = None,
    ) -> List[Photo]:
        """
        为人脸识别准备工作图。
        原始上传文件保持不变；仅在 HEIC 或带方向标签的图片上生成标准朝向的 JPEG 工作图。

        Args:
            photos: 照片列表

        Returns:
            转换后的照片列表
        """
        total_photos = len(photos)
        for index, photo in enumerate(photos, start=1):
            photo.original_path = photo.path
            lower_filename = photo.filename.lower()
            is_live_like_source = lower_filename.endswith(('.heic', '.heif', '.livp'))
            needs_working_copy = lower_filename.endswith(('.heic', '.heif', '.livp'))
            if normalize_live_photos and is_live_like_source:
                needs_working_copy = True

            if not needs_working_copy:
                try:
                    with Image.open(photo.path) as source:
                        orientation = source.getexif().get(274, 1)
                        needs_working_copy = orientation not in (None, 1)
                except Exception:
                    needs_working_copy = False

            if not needs_working_copy:
                continue

            try:
                jpeg_filename = f"face_input_{photo.photo_id}.jpg"
                jpeg_path = os.path.join(self.jpeg_dir, jpeg_filename)

                image = self._open_working_image(photo.path)
                normalized = ImageOps.exif_transpose(image)
                exif = normalized_exif_bytes(normalized)
                working = normalized.convert("RGB")
                if exif:
                    working.save(jpeg_path, "JPEG", quality=95, exif=exif)
                else:
                    working.save(jpeg_path, "JPEG", quality=95)

                photo.path = jpeg_path

            except Exception as e:
                print(f"警告：转换HEIC失败 {photo.filename}: {e}")
                photo.processing_errors["convert_to_jpeg"] = str(e)
            finally:
                if progress_callback is not None:
                    progress_callback(index, total_photos, photo)

        return photos

    def _open_working_image(self, image_path: str) -> Image.Image:
        lower_path = image_path.lower()
        if lower_path.endswith(".livp"):
            extracted = self._extract_livp_still(image_path)
            if extracted is not None:
                return extracted
        return Image.open(image_path)

    def _extract_livp_still(self, image_path: str) -> Optional[Image.Image]:
        try:
            with ZipFile(image_path) as archive:
                candidate_names = sorted(
                    name
                    for name in archive.namelist()
                    if not name.endswith("/")
                    and os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
                )
                for name in candidate_names:
                    suffix = os.path.splitext(name)[1].lower() or ".jpg"
                    with archive.open(name) as handle:
                        payload = handle.read()
                    if not payload:
                        continue
                    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                    temp_path = temp_file.name
                    try:
                        temp_file.write(payload)
                        temp_file.flush()
                        temp_file.close()
                        with Image.open(temp_path) as image:
                            return image.copy()
                    finally:
                        try:
                            temp_file.close()
                        except Exception:
                            pass
                        try:
                            os.unlink(temp_path)
                        except FileNotFoundError:
                            pass
        except Exception:
            return None
        return None

    def preprocess(
        self,
        photos: List[Photo],
        *,
        progress_callback: Callable[[int, int, Photo], None] | None = None,
    ) -> List[Photo]:
        """
        预处理照片：压缩

        Args:
            photos: 照片列表

        Returns:
            处理后的照片列表
        """
        if progress_callback is None:
            return self._compress_photos(photos)
        return self._compress_photos(photos, progress_callback=progress_callback)

    def dedupe_before_face_recognition(self, photos: List[Photo]) -> List[Photo]:
        """
        在人脸识别前做轻量去重，优先消除 burst shots 和重复上传。

        说明：
        - 这里只影响后续分析链路的输入，不修改原始上传文件。
        - 同一 source_hash 的照片直接去重。
        - 对时间接近的照片再做轻量感知哈希比较，减少近重复帧。
        """
        if len(photos) <= 1:
            self.last_dedupe_report = {
                "total_images": len(photos),
                "exact_duplicates_removed": 0,
                "near_duplicates_removed": 0,
                "burst_groups": [],
                "representative_photo_ids": [photo.photo_id for photo in photos],
                "duplicate_backrefs": {},
            }
            return photos

        deduped: List[Photo] = []
        source_hash_seen: Dict[str, str] = {}
        recent_representatives: List[Tuple[Photo, Optional[str]]] = []
        burst_groups: Dict[str, List[str]] = {}
        duplicate_backrefs: Dict[str, Dict[str, object]] = {}
        exact_duplicates_removed = 0
        near_duplicates_removed = 0

        for photo in photos:
            if photo.source_hash and photo.source_hash in source_hash_seen:
                exact_duplicates_removed += 1
                representative_photo_id = source_hash_seen[photo.source_hash]
                duplicate_backrefs[photo.photo_id] = {
                    "duplicate_type": "exact_duplicate",
                    "representative_photo_id": representative_photo_id,
                    "source_hash": photo.source_hash,
                }
                continue

            current_hash = self._compute_average_hash(photo.path)
            is_duplicate = False
            updated_recent: List[Tuple[Photo, Optional[str]]] = []

            for representative, representative_hash in recent_representatives:
                age_seconds = abs((photo.timestamp - representative.timestamp).total_seconds())
                if age_seconds <= DEDUP_BURST_WINDOW_SECONDS:
                    updated_recent.append((representative, representative_hash))
                    if self._is_near_duplicate(photo, current_hash, representative, representative_hash):
                        is_duplicate = True
                        near_duplicates_removed += 1
                        burst_key = representative.photo_id
                        burst_groups.setdefault(burst_key, [representative.photo_id])
                        burst_groups[burst_key].append(photo.photo_id)
                        duplicate_backrefs[photo.photo_id] = {
                            "duplicate_type": "near_duplicate",
                            "representative_photo_id": representative.photo_id,
                            "age_seconds": age_seconds,
                        }
                # 超出 burst window 的代表图不再参与比较

            if is_duplicate:
                continue

            deduped.append(photo)
            if photo.source_hash:
                source_hash_seen[photo.source_hash] = photo.photo_id
            updated_recent.append((photo, current_hash))
            recent_representatives = updated_recent

        for photo in deduped:
            burst_groups.setdefault(photo.photo_id, [photo.photo_id])

        self.last_dedupe_report = {
            "total_images": len(photos),
            "retained_images": len(deduped),
            "exact_duplicates_removed": exact_duplicates_removed,
            "near_duplicates_removed": near_duplicates_removed,
            "burst_group_count": len(burst_groups),
            "burst_groups": [
                {
                    "representative_photo_id": representative_photo_id,
                    "photo_ids": photo_ids,
                }
                for representative_photo_id, photo_ids in burst_groups.items()
            ],
            "representative_photo_ids": [photo.photo_id for photo in deduped],
            "duplicate_backrefs": duplicate_backrefs,
        }

        return deduped

    def _compress_photos(
        self,
        photos: List[Photo],
        *,
        progress_callback: Callable[[int, int, Photo], None] | None = None,
    ) -> List[Photo]:
        """
        压缩所有照片

        Args:
            photos: 照片列表

        Returns:
            压缩后的照片列表
        """
        total = len(photos)
        processed = 0

        for photo in photos:
            try:
                # 压缩文件名
                compressed_filename = f"compressed_{photo.photo_id}.webp"
                compressed_path = os.path.join(self.compress_dir, compressed_filename)

                # 压缩
                compress_image(photo.path, compressed_path, MAX_IMAGE_SIZE, JPEG_QUALITY)

                # 更新照片对象
                photo.compressed_path = compressed_path

            except Exception as e:
                print(f"警告：压缩照片 {photo.filename} 失败: {e}")
                photo.processing_errors["preprocess"] = str(e)
            finally:
                processed += 1
                if progress_callback is not None:
                    progress_callback(processed, total, photo)

        return photos

    def _compute_average_hash(self, image_path: str) -> Optional[str]:
        try:
            with Image.open(image_path) as source:
                normalized = ImageOps.exif_transpose(source).convert("L")
                resized = normalized.resize((DEDUP_HASH_SIZE, DEDUP_HASH_SIZE), Image.Resampling.LANCZOS)
                pixels = list(resized.getdata())
        except Exception:
            return None

        if not pixels:
            return None

        avg = sum(pixels) / len(pixels)
        return "".join("1" if pixel >= avg else "0" for pixel in pixels)

    def _is_near_duplicate(
        self,
        current_photo: Photo,
        current_hash: Optional[str],
        representative: Photo,
        representative_hash: Optional[str],
    ) -> bool:
        if not current_hash or not representative_hash:
            return False

        if not self._shares_location_bucket(current_photo, representative):
            return False

        return self._hamming_distance(current_hash, representative_hash) <= DEDUP_MAX_DISTANCE

    def _shares_location_bucket(self, left: Photo, right: Photo) -> bool:
        left_name = str((left.location or {}).get("name") or "").strip().lower()
        right_name = str((right.location or {}).get("name") or "").strip().lower()

        if left_name and right_name:
            return left_name == right_name

        return True

    def _hamming_distance(self, left: str, right: str) -> int:
        return sum(ch1 != ch2 for ch1, ch2 in zip(left, right))

    def _read_exif(self, image_path: str) -> dict:
        """
        读取照片的EXIF信息（支持HEIC，使用exifread）

        Args:
            image_path: 照片路径

        Returns:
            EXIF信息字典
        """
        exif_data = {}

        try:
            # 使用 exifread 读取（支持HEIC）
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

                # 读取时间
                for tag_name in ("EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime"):
                    parsed_datetime = self._parse_exif_datetime_value(tags.get(tag_name))
                    if parsed_datetime is not None:
                        exif_data["datetime"] = parsed_datetime
                        break

                # 读取GPS信息
                lat_tag = tags.get('GPS GPSLatitude')
                lat_ref_tag = tags.get('GPS GPSLatitudeRef')
                lon_tag = tags.get('GPS GPSLongitude')
                lon_ref_tag = tags.get('GPS GPSLongitudeRef')

                if lat_tag and lat_ref_tag and lon_tag and lon_ref_tag:
                    try:
                        lat = self._convert_dms_to_decimal(lat_tag.values, str(lat_ref_tag))
                        lon = self._convert_dms_to_decimal(lon_tag.values, str(lon_ref_tag))

                        # 逆地理编码获取地址
                        address = self._reverse_geocode(lon, lat)
                        exif_data["location"] = {"lat": lat, "lng": lon, "name": address}
                    except Exception as e:
                        pass

        except Exception as e:
            pass

        # exifread 对 WebP 等容器格式支持有限，补一层 Pillow 回退
        if "datetime" not in exif_data or "location" not in exif_data:
            self._read_exif_with_pillow(image_path, exif_data)

        # 很多截图/AI图没有 EXIF，但文件名里仍包含原始拍摄或导出时间
        if "datetime" not in exif_data:
            filename_datetime = self._parse_datetime_from_filename(os.path.basename(image_path))
            if filename_datetime is not None:
                exif_data["datetime"] = filename_datetime

        # 如果EXIF中没有时间，使用文件修改时间
        if "datetime" not in exif_data:
            try:
                file_mtime = os.path.getmtime(image_path)
                exif_data["datetime"] = datetime.fromtimestamp(file_mtime)
            except:
                exif_data["datetime"] = datetime.now()

        return exif_data

    def _read_exif_with_pillow(self, image_path: str, exif_data: dict):
        try:
            with Image.open(image_path) as image:
                raw_exif = image.getexif()
                if not raw_exif:
                    return

                mapped = {
                    ExifTags.TAGS.get(tag, tag): value
                    for tag, value in raw_exif.items()
                }

                if "datetime" not in exif_data:
                    parsed_datetime = (
                        self._parse_exif_datetime_value(mapped.get("DateTimeOriginal"))
                        or self._parse_exif_datetime_value(mapped.get("DateTimeDigitized"))
                        or self._parse_exif_datetime_value(mapped.get("DateTime"))
                    )
                    if parsed_datetime is not None:
                        exif_data["datetime"] = parsed_datetime

                if "location" not in exif_data:
                    gps_info = mapped.get("GPSInfo")
                    if gps_info:
                        gps_tags = {
                            ExifTags.GPSTAGS.get(tag, tag): value
                            for tag, value in gps_info.items()
                        }
                        lat = self._convert_pillow_gps(
                            gps_tags.get("GPSLatitude"),
                            gps_tags.get("GPSLatitudeRef"),
                        )
                        lon = self._convert_pillow_gps(
                            gps_tags.get("GPSLongitude"),
                            gps_tags.get("GPSLongitudeRef"),
                        )
                        if lat is not None and lon is not None:
                            address = self._reverse_geocode(lon, lat)
                            exif_data["location"] = {"lat": lat, "lng": lon, "name": address}
        except Exception:
            pass

    def _parse_exif_datetime_value(self, raw_value: object) -> Optional[datetime]:
        text = str(raw_value or "").strip().replace("\x00", "")
        if not text:
            return None
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text[:19], fmt)
            except Exception:
                continue
        return None

    def _parse_datetime_from_filename(self, filename: str) -> Optional[datetime]:
        stem = os.path.splitext(os.path.basename(filename or ""))[0]
        if not stem:
            return None

        for pattern in FILENAME_DATETIME_PATTERNS:
            match = pattern.search(stem)
            if not match:
                continue
            try:
                year, month, day, hour, minute, second = [int(part) for part in match.groups()]
                return datetime(year, month, day, hour, minute, second)
            except ValueError:
                continue

        return None

    def _reverse_geocode(self, lng: float, lat: float) -> str:
        """
        逆地理编码：将经纬度转换为地址（到区级）

        Args:
            lng: 经度
            lat: 纬度

        Returns:
            地址字符串，如 "北京市海淀区"
        """
        if not self.amap_api_key:
            return ""

        try:
            import urllib.request
            import json

            url = f"https://restapi.amap.com/v3/geocode/regeo?key={self.amap_api_key}&location={lng},{lat}&extensions=base"

            with urllib.request.urlopen(url, timeout=3) as response:
                data = json.loads(response.read().decode())

                if data.get("status") == "1":
                    addr_comp = data.get("regeocode", {}).get("addressComponent", {})
                    province = addr_comp.get("province", "")
                    city = addr_comp.get("city", "")
                    district = addr_comp.get("district", "")
                    township = addr_comp.get("township", "")

                    # 组装地址：省+市+区
                    parts = []
                    if province:
                        # 处理直辖市情况（province直接是区名）
                        if province.endswith("市") or province in ["北京", "上海", "重庆", "天津"]:
                            parts.append(province.replace("市", ""))
                        else:
                            parts.append(province.replace("省", ""))
                    if city and isinstance(city, str) and city != province:
                        parts.append(city.replace("市", ""))
                    if district:
                        parts.append(district)
                    if not parts and township:
                        parts.append(township)

                    return "".join(parts)

        except Exception as e:
            pass

        return ""

    def _convert_dms_to_decimal(self, dms, ref):
        """
        将度分秒格式转换为十进制度数

        Args:
            dms: [度, 分, 秒] 列表，每个元素是分数对象
            ref: 方向引用 (N/S/E/W)

        Returns:
            十进制度数
        """
        degrees = dms[0].num / dms[0].den
        minutes = dms[1].num / dms[1].den
        seconds = dms[2].num / dms[2].den
        decimal = degrees + minutes / 60 + seconds / 3600
        if ref in ['S', 'W']:
            decimal = -decimal
        return decimal

    def _convert_pillow_gps(self, dms, ref):
        if not dms or not ref:
            return None
        try:
            degrees = self._ratio_to_float(dms[0])
            minutes = self._ratio_to_float(dms[1])
            seconds = self._ratio_to_float(dms[2])
            decimal = degrees + minutes / 60 + seconds / 3600
            if str(ref) in ['S', 'W']:
                decimal = -decimal
            return decimal
        except Exception:
            return None

    def _ratio_to_float(self, value):
        if hasattr(value, "num") and hasattr(value, "den"):
            return value.num / value.den
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            return value.numerator / value.denominator
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return value[0] / value[1]
        return float(value)

    def _resolve_face_box(self, face: Dict) -> Dict[str, int]:
        bbox_xywh = face.get("bbox_xywh")
        if bbox_xywh:
            return {
                "x": int(bbox_xywh["x"]),
                "y": int(bbox_xywh["y"]),
                "w": int(bbox_xywh["w"]),
                "h": int(bbox_xywh["h"]),
            }

        bbox = face.get("bbox", [0, 0, 0, 0])
        if isinstance(bbox, dict):
            return {
                "x": int(bbox.get("x", 0)),
                "y": int(bbox.get("y", 0)),
                "w": int(bbox.get("w", 0)),
                "h": int(bbox.get("h", 0)),
            }

        x1, y1, x2, y2 = bbox[:4]
        return {
            "x": int(x1),
            "y": int(y1),
            "w": max(0, int(x2 - x1)),
            "h": max(0, int(y2 - y1)),
        }

    def save_face_crop(self, photo, face) -> Optional[str]:
        """按检测框裁剪单张人脸，供前端按人物聚合展示。"""
        try:
            img = Image.open(photo.path)
            img = ImageOps.exif_transpose(img)
            box = self._resolve_face_box(face)
            if box["w"] <= 0 or box["h"] <= 0:
                return None

            padding_x = max(8, int(box["w"] * 0.12))
            padding_y = max(8, int(box["h"] * 0.12))
            left = max(0, box["x"] - padding_x)
            top = max(0, box["y"] - padding_y)
            right = min(img.width, box["x"] + box["w"] + padding_x)
            bottom = min(img.height, box["y"] + box["h"] + padding_y)
            cropped = img.crop((left, top, right, bottom))

            output_filename = f"{photo.photo_id}_{face['person_id']}_{face['face_id'][:8]}.webp"
            output_path = os.path.join(self.face_dir, output_filename)
            cropped.save(output_path, "WEBP", quality=90, method=6)
            return output_path
        except Exception as e:
            photo.processing_errors["face_crop"] = str(e)
            return None

    def draw_face_boxes(self, photo) -> str:
        """
        在照片上画人脸框和标签，保存为boxed_images/

        Args:
            photo: 照片对象

        Returns:
            带框图片的保存路径，如果没有人脸则返回None
        """
        if not photo.faces:
            return None

        try:
            # 读取原图（保留EXIF）
            img = Image.open(photo.path)
            img = ImageOps.exif_transpose(img)
            exif = normalized_exif_bytes(img)
            img = img.convert("RGBA") if "A" in img.getbands() else img.convert("RGB")

            # 创建绘图对象
            draw = ImageDraw.Draw(img)

            # 尝试加载字体
            try:
                # macOS系统字体
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
            except:
                try:
                    # Linux系统字体
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
                except:
                    font = ImageFont.load_default()

            primary_person_id = getattr(photo, "primary_person_id", None)

            # 为每个人脸画框
            for face in photo.faces:
                person_id = face["person_id"]
                box = self._resolve_face_box(face)
                x = box["x"]
                y = box["y"]
                w = box["w"]
                h = box["h"]

                # 根据person_id选择颜色
                if primary_person_id and person_id == primary_person_id:
                    color = (255, 0, 0)  # 红色 - 主用户
                else:
                    color = (0, 0, 255)  # 蓝色 - 其他人物

                # 画框
                draw.rectangle([x, y, x+w, y+h], outline=color, width=3)

                # 画标签（框上方）
                label = person_id
                # 计算标签背景
                text_bbox = draw.textbbox((x, y - 35), label, font=font)
                draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
                draw.text((x, y - 35), label, fill=(255, 255, 255), font=font)

            # 保存（保留EXIF）
            output_filename = f"{os.path.splitext(photo.filename)[0]}_boxed.webp"
            output_path = os.path.join(self.boxed_dir, output_filename)

            save_kwargs = {"quality": 90, "method": 6}
            if exif:
                save_kwargs["exif"] = exif
            img.save(output_path, "WEBP", **save_kwargs)

            return output_path

        except Exception as e:
            print(f"警告：绘制人脸框失败 {photo.filename}: {e}")
            photo.processing_errors["draw_face_boxes"] = str(e)
            return None
