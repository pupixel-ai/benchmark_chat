"""
图片预处理模块
"""
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from PIL import Image, ExifTags, ImageDraw, ImageFont, ImageOps
from pillow_heif import register_heif_opener
import exifread
from models import Photo
from config import CACHE_DIR, MAX_IMAGE_SIZE, JPEG_QUALITY, DEDUP_TIME_WINDOW, AMAP_API_KEY, PROJECT_ROOT
from utils import compress_image, normalized_exif_bytes, smart_deduplicate

# 注册HEIC格式支持
register_heif_opener()


class ImageProcessor:
    """图片处理器"""

    def __init__(self, cache_dir: str = CACHE_DIR, amap_api_key: str = AMAP_API_KEY):
        self.cache_dir = cache_dir
        self.amap_api_key = amap_api_key
        self.compress_dir = os.path.join(cache_dir, "compressed_images")
        self.jpeg_dir = os.path.join(cache_dir, "jpeg_images")  # 全尺寸JPEG，用于人脸识别
        self.boxed_dir = os.path.join(cache_dir, "boxed_images")  # 带人脸框的图片
        self.face_dir = os.path.join(cache_dir, "face_crops")  # 人脸裁剪图
        os.makedirs(self.compress_dir, exist_ok=True)
        os.makedirs(self.jpeg_dir, exist_ok=True)
        os.makedirs(self.boxed_dir, exist_ok=True)
        os.makedirs(self.face_dir, exist_ok=True)

    def list_supported_photos(self, photo_dir: str) -> List[str]:
        supported_formats = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp'}
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
                    location=exif.get("location", {})
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

    def convert_to_jpeg(self, photos: List[Photo]) -> List[Photo]:
        """
        为人脸识别准备工作图。
        原始上传文件保持不变；仅在 HEIC 或带方向标签的图片上生成标准朝向的 JPEG 工作图。

        Args:
            photos: 照片列表

        Returns:
            转换后的照片列表
        """
        for photo in photos:
            photo.original_path = photo.path
            needs_working_copy = photo.filename.lower().endswith(('.heic', '.heif'))

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

                image = Image.open(photo.path)
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
                continue

        return photos

    def preprocess(self, photos: List[Photo]) -> List[Photo]:
        """
        预处理照片：压缩 + 去重

        Args:
            photos: 照片列表

        Returns:
            处理后的照片列表
        """
        # Step 1: 压缩
        compressed = self._compress_photos(photos)

        # Step 2: 暂时禁用去重（因为EXIF时间读取问题）
        # deduped = smart_deduplicate(compressed, DEDUP_TIME_WINDOW)
        deduped = compressed

        return deduped

    def _compress_photos(self, photos: List[Photo]) -> List[Photo]:
        """
        压缩所有照片

        Args:
            photos: 照片列表

        Returns:
            压缩后的照片列表
        """
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
                continue

        return photos

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
                datetime_str = str(tags.get('Image DateTime', ''))
                if datetime_str:
                    try:
                        exif_data["datetime"] = datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")
                    except:
                        pass

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
                    datetime_str = mapped.get("DateTimeOriginal") or mapped.get("DateTime")
                    if datetime_str:
                        try:
                            exif_data["datetime"] = datetime.strptime(str(datetime_str), "%Y:%m:%d %H:%M:%S")
                        except Exception:
                            pass

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
