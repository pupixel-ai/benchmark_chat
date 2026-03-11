"""
图片预处理模块
"""
import os
from datetime import datetime
from typing import List, Tuple, Optional
from PIL import Image, ExifTags, ImageDraw, ImageFont
from pillow_heif import register_heif_opener
import exifread
from models import Photo
from config import CACHE_DIR, MAX_IMAGE_SIZE, JPEG_QUALITY, DEDUP_TIME_WINDOW, AMAP_API_KEY, PROJECT_ROOT
from utils import compress_image, smart_deduplicate

# 注册HEIC格式支持
register_heif_opener()


class ImageProcessor:
    """图片处理器"""

    def __init__(self):
        self.compress_dir = os.path.join(CACHE_DIR, "compressed_images")
        self.jpeg_dir = os.path.join(CACHE_DIR, "jpeg_images")  # 全尺寸JPEG，用于人脸识别
        self.boxed_dir = os.path.join(CACHE_DIR, "boxed_images")  # 带人脸框的图片
        os.makedirs(self.compress_dir, exist_ok=True)
        os.makedirs(self.jpeg_dir, exist_ok=True)
        os.makedirs(self.boxed_dir, exist_ok=True)

    def load_photos(self, photo_dir: str, max_photos: int = None) -> List[Photo]:
        """
        加载照片目录中的所有照片

        Args:
            photo_dir: 照片目录路径
            max_photos: 最多加载多少张

        Returns:
            照片列表
        """
        photos = []
        supported_formats = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}

        # 遍历目录
        for filename in os.listdir(photo_dir):
            if max_photos and len(photos) >= max_photos:
                break

            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported_formats:
                continue

            path = os.path.join(photo_dir, filename)

            try:
                # 读取EXIF信息
                exif = self._read_exif(path)

                photo = Photo(
                    photo_id=f"photo_{len(photos) + 1:03d}",
                    filename=filename,
                    path=path,
                    timestamp=exif.get("datetime", datetime.now()),
                    location=exif.get("location", {})
                )

                photos.append(photo)

            except Exception as e:
                print(f"警告：无法读取照片 {filename}: {e}")
                continue

        # 按时间排序
        photos.sort(key=lambda p: p.timestamp)

        return photos

    def convert_to_jpeg(self, photos: List[Photo]) -> List[Photo]:
        """
        将HEIC转换为全尺寸JPEG（用于人脸识别），保留EXIF信息

        Args:
            photos: 照片列表

        Returns:
            转换后的照片列表
        """
        for photo in photos:
            # 只转换HEIC格式
            if not photo.filename.lower().endswith(('.heic', '.heif')):
                # 非HEIC格式，直接使用原路径
                photo.original_path = photo.path
                continue

            try:
                # 生成全尺寸JPEG文件名
                jpeg_filename = f"fullsize_{photo.photo_id}.jpg"
                jpeg_path = os.path.join(self.jpeg_dir, jpeg_filename)

                # 转换HEIC到JPEG（保持原尺寸和EXIF）
                image = Image.open(photo.path)

                # 保留EXIF信息
                exif = image.info.get('exif')
                if exif:
                    image.save(jpeg_path, "JPEG", quality=95, exif=exif)
                else:
                    image.save(jpeg_path, "JPEG", quality=95)

                # 保存原路径并更新为JPEG路径
                photo.original_path = photo.path
                photo.path = jpeg_path

            except Exception as e:
                print(f"警告：转换HEIC失败 {photo.filename}: {e}")
                photo.original_path = photo.path
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
                compressed_filename = f"compressed_{photo.photo_id}.jpg"
                compressed_path = os.path.join(self.compress_dir, compressed_filename)

                # 压缩
                compress_image(photo.path, compressed_path, MAX_IMAGE_SIZE, JPEG_QUALITY)

                # 更新照片对象
                photo.compressed_path = compressed_path

            except Exception as e:
                print(f"警告：压缩照片 {photo.filename} 失败: {e}")
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

        # 如果EXIF中没有时间，使用文件修改时间
        if "datetime" not in exif_data:
            try:
                file_mtime = os.path.getmtime(image_path)
                exif_data["datetime"] = datetime.fromtimestamp(file_mtime)
            except:
                exif_data["datetime"] = datetime.now()

        return exif_data

    def _reverse_geocode(self, lng: float, lat: float) -> str:
        """
        逆地理编码：将经纬度转换为地址（到区级）

        Args:
            lng: 经度
            lat: 纬度

        Returns:
            地址字符串，如 "北京市海淀区"
        """
        if not AMAP_API_KEY:
            return ""

        try:
            import urllib.request
            import json

            url = f"https://restapi.amap.com/v3/geocode/regeo?key={AMAP_API_KEY}&location={lng},{lat}&extensions=base"

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
            exif = img.info.get('exif')

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
                bbox_xywh = face.get("bbox_xywh")
                if bbox_xywh:
                    x = bbox_xywh["x"]
                    y = bbox_xywh["y"]
                    w = bbox_xywh["w"]
                    h = bbox_xywh["h"]
                else:
                    bbox = face.get("bbox", [0, 0, 0, 0])
                    if isinstance(bbox, dict):
                        x = bbox.get("x", 0)
                        y = bbox.get("y", 0)
                        w = bbox.get("w", 0)
                        h = bbox.get("h", 0)
                    else:
                        x1, y1, x2, y2 = bbox[:4]
                        x = int(x1)
                        y = int(y1)
                        w = max(0, int(x2 - x1))
                        h = max(0, int(y2 - y1))

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
            output_filename = f"{os.path.splitext(photo.filename)[0]}_boxed.jpg"
            output_path = os.path.join(self.boxed_dir, output_filename)

            if exif:
                img.save(output_path, "JPEG", quality=95, exif=exif)
            else:
                img.save(output_path, "JPEG", quality=95)

            return output_path

        except Exception as e:
            print(f"警告：绘制人脸框失败 {photo.filename}: {e}")
            return None
