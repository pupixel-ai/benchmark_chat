#!/usr/bin/env python3
"""
测试脚本：随机10张照片 → HEIC转JPEG → 人脸识别
复用现有服务，验证人脸识别效果
"""
import os
import sys
import random
from datetime import datetime
from typing import List

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

from config import CACHE_DIR, FACE_OUTPUT_PATH, PROJECT_ROOT
from services.image_processor import ImageProcessor
from services.face_recognition import FaceRecognition
from models import Photo
from utils import save_json


def print_header(title: str):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run_test_pipeline(max_samples: int = 10, data_dir: str = None):
    """
    运行测试流程：加载 → 随机采样 → HEIC转JPEG → 人脸识别

    Args:
        max_samples: 要测试的样本数（默认10）
        data_dir: 数据目录（默认 data/raw）
    """
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data", "raw")

    print_header(f"[1/4] 加载照片")
    print(f"从 {data_dir} 加载所有照片...")

    # Step 1: 加载所有照片
    image_processor = ImageProcessor()
    all_photos = image_processor.load_photos(data_dir, max_photos=None)
    print(f"✓ 加载完成：共 {len(all_photos)} 张照片\n")

    if len(all_photos) == 0:
        print("错误：没有找到照片！")
        return

    # Step 2: 随机采样
    print_header(f"[2/4] 随机采样 ({max_samples} 张)")
    sample_size = min(max_samples, len(all_photos))
    selected_photos = random.sample(all_photos, sample_size)

    # 按时间戳排序（便于查看）
    selected_photos.sort(key=lambda p: p.timestamp)

    print(f"✓ 已随机选择 {sample_size} 张照片:\n")
    for i, photo in enumerate(selected_photos, 1):
        location_str = photo.location.get('name', '未知位置') if photo.location else '未知位置'
        print(f"  {i:2}. {photo.filename}")
        print(f"       时间: {photo.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"       位置: {location_str}\n")

    # Step 3: HEIC转JPEG
    print_header(f"[3/4] HEIC转JPEG")
    converted_photos = image_processor.convert_to_jpeg(selected_photos)
    heic_count = sum(1 for p in converted_photos if p.original_path and p.original_path.lower().endswith(('.heic', '.heif')))
    print(f"✓ 转换完成：{heic_count} 张HEIC已转换为JPEG\n")

    # Step 4: 人脸识别
    print_header(f"[4/4] 人脸识别")
    face_recognition = FaceRecognition()
    detected_faces = {}
    person_ids = set()

    for i, photo in enumerate(converted_photos, 1):
        print(f"\n处理 [{i}/{len(converted_photos)}] {photo.filename}...", end="", flush=True)

        try:
            faces = face_recognition.process_photo(photo)
            detected_faces[photo.photo_id] = {
                'filename': photo.filename,
                'timestamp': photo.timestamp.isoformat(),
                'location': photo.location.get('name', '') if photo.location else '',
                'face_count': len(faces),
                'persons': [f['person_id'] for f in faces],
                'scores': [f.get('score') for f in faces]
            }

            if faces:
                print(f" ✓ 检测到 {len(faces)} 张人脸", end="")
                for face in faces:
                    person_ids.add(face['person_id'])
                    print(f" [{face['person_id']}: {face.get('score', 0):.2f}]", end="")
            else:
                print(f" ○ 未检测到人脸", end="")
        except Exception as e:
            print(f" ✗ 错误: {e}", end="")
            detected_faces[photo.photo_id] = {
                'filename': photo.filename,
                'timestamp': photo.timestamp.isoformat(),
                'error': str(e)
            }

    # 保存人脸库
    print(f"\n\n✓ 保存人脸库...", end="", flush=True)
    face_recognition.save()
    print(" 完成")

    # 统计结果
    print_header("测试结果摘要")

    print(f"📊 照片统计:")
    print(f"  • 总数: {len(converted_photos)} 张")
    print(f"  • 有人脸: {sum(1 for v in detected_faces.values() if v.get('face_count', 0) > 0)} 张")
    print(f"  • 无人脸: {sum(1 for v in detected_faces.values() if v.get('face_count', 0) == 0)} 张")

    total_faces = sum(v.get('face_count', 0) for v in detected_faces.values())
    print(f"\n👤 人脸统计:")
    print(f"  • 检测到的人脸: {total_faces} 张")
    print(f"  • 识别出的人物: {len(person_ids)} 个")

    # 人物详情
    all_persons = face_recognition.get_all_persons()
    print(f"\n📋 识别出的人物:")
    for person_id in sorted(person_ids):
        person = all_persons.get(person_id)
        if person:
            print(f"  • {person_id}")
            print(f"      出现次数: {person.photo_count}")
            print(f"      首次看到: {person.first_seen.strftime('%Y-%m-%d %H:%M:%S') if person.first_seen else 'N/A'}")
            print(f"      最后看到: {person.last_seen.strftime('%Y-%m-%d %H:%M:%S') if person.last_seen else 'N/A'}")
            print(f"      平均检测分: {person.avg_confidence:.2f}")

    # 保存详细结果
    output_file = os.path.join(CACHE_DIR, "test_pipeline_result.json")
    save_json(detected_faces, output_file)
    print(f"\n✓ 详细结果已保存到: {output_file}")

    # 缓存信息
    print(f"\n📁 缓存信息:")
    print(f"  • 人脸输出: {FACE_OUTPUT_PATH}")
    print(f"  • JPEG图片: {os.path.join(CACHE_DIR, 'jpeg_images/')}")
    print(f"  • 测试结果: {output_file}")

    print(f"\n✅ 测试完成！可以继续执行主流程进行VLM分析和事件提取。")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="随机10张图片运行至人脸识别")
    parser.add_argument("--samples", type=int, default=10, help="采样照片数（默认10）")
    parser.add_argument("--data-dir", type=str, default=None, help="数据目录（默认data/raw）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（用于可重现性）")

    args = parser.parse_args()

    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)

    try:
        run_test_pipeline(max_samples=args.samples, data_dir=args.data_dir)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
