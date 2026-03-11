#!/usr/bin/env python3
"""
测试脚本：随机10张照片 → 人脸识别 → 压缩 → VLM分析
完整流程验证，可选保存结果
"""
import os
import sys
import random
from datetime import datetime
from typing import List, Dict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

from config import (
    API_PROXY_KEY,
    API_PROXY_URL,
    CACHE_DIR,
    FACE_OUTPUT_PATH,
    GEMINI_API_KEY,
    PROJECT_ROOT,
    USE_API_PROXY,
)
from services.image_processor import ImageProcessor
from services.face_recognition import FaceRecognition
from services.vlm_analyzer import VLMAnalyzer
from models import Photo
from utils import save_json, load_json


def print_header(title: str):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def check_api_key():
    """检查当前模式所需的 API 配置"""
    if USE_API_PROXY:
        if API_PROXY_URL and API_PROXY_KEY:
            return

        print("\n❌ 错误：代理模式下缺少 API_PROXY_URL 或 API_PROXY_KEY")
        print("\n请在 .env 中配置：")
        print("   USE_API_PROXY=true")
        print("   API_PROXY_URL=https://your-proxy-host.example.com")
        print("   API_PROXY_KEY=your_proxy_api_key_here")
        sys.exit(1)

    if GEMINI_API_KEY:
        return

    print("\n❌ 错误：未检测到 GEMINI_API_KEY")
    print("\n需要配置步骤:")
    print("1. 访问 https://makersuite.google.com/app/apikey")
    print("2. 创建或复制你的 API Key")
    print("3. 在项目根目录创建 .env 文件：")
    print("   GEMINI_API_KEY=your_api_key_here")
    print("\n或者设置环境变量：")
    print("   export GEMINI_API_KEY=your_api_key_here")
    sys.exit(1)


def run_pipeline_with_vlm(max_samples: int = 10, data_dir: str = None, save_results: bool = True):
    """
    运行完整管道：加载 → 随机采样 → HEIC转JPEG → 人脸识别 → 压缩 → VLM分析

    Args:
        max_samples: 要处理的样本数（默认10）
        data_dir: 数据目录（默认 data/raw）
        save_results: 是否保存 VLM 缓存
    """
    # 检查 API Key
    check_api_key()

    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data", "raw")

    # ============ [1/7] 加载照片 ============
    print_header("[1/7] 加载照片")
    print(f"从 {data_dir} 加载所有照片...")

    image_processor = ImageProcessor()
    all_photos = image_processor.load_photos(data_dir, max_photos=None)
    print(f"✓ 加载完成：共 {len(all_photos)} 张照片\n")

    if len(all_photos) == 0:
        print("❌ 错误：没有找到照片！")
        return

    # ============ [2/7] 随机采样 ============
    print_header(f"[2/7] 随机采样 ({max_samples} 张)")
    sample_size = min(max_samples, len(all_photos))
    selected_photos = random.sample(all_photos, sample_size)
    selected_photos.sort(key=lambda p: p.timestamp)

    print(f"✓ 已随机选择 {sample_size} 张照片:\n")
    for i, photo in enumerate(selected_photos, 1):
        location_str = photo.location.get('name', '未知位置') if photo.location else '未知位置'
        print(f"  {i:2}. {photo.filename:30} | {photo.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {location_str}")

    # ============ [3/7] HEIC转JPEG ============
    print_header("[3/7] HEIC转JPEG")
    converted_photos = image_processor.convert_to_jpeg(selected_photos)
    heic_count = sum(1 for p in converted_photos if p.original_path and p.original_path.lower().endswith(('.heic', '.heif')))
    print(f"✓ 转换完成：{heic_count} 张HEIC已转换为JPEG\n")

    # ============ [4/7] 人脸识别 + 确定主用户 ============
    print_header("[4/7] 人脸识别")
    face_recognition = FaceRecognition()
    detected_count = 0

    for i, photo in enumerate(converted_photos, 1):
        print(f"  [{i}/{len(converted_photos)}] {photo.filename:30}", end="", flush=True)
        try:
            faces = face_recognition.process_photo(photo)
            if faces:
                detected_count += len(faces)
                print(f" ✓ 检测到 {len(faces)} 张人脸")
            else:
                print(f" ○ 未检测到人脸")
        except Exception as e:
            print(f" ✗ 错误: {e}")

    face_db = face_recognition.get_all_persons()
    print(f"\n✓ 人脸识别完成：检测到 {detected_count} 张人脸，识别出 {len(face_db)} 个人物\n")

    # 确定主用户
    print("  确定主用户...", end="", flush=True)
    face_recognition.reorder_protagonist(converted_photos)
    face_db = face_recognition.get_all_persons()
    primary_person_id = face_recognition.get_primary_person_id()
    protagonist = face_db.get(primary_person_id) if primary_person_id else None
    if protagonist and primary_person_id:
        print(f" 主用户：{primary_person_id}（出现 {protagonist.photo_count} 次）")
    else:
        print(" 无主用户")

    # 绘制人脸框
    print("  绘制人脸框...", end="", flush=True)
    boxed_count = 0
    for photo in converted_photos:
        if photo.faces:
            photo.primary_person_id = primary_person_id
            boxed_path = image_processor.draw_face_boxes(photo)
            if boxed_path:
                photo.boxed_path = boxed_path
                boxed_count += 1
    print(f" {boxed_count} 张完成\n")

    # ============ [5/7] 压缩照片 ============
    print_header("[5/7] 压缩照片（用于VLM）")
    print("  压缩中...", end="", flush=True)
    converted_photos = image_processor.preprocess(converted_photos)
    print(" 完成\n")

    # ============ [6/7] VLM分析 ============
    print_header("[6/7] VLM分析")
    print(f"使用模型：Gemini 2.0 Flash")
    print(f"处理照片数：{len(converted_photos)}\n")

    vlm = VLMAnalyzer()
    vlm_results_summary = {}
    vlm_success_count = 0
    vlm_fail_count = 0

    for i, photo in enumerate(converted_photos, 1):
        print(f"  [{i}/{len(converted_photos)}] {photo.filename:30}", end="", flush=True)

        try:
            result = vlm.analyze_photo(photo, face_db, primary_person_id)

            if result:
                vlm_success_count += 1
                vlm.add_result(photo, result)

                # 记录摘要信息
                vlm_results_summary[photo.photo_id] = {
                    'filename': photo.filename,
                    'status': 'success',
                    'summary': result.get('summary', '')[:100] + '...' if result.get('summary') else '',
                    'activity': result.get('event', {}).get('activity', ''),
                    'location': result.get('scene', {}).get('location_detected', ''),
                    'faces_detected': len(photo.faces)
                }
                print(f" ✓ 分析完成")
            else:
                vlm_fail_count += 1
                vlm_results_summary[photo.photo_id] = {
                    'filename': photo.filename,
                    'status': 'failed',
                    'reason': 'Gemini 返回空结果'
                }
                print(f" ⚠ 分析失败")

        except Exception as e:
            vlm_fail_count += 1
            vlm_results_summary[photo.photo_id] = {
                'filename': photo.filename,
                'status': 'error',
                'error': str(e)[:100]
            }
            print(f" ✗ 错误: {str(e)[:50]}")

    print(f"\n✓ VLM分析完成")
    print(f"  成功：{vlm_success_count} 张")
    print(f"  失败：{vlm_fail_count} 张\n")

    # ============ [7/7] 保存结果 ============
    print_header("[7/7] 保存结果")

    if save_results:
        # 保存 VLM 缓存
        print("  保存 VLM 缓存...", end="", flush=True)
        vlm.save_cache()
        print(f" 完成")

        # 保存测试结果摘要
        test_result_file = os.path.join(CACHE_DIR, "test_pipeline_vlm_result.json")
        print(f"  保存 VLM 结果摘要...", end="", flush=True)
        save_json(vlm_results_summary, test_result_file)
        print(f" 完成")

    # ============ 最终统计 ============
    print_header("📊 完整流程统计")

    print("✅ 照片处理")
    print(f"  • 加载: {len(all_photos)} 张")
    print(f"  • 采样: {len(selected_photos)} 张")
    print(f"  • 有人脸: {sum(1 for p in converted_photos if p.faces)} 张")
    print(f"  • 无人脸: {sum(1 for p in converted_photos if not p.faces)} 张")

    print(f"\n👤 人脸识别")
    print(f"  • 检测人脸: {detected_count} 张")
    print(f"  • 识别人物: {len(face_db)} 个")
    if protagonist:
        print(f"  • 主用户: {primary_person_id}（{protagonist.photo_count} 次）")

    print(f"\n🤖 VLM分析")
    print(f"  • 成功: {vlm_success_count} 张")
    print(f"  • 失败: {vlm_fail_count} 张")
    print(f"  • 成功率: {vlm_success_count / len(converted_photos) * 100:.1f}%")

    # 人物信息
    print(f"\n📋 识别出的人物")
    for person_id in sorted(face_db.keys()):
        person = face_db[person_id]
        label = "【主用户】" if person_id == primary_person_id else ""
        print(f"  • {person_id} {label}")
        print(f"      出现: {person.photo_count} 次 | 平均检测分: {person.avg_confidence:.2f}")

    # 缓存信息
    print(f"\n📁 缓存和输出")
    print(f"  • 人脸输出: {FACE_OUTPUT_PATH}")
    print(f"  • VLM缓存: {os.path.join(CACHE_DIR, 'Vyoyo.json')}")
    print(f"  • 测试结果: {os.path.join(CACHE_DIR, 'test_pipeline_vlm_result.json')}")
    print(f"  • JPEG图片: {os.path.join(CACHE_DIR, 'jpeg_images/')}")
    print(f"  • 压缩图片: {os.path.join(CACHE_DIR, 'compressed_images/')}")
    print(f"  • 带框图片: {os.path.join(CACHE_DIR, 'boxed_images/')}")

    print(f"\n✅ 完整流程运行完成！")
    print(f"\n下一步可以运行主流程进行事件提取和关系推断：")
    print(f"   python3 main.py --photos data/raw --max-photos {len(selected_photos)} --use-cache")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="随机N张图片完整管道测试（含VLM分析）")
    parser.add_argument("--samples", type=int, default=10, help="采样照片数（默认10）")
    parser.add_argument("--data-dir", type=str, default=None, help="数据目录（默认data/raw）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（用于可重现性）")
    parser.add_argument("--no-save", action="store_true", help="不保存VLM缓存")

    args = parser.parse_args()

    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)

    try:
        run_pipeline_with_vlm(
            max_samples=args.samples,
            data_dir=args.data_dir,
            save_results=not args.no_save
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
