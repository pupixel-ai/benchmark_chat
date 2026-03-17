#!/usr/bin/env python3
"""
记忆工程项目 - 主入口
"""
import os
import sys
import json
import argparse
from datetime import datetime
from typing import List

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

from config import *
from memory_module import MemoryModuleService
from services.image_processor import ImageProcessor
from services.face_recognition import FaceRecognition
from services.vlm_analyzer import VLMAnalyzer
from services.llm_processor import LLMProcessor
from utils import save_json


def show_progress(current: int, total: int, message: str = ""):
    """显示进度条"""
    if not SHOW_PROGRESS:
        return

    percent = (current / total) * 100
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)

    print(f"\r处理中... [{bar}] {current}/{total} ({percent:.0f}%) {message}", end='', flush=True)


def print_simple_summary(
    events: List,
    relationships: List,
    profile_path: str = None,
    primary_person_id: str = None,
):
    """打印简洁版摘要（终端输出）"""
    print("\n\n" + "=" * 50)
    print("记忆提取完成")
    print("=" * 50)

    # 事件
    print(f"\n事件：{len(events)}个")
    for event in events:
        print(f"  ├─ {event.date} {event.time_range}: {event.title} (置信度: {event.confidence:.0%})")

    # 人物关系
    print(f"\n人物关系：{len(relationships)}个")
    for rel in relationships:
        print(f"  ├─ {rel.person_id}: {rel.label} (置信度: {rel.confidence:.0%})")

    # 用户画像
    print("\n用户画像：")
    if profile_path:
        print(f"  画像报告已生成: {profile_path}")
    else:
        print(f"  画像生成失败")

    if primary_person_id:
        print(f"\n主用户ID：{primary_person_id}")

    # 输出路径
    print(f"\n结果已保存到: {OUTPUT_PATH}")
    print(f"详细报告已保存到: {DETAILED_OUTPUT_PATH}")
    print(f"VLM缓存已保存到: {VLM_CACHE_PATH}")
    print(f"人脸识别输出已保存到: {FACE_OUTPUT_PATH}")


def save_detailed_report(events: List, relationships: List, face_output: dict):
    """保存详细版报告（Markdown格式）- 不包含用户画像"""
    report = []
    report.append("# 记忆工程 - 详细报告\n")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    primary_person_id = face_output.get("primary_person_id")
    metrics = face_output.get("metrics", {})
    report.append("## 人脸识别摘要\n")
    report.append(f"- **主用户 ID**: {primary_person_id or '未识别'}\n")
    report.append(f"- **图片数**: {metrics.get('total_images', 0)}\n")
    report.append(f"- **人脸数**: {metrics.get('total_faces', 0)}\n")
    report.append(f"- **人物数**: {metrics.get('total_persons', 0)}\n\n")

    # 事件详情
    report.append("## 事件详情\n")
    for i, event in enumerate(events, 1):
        report.append(f"### {i}. {event.title}\n")
        report.append(f"- **时间**: {event.date} {event.time_range}（{event.duration}）\n")
        report.append(f"- **类型**: {event.type}\n")
        report.append(f"- **地点**: {event.location}\n")
        report.append(f"- **参与者**: {', '.join(event.participants)}\n")
        if event.evidence_photos:
            report.append(f"- **证据照片**: {', '.join(event.evidence_photos)}\n")
        report.append(f"- **照片数**: {event.photo_count}张\n")
        report.append(f"- **描述**: {event.description}\n")
        if event.narrative:
            report.append(f"- **客观叙事**: {event.narrative}\n")
        if event.social_interaction:
            core_ids = event.social_interaction.get('core_person_ids', [])
            if core_ids:
                report.append(f"- **核心同伴**: {', '.join(core_ids)}\n")
            details = event.social_interaction.get('interaction_details', '')
            if details:
                report.append(f"- **互动详情**: {details}\n")
        if event.lifestyle_tags:
            report.append(f"- **生活标签**: {', '.join(event.lifestyle_tags)}\n")

        # 新增：社交切片
        if event.social_slices:
            report.append(f"- **社交切片**:\n")
            for slice in event.social_slices:
                report.append(f"  - {slice.get('person_id', '未知')}: {slice.get('interaction', '')} | {slice.get('relationship', '')} ({slice.get('confidence', 0):.0%})\n")

        # 新增：人格/性格/审美线索
        if event.persona_evidence:
            report.append(f"- **人格/性格/审美线索**:\n")
            evidence = event.persona_evidence
            if evidence.get('behavioral'):
                report.append(f"  - 行为特征: {', '.join(evidence['behavioral'])}\n")
            if evidence.get('aesthetic'):
                report.append(f"  - 审美: {', '.join(evidence['aesthetic'])}\n")
            if evidence.get('other'):
                report.append(f"  - 其他: {', '.join(evidence['other'])}\n")

        report.append(f"- **置信度**: {event.confidence:.0%}\n")
        report.append(f"- **推理依据**: {event.reason}\n\n")

    # 人物关系详情
    report.append("## 人物关系详情\n")
    for i, rel in enumerate(relationships, 1):
        report.append(f"### {i}. {rel.person_id} - {rel.label}\n")
        report.append(f"- **置信度**: {rel.confidence:.0%}\n")
        report.append(f"- **证据**: 共同出现{rel.evidence['photo_count']}次，时间跨度{rel.evidence['time_span']}\n")
        report.append(f"- **场景**: {', '.join(rel.evidence['scenes'])}\n")
        report.append(f"- **推理**: {rel.reason}\n\n")

    # 保存
    with open(DETAILED_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(report)


def save_final_output(
    events: List,
    relationships: List,
    face_output: dict,
    memory_output: dict = None,
    profile_markdown: str = "",
    primary_person_id: str = None,
):
    """保存最终输出（JSON格式）- 不包含用户画像（画像单独保存为Markdown）"""
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_events": len(events),
            "total_relationships": len(relationships),
            "models": {
                "vlm": VLM_MODEL,
                "llm": LLM_MODEL,
            },
            "primary_person_id": primary_person_id,
        },
        "events": [
            {
                "event_id": e.event_id,
                "date": e.date,
                "time_range": e.time_range,
                "duration": e.duration,
                "title": e.title,
                "type": e.type,
                "participants": e.participants,
                "location": e.location,
                "description": e.description,
                "photo_count": e.photo_count,
                "confidence": e.confidence,
                "reason": e.reason,
                "narrative": e.narrative,
                "social_interaction": e.social_interaction,
                "evidence_photos": e.evidence_photos,
                "lifestyle_tags": e.lifestyle_tags,
                "social_slices": e.social_slices,
                "persona_evidence": e.persona_evidence
            }
            for e in events
        ],
        "relationships": [
            {
                "person_id": r.person_id,
                "relationship_type": r.relationship_type,
                "label": r.label,
                "confidence": r.confidence,
                "evidence": r.evidence,
                "reason": r.reason
            }
            for r in relationships
        ],
        "face_recognition": face_output,
        "profile_markdown": profile_markdown,
        "memory": memory_output or {},
    }

    save_json(output, OUTPUT_PATH)


def save_profile_report(profile_markdown: str):
    """保存用户画像报告（Markdown格式）"""
    with open(PROFILE_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(profile_markdown)
    return PROFILE_REPORT_PATH


def main():
    """主流程"""
    parser = argparse.ArgumentParser(description='记忆工程项目 - 从相册提取记忆和画像')
    parser.add_argument('--photos', type=str, required=True, help='照片目录路径')
    parser.add_argument('--max-photos', type=int, default=MAX_PHOTOS, help='最多处理多少张照片')
    parser.add_argument('--use-cache', action='store_true', help='使用VLM缓存（跳过VLM分析）')

    args = parser.parse_args()

    print("=" * 50)
    print("记忆工程项目")
    print("=" * 50)

    # Step 1: 加载照片
    print("\n[1/9] 加载照片...")
    image_processor = ImageProcessor()
    photos = image_processor.load_photos(args.photos, args.max_photos)
    print(f"  加载了 {len(photos)} 张照片")

    # Step 2: 转换HEIC到全尺寸JPEG（用于人脸识别）
    print("\n[2/9] 转换HEIC到JPEG...")
    photos = image_processor.convert_to_jpeg(photos)
    print(f"  转换完成")

    # Step 3: 人脸识别前去重
    print("\n[3/9] 人脸识别前去重...")
    face_input_photos = image_processor.dedupe_before_face_recognition(photos)
    print(f"  去重后保留 {len(face_input_photos)} / {len(photos)} 张")

    # Step 3: 人脸识别
    print("\n[4/9] 人脸识别...")
    face_rec = FaceRecognition()

    for i, photo in enumerate(face_input_photos):
        show_progress(i + 1, len(face_input_photos), f"人脸识别")
        face_rec.process_photo(photo)

    face_db = face_rec.get_all_persons()
    face_output = face_rec.get_face_output()
    print(f"\n  识别了 {face_output['metrics']['total_persons']} 个人物")

    # Step 4: 确定主用户 + 画框
    print("\n[5/9] 确定主用户并绘制人脸框...")
    face_rec.reorder_protagonist(face_input_photos)
    face_db = face_rec.get_all_persons()
    face_output = face_rec.get_face_output()

    primary_person_id = face_rec.get_primary_person_id()
    primary_person = face_db.get(primary_person_id) if primary_person_id else None
    if primary_person_id and primary_person:
        print(f"  主用户：{primary_person_id}（出现 {primary_person.photo_count} 次）")

    # 绘制人脸框
    boxed_count = 0
    for photo in face_input_photos:
        if photo.faces:
            photo.primary_person_id = primary_person_id
            boxed_path = image_processor.draw_face_boxes(photo)
            if boxed_path:
                photo.boxed_path = boxed_path
                boxed_count += 1
    print(f"  绘制了 {boxed_count} 张带框图片")

    # Step 5: 压缩照片（用于VLM）
    print("\n[6/9] 压缩照片...")
    photos = image_processor.preprocess(face_input_photos)
    print(f"  压缩完成")

    # Step 6: VLM分析
    vlm = VLMAnalyzer()

    if args.use_cache and vlm.load_cache():
        print("\n[7/9] 使用VLM缓存（跳过VLM分析）...")
    else:
        print("\n[7/9] VLM分析（这可能需要30-40分钟）...")

        for i, photo in enumerate(photos):
            show_progress(i + 1, len(photos), f"VLM分析")

            if SHOW_PHOTO_DETAILS:
                print(f"\n  [{i+1}/{len(photos)}] {photo.filename}")
                print(f"    人脸: {', '.join([f['person_id'] for f in photo.faces])}")

            result = vlm.analyze_photo(photo, face_db, primary_person_id)

            if result:
                vlm.add_result(photo, result)

        # 保存VLM缓存
        vlm.save_cache()
        print("\n  VLM分析完成，结果已缓存")

    # Step 7: LLM处理
    print("\n[8/9] LLM处理（事件提取、关系推断、画像生成）...")
    events = []
    relationships = []
    profile_markdown = ""
    profile_path = None

    if not vlm.results:
        print("  - VLM结果为空，跳过LLM处理，仅保存人脸识别结果")
    else:
        llm = LLMProcessor()

        # 提取事件
        print("  - 提取事件...")
        events = llm.extract_events(vlm.results, primary_person_id)
        print(f"    提取了 {len(events)} 个事件")

        # 推断关系
        print("  - 推断人物关系...")
        relationships = llm.infer_relationships(vlm.results, face_db, primary_person_id)
        print(f"    推断了 {len(relationships)} 个关系")

        # 生成画像
        print("  - 生成用户画像（使用 Flash 2.5）...")
        profile_markdown = llm.generate_profile(events, relationships, primary_person_id)
        if profile_markdown:
            profile_path = save_profile_report(profile_markdown)
            print(f"    画像报告已保存: {profile_path}")
        else:
            print("    画像生成失败")

    # Step 8: Memory 框架物化
    print("\n[8/9] 生成 Memory Framework 输出...")
    memory_output = MemoryModuleService(
        task_id="cli_run",
        task_dir=PROJECT_ROOT,
        pipeline_version=APP_VERSION,
    ).materialize(
        photos=photos,
        face_output=face_output,
        vlm_results=vlm.results,
        events=events,
        relationships=relationships,
        profile_markdown=profile_markdown,
    )

    # Step 9: 保存结果
    print("\n[9/9] 保存结果...")
    save_final_output(
        events,
        relationships,
        face_output,
        memory_output=memory_output,
        profile_markdown=profile_markdown,
        primary_person_id=primary_person_id,
    )
    save_detailed_report(events, relationships, face_output)

    # 输出摘要
    print_simple_summary(events, relationships, profile_path, primary_person_id)


if __name__ == "__main__":
    main()
