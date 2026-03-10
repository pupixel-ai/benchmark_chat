#!/usr/bin/env python3
"""
测试新的事件提取prompt v2.1
"""
import json
import os
import sys
from google import genai

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))
from services.llm_processor import LLMProcessor

# 读取VLM缓存
CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache", "vlm_results.json")


def main():
    # 读取VLM缓存
    print(f"读取VLM缓存: {CACHE_PATH}")
    with open(CACHE_PATH, 'r', encoding='utf-8') as f:
        vlm_data = json.load(f)

    vlm_results = vlm_data['photos']
    print(f"共 {len(vlm_results)} 张照片的VLM分析结果")

    # 获取API密钥
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n错误：未设置 GEMINI_API_KEY 环境变量")
        return

    # 使用LLMProcessor
    print("\n调用 Gemini 2.0 Flash (新prompt v2.1)...")
    processor = LLMProcessor()

    try:
        events = processor.extract_events(vlm_results)

        print(f"\n✓ 成功提取 {len(events)} 个事件\n")

        # 打印事件摘要
        for i, event in enumerate(events, 1):
            print(f"【事件{i}】{event.event_id}")
            if event.meta_info.get("title"):
                print(f"  标题: {event.meta_info.get('title')}")
            if event.meta_info.get("timestamp"):
                print(f"  时间: {event.meta_info.get('timestamp')}")
            if event.meta_info.get("location_context"):
                print(f"  地点: {event.meta_info.get('location_context')}")
            if event.meta_info.get("photo_count"):
                print(f"  照片数: {event.meta_info.get('photo_count')}")
            if event.narrative_synthesis:
                print(f"  深度还原: {event.narrative_synthesis}")
            if event.objective_fact.get("scene_description"):
                desc = event.objective_fact["scene_description"]
                print(f"  客观描述: {desc[:80]}...")
            if event.tags:
                print(f"  标签: {' '.join(event.tags)}")
            if event.social_dynamics:
                print(f"  社交动态:")
                for sd in event.social_dynamics:
                    print(f"    - {sd.get('target_id')}: {sd.get('interaction_type')} ({sd.get('relation_hypothesis')})")
            if event.persona_evidence.get('socioeconomic'):
                print(f"  社经线索: {', '.join(event.persona_evidence['socioeconomic'])}")
            print()

        # 保存结果
        output_path = os.path.join(os.path.dirname(__file__), "output", "events_test_v3.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 转换为可序列化格式
        events_data = {
            "events": [
                {
                    "event_id": e.event_id,
                    "narrative_synthesis": e.narrative_synthesis,
                    "meta_info": e.meta_info,
                    "objective_fact": e.objective_fact,
                    "social_dynamics": e.social_dynamics,
                    "persona_evidence": e.persona_evidence,
                    "tags": e.tags,
                }
                for e in events
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(events_data, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_path}")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
