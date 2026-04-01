#!/usr/bin/env python3
"""
LP3 独立运行脚本 - 从预计算数据生成新画像
使用官方 Google Gemini API（不通过代理）

使用方式:
  python3 scripts/run_lp3_fresh.py \
    --case-dir "/path/to/08_chenmeiyi" \
    --gemini-key "YOUR_OFFICIAL_API_KEY"
"""
import os
import sys
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class OfficialGeminiLLMProcessor:
    """官方 Google Gemini API 适配器"""

    def __init__(self, api_key: str, primary_person_id: str = None):
        self.api_key = api_key
        self.primary_person_id = primary_person_id
        self.model = "gemini-2.5-flash"
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        print(f"[LLM] 使用官方 Google Gemini API")
        print(f"[LLM] 模型: {self.model}")
        print(f"[LLM] 主角: {primary_person_id}")

    def _call_llm_via_official_api(self, prompt: str, response_mime_type: str = None, model_override: str = None) -> dict:
        """调用官方 Google Gemini API"""
        model = model_override or self.model
        url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        # 强制 JSON 输出
        if response_mime_type:
            payload["generationConfig"] = {
                "responseMimeType": response_mime_type
            }

        max_retries = 3
        timeout = 300

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=timeout
                )

                if response.status_code == 200:
                    response_data = response.json()

                    # 提取文本内容
                    if "candidates" in response_data and len(response_data["candidates"]) > 0:
                        candidate = response_data["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            for part in candidate["content"]["parts"]:
                                if "text" in part:
                                    try:
                                        # 尝试解析为 JSON
                                        result = json.loads(part["text"])
                                        return result
                                    except json.JSONDecodeError:
                                        # 如果是 Markdown JSON，尝试提取
                                        text = part["text"]
                                        import re
                                        json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
                                        if json_match:
                                            json_str = json_match.group(1)
                                            try:
                                                result = json.loads(json_str)
                                                return result
                                            except:
                                                pass
                                        # 返回原始文本
                                        return {"text": text}

                    return None
                else:
                    error_msg = f"Google API 返回状态码 {response.status_code}"
                    if response.text:
                        try:
                            error_data = response.json()
                            if isinstance(error_data, dict) and "error" in error_data:
                                error_info = error_data.get("error", {})
                                if isinstance(error_info, dict):
                                    error_msg += f": {error_info.get('message', str(error_info))}"
                                else:
                                    error_msg += f": {error_info}"
                        except:
                            error_msg += f": {response.text[:200]}"
                    raise Exception(error_msg)

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                # 超时或连接错误，进行重试
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避：2s, 4s
                    print(f"      [LLM重试 {attempt + 1}/{max_retries}] 等待 {wait_time}s 后重试...", flush=True)
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"LLM调用重试{max_retries}次仍失败: {e}")

    def _call_profile_via_official_api(self, prompt: str) -> str:
        """调用官方 API 生成画像（返回文本）"""
        model = self.model
        url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        timeout = 300
        response = requests.post(
            url,
            json=payload,
            timeout=timeout
        )

        if response.status_code == 200:
            response_data = response.json()
            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                candidate = response_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            return part["text"]

        raise Exception(f"Google API 返回错误: {response.status_code}")


def main():
    parser = argparse.ArgumentParser(
        description="LP3 独立生成新画像 - 从预计算数据（官方 Gemini API）"
    )
    parser.add_argument(
        "--case-dir",
        type=str,
        required=True,
        help="包含预计算数据的目录（如 08_chenmeiyi）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认为 case_dir/offline_eval_output_fresh_run）",
    )
    parser.add_argument(
        "--gemini-key",
        type=str,
        required=True,
        help="官方 Gemini API Key",
    )

    args = parser.parse_args()

    case_path = Path(args.case_dir)
    if not case_path.exists():
        print(f"❌ 错误：case_dir 不存在: {case_path}")
        sys.exit(1)

    # 确定输出目录
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = case_path / f"offline_eval_output_fresh_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 LP3 独立生成新画像（官方 Gemini API）")
    print(f"   输入数据: {case_path}")
    print(f"   输出目录: {output_path}")
    print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 导入必要的模块
    try:
        from services.memory_pipeline.precomputed_loader import load_precomputed_memory_state
        from services.memory_pipeline.profile_fields import generate_structured_profile
        from utils import save_json
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        sys.exit(1)

    # 第1步：加载预计算的 MemoryState
    print("📥 [1/4] 加载预计算数据...")
    try:
        state = load_precomputed_memory_state(case_path)
        print(f"   ✓ 人脸库: {len(state.face_db)} 个人物")
        print(f"   ✓ VLM 结果: {len(state.vlm_results)} 张照片")
        print(f"   ✓ 事件: {len(state.events)} 个事件")
        print(f"   ✓ 关系: {len(state.relationships)} 个关系")
        print(f"   ✓ 主角: {state.primary_decision.get('primary_person_id', 'unknown')}\n")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 第2步：创建官方 API 处理器
    print("🤖 [2/4] 初始化 LLM 处理器（官方 Google API）...")
    try:
        primary_person_id = state.primary_decision.get('primary_person_id')
        llm_processor = OfficialGeminiLLMProcessor(args.gemini_key, primary_person_id=primary_person_id)
        print(f"   ✓ LLM 处理器初始化完成\n")
    except Exception as e:
        print(f"❌ LLM 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 第3步：运行 LP3 ProfileAgent
    print("🎯 [3/4] 运行 ProfileAgent 生成画像...")
    try:
        result = generate_structured_profile(state, llm_processor=llm_processor)
        structured = result.get("structured", {})
        field_decisions = result.get("field_decisions", [])
        consistency = result.get("consistency", {})

        print(f"   ✓ 生成完成")
        print(f"   ✓ 非空字段数: {_count_fields(structured)}")
        print(f"   ✓ null 字段数: {_count_null_fields(structured)}")
        print(f"   ✓ 决策数: {len(field_decisions)}\n")
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 第4步：保存完整输出
    print("💾 [4/4] 保存输出...")

    # 保存 structured profile
    profile_output = {
        "metadata": {
            "case_dir": str(case_path),
            "generated_at": datetime.now().isoformat(),
            "pipeline": "LP3 独立生成（官方 Gemini API）",
            "primary_person_id": state.primary_decision.get('primary_person_id'),
            "data_source": {
                "photos": len(state.vlm_results),
                "events": len(state.events),
                "people": len(state.face_db),
                "relationships": len(state.relationships),
            },
        },
        "structured": structured,
        "field_decisions": field_decisions,
        "consistency_report": consistency,
    }

    profile_path = output_path / "profile_fresh_run.json"
    save_json(profile_output, str(profile_path))
    print(f"   ✓ 完整输出: {profile_path.name}")

    # 保存人可读的 Markdown 报告
    md_path = output_path / "profile_fresh_run.md"
    _save_markdown_report(structured, state, md_path)
    print(f"   ✓ Markdown 报告: {md_path.name}")

    # 保存简要统计
    stats_path = output_path / "statistics.json"
    stats = {
        "total_photos": len(state.vlm_results),
        "total_events": len(state.events),
        "total_people": len(state.face_db),
        "relationships_count": len(state.relationships),
        "fields_with_value": _count_fields(structured),
        "fields_null": _count_null_fields(structured),
        "field_decisions": len(field_decisions),
    }
    save_json(stats, str(stats_path))
    print(f"   ✓ 统计数据: {stats_path.name}")

    print(f"\n✅ LP3 独立生成完成")
    print(f"   所有输出已保存到: {output_path}\n")

    return 0


def _count_fields(structured: Dict[str, Any]) -> int:
    """计算非 null 的字段数"""
    count = 0
    for category_name, category_data in structured.items():
        if isinstance(category_data, dict) and "facts" in category_data:
            for field_name, field_value in category_data.get("facts", {}).items():
                if field_value is not None and isinstance(field_value, dict):
                    val = field_value.get("value")
                    if val is not None:
                        count += 1
    return count


def _count_null_fields(structured: Dict[str, Any]) -> int:
    """计算 null 的字段数"""
    count = 0
    for category_name, category_data in structured.items():
        if isinstance(category_data, dict) and "facts" in category_data:
            for field_name, field_value in category_data.get("facts", {}).items():
                if field_value is None or (isinstance(field_value, dict) and field_value.get("value") is None):
                    count += 1
    return count


def _save_markdown_report(structured: Dict[str, Any], state: Any, output_path: Path):
    """生成人可读的 Markdown 报告"""
    lines = []
    lines.append("# 陈美伊 - 新画像生成报告（LP3 独立运行）\n\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    lines.append(f"## 数据源统计\n\n")
    lines.append(f"- 照片数: {len(state.vlm_results)}\n")
    lines.append(f"- 事件数: {len(state.events)}\n")
    lines.append(f"- 人物数: {len(state.face_db)}\n")
    lines.append(f"- 关系数: {len(state.relationships)}\n")
    lines.append(f"- 主角: {state.primary_decision.get('primary_person_id', 'unknown')}\n\n")

    for category_name, category_data in structured.items():
        if isinstance(category_data, dict):
            lines.append(f"## {category_name}\n\n")
            if "facts" in category_data:
                facts = category_data.get("facts", {})
                for field_name, field_value in facts.items():
                    if field_value is not None and isinstance(field_value, dict):
                        val = field_value.get("value")
                        if val is not None:
                            conf = field_value.get("confidence", "N/A")
                            lines.append(f"### {field_name}\n")
                            lines.append(f"- **值**: {_format_value(val)}\n")
                            lines.append(f"- **置信度**: {conf}\n")
                            if "reasoning" in field_value:
                                reasoning = field_value.get("reasoning")
                                if reasoning:
                                    lines.append(f"- **推理**: {reasoning[:500]}...\n" if len(str(reasoning)) > 500 else f"- **推理**: {reasoning}\n")
                            lines.append("\n")
                        else:
                            lines.append(f"- {field_name}: (null / upstream_blocked)\n")
                    else:
                        lines.append(f"- {field_name}: (null / upstream_blocked)\n")
                lines.append("\n")

    output_path.write_text("".join(lines), encoding="utf-8")


def _format_value(val: Any) -> str:
    """格式化值为可读字符串"""
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


if __name__ == "__main__":
    sys.exit(main())
