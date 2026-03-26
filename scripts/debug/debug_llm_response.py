#!/usr/bin/env python3
"""LP3 调试脚本 - 检查 LLM 的原始响应内容。"""
import os
import sys
import json
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

def test_llm_call(api_key: str):
    """测试 LLM 对一个简单字段的推理"""

    # 模拟一个简单的 prompt（只包含一个字段）
    prompt = """# Role
你是结构化画像的字段裁决 agent。

# Domain: Foundation & Social Identity
Current fields: ['long_term_facts.identity.name']

# Reasoning Protocol
Step 1: 先基于字段级 COT 形成草案。
Step 2: Strong Evidence / Stats / Counter 作为证据权重与置信度加权依据，而不是硬门槛。
Step 3: 若反证更强或证据不足可直接输出 null；不要输出反思阶段结构。

# Field Units
### Field: long_term_facts.identity.name
Risk: P0
Strong Evidence:
- 文件信息（论文、订单、身份证等明确写明的姓名）
COT Steps:
- 先看该字段允许证据源里的主线信号，优先确认 文件信息（论文、订单、身份证等明确写明的姓名）
- 再确认这些信号稳定绑定主角本人，而不是同框人物、环境信息、截图内容或外部上下文
- 最后综合强证据、统计分布与反证强度做裁决，证据不足时保守输出 null
Owner Resolution:
- 先检查证据是否明确指向主角本人
- 如果证据只指向他人物品、他人关系、环境背景或截图文本，则不计入主角字段
Time Reasoning:
- 优先确认该字段对应的是长期模式还是短期变化
- 若无法区分长期/短期层级，则按更保守的层级处理或退回 null
Counter Evidence Checks:
- 检查是否存在与草案相反的证据没有被引用
- 检查是否存在明确的"改名"或"昵称变更"信号
Reflection Questions:
- 这个名字在整个记忆周期内是否保持一致？
- 有没有可能这是昵称、化名或笔名而非法定姓名？
Null Preferred:
- 证据不足时宁可输出 null

Evidence Summary:
Primary signals: event_012 提到"陈美伊"学号12518110，event_031、event_032 订单收货人"陈美伊"，photo_128 网易云音乐"_小陈"

Stats Summary:
- 姓名出现次数: 5次（高频率）
- 来源多样性: 高（论文、订单、音乐平台）
- 时间分布: 2025-2026跨度

Ownership Summary:
- 全部证据直接指向主角本人
- 无他人物品或背景信息

Counter Summary:
- 无相反证据

# Output Contract
严格输出 JSON:
{
  "fields": {
    "field_key": {
      "value": null,
      "confidence": 0.0,
      "reasoning": "",
      "supporting_ref_ids": [],
      "contradicting_ref_ids": [],
      "null_reason": null
    }
  }
}"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    print("📤 发送 prompt 到 Gemini API...")
    print(f"   URL: {url}")
    print(f"   Prompt 长度: {len(prompt)} 字符\n")

    response = requests.post(url, json=payload, timeout=300)

    print(f"📥 收到响应")
    print(f"   状态码: {response.status_code}\n")

    if response.status_code == 200:
        response_data = response.json()

        # 打印原始响应
        print("【原始响应（部分）】")
        print(json.dumps(response_data, indent=2, ensure_ascii=False)[:1500])

        # 提取文本内容
        if "candidates" in response_data and len(response_data["candidates"]) > 0:
            candidate = response_data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    if "text" in part:
                        text = part["text"]
                        print(f"\n【LLM 返回的文本内容】")
                        print(text)

                        # 尝试解析 JSON
                        try:
                            result = json.loads(text)
                            print(f"\n【解析后的 JSON】")
                            print(json.dumps(result, indent=2, ensure_ascii=False))
                        except:
                            print(f"\n⚠️  无法解析为 JSON")
    else:
        print(f"❌ API 调用失败: {response.text[:500]}")


if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("请先 export GEMINI_API_KEY 或 GOOGLE_API_KEY 再运行该脚本。")
    test_llm_call(api_key)
