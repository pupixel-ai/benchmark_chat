from __future__ import annotations

from typing import Any, List


def extract_events_from_state(state: Any, llm_processor: Any) -> List[Any]:
    """调用 LP1 事件提取。

    当前主策略严格遵循事件手册版本：
    - LP1 读取全量 VLM 结果一次调用
    - 不在 LLM 之前做事件簇聚合
    - 代码层只负责对返回事件做轻量规范化
    """

    raw_events = list(llm_processor.extract_events(state.vlm_results or []) or [])

    for index, event in enumerate(raw_events, start=1):
        event.event_id = f"EVT_{index:03d}"

    return raw_events
