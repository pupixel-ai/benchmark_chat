"""
services package

避免在包导入阶段拉起重型依赖（InsightFace / MediaPipe / requests）。
需要具体服务时，请直接从对应模块导入。
"""

__all__: list[str] = []
