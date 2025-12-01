"""
检索结果适配器

将不同搜索工具的原始输出统一转换为RetrievalResult数据模型，便于多Agent管线消费。
"""

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence
import uuid

from graphrag_agent.agents.multi_agent.core.retrieval_result import (
    RetrievalMetadata,
    RetrievalResult,
)
