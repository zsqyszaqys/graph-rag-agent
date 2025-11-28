import re
import ast
import time
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

from graphrag_agent.config.prompts import system_template_build_index, user_template_build_index
from graphrag_agent.models.get_models import get_llm_model
from graphrag_agent.graph.core import connection_manager, timer, get_performance_stats, print_performance_stats
from graphrag_agent.config.settings import ENTITY_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS

class EntityMerger:
    """
     实体合并管理器，负责基于LLM决策合并相似实体。
    主要功能包括使用LLM分析实体相似性、解析合并建议，以及执行实体合并操作。
    """

    def __init__(self, batch_size:int = 20, max_workers:int = 4):
        """
        初始化实体合并管理器
        :param batch_size:批处理大小
        :param max_workers:并行工作线程数
        """

        # 初始化图数据库连接
        self.graph = connection_manager.get_connection()

        # 获取语言模型
        self.llm = get_llm_model()

        # 批处理与并行参数
        self.batch_size = batch_size or ENTITY_BATCH_SIZE
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS

        # 创建索引
        self._create_indexes()

        # 性能监控
        self.llm_time = 0
        self.db_time = 0
        self.parse_time = 0

    def _create_indexes(self) -> None:
        """创建必要的索引以优化查询性能"""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (e:`__Entity__`) ON (e.id)"
        ]

    def _setup_llm_chain(self)->None:
        """
        设置LLM处理链，用于实体合并决策
        包括创建提示模板和构建处理链
        :return:
        """
        # 检查模型能力
        if not hasattr(self.llm, 'with_structured_output')
            print("当前LLM模型不支持结构化输出")

        # 创建提示词模板
        system_message_prompt = SystemMessagePromptTemplate.format_messages(system_template_build_index)
        human_message_prompt = HumanMessagePromptTemplate.format_messages(user_template_build_index)

