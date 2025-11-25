import time
import os
import pickle
import concurrent.futures
from typing import List, Tuple, Optional
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from graphrag_agent.graph.core import retry, generate_hash
from graphrag_agent.config.settings import MAX_WORKERS as DEFAULT_MAX_WORKERS, BATCH_SIZE as DEFAULT_BATCH_SIZE


class EntityRelationExtractor:
    """
    实体关系提取器，负责从文本中提取实体和关系。
    使用LLM分析文本块，生成结构化的实体和关系数据。
    """

    def __init__(self, llm, system_template, human_template,
                 entity_types: List[str], relationship_types: List[str],
                 cache_dir="./cache/graph", max_workers=4, batch_size=5):
        """
        初始化实体关系提取器

        :param llm:语言模型
        :param system_template:系统提示模板
        :param human_template:用户提示模板
        :param entity_types:实体类型列表
        :param relationship_types:关系类型列表
        :param cache_dir:缓存目录
        :param max_workers:并行工作线程数
        :param batch_size:批处理大小
        """
        self.llm = llm
        self.entity_types = entity_types
        self.relationship_types = relationship_types
        self.chat_history = []

        # 设置分隔符
        self.tuple_delimiter = " : "
        self.record_delimiter = "\n"
        self.completion_delimiter = "\n\n"

        # 创建提示模板
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        self.chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            MessagesPlaceholder("chat_history"),
            human_message_prompt
        ])

        # 创建处理链
        self.chain = self.chat_prompt | self.llm

        # 缓存设置
        self.cache_dir = cache_dir
        self.enable_cache = True

        # 确保缓存目录存在
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # 并行处理配置
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE

        # 缓存统计
        self.cache_hits = 0
        self.cache_misses = 0