from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Agent 基类，定义通用功能和接口"""

    def __init__(self, chche_dir="./cache", memory_only=False):
        """
        初始化搜索工具
        :param chche_dir:缓存目录，用于存储搜索结果
        :param memory_only:
        """

        # 初始化普通LLM和流式LLM
