import os
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from graphrag_agent.config.settings import FILE_REGISTRY_PATH

class FileChangeManager:
    """
    文件变更管理器，负责追踪文件的变更状态。

    主要功能：
    1. 扫描文件目录，计算文件哈希值
    2. 与历史记录比较，识别变更的文件
    3. 更新文件注册表
    """

    def __init__(self, files_dir: str, registry_path: str = None):
        """
        初始化文件变更管理器

        Args:
            files_dir: 要监控的文件目录
            registry_path: 文件注册表保存路径，默认使用配置中的路径
        """
        if registry_path is None:
            registry_path = str(FILE_REGISTRY_PATH)

        self.files_dir = Path(files_dir)
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()
