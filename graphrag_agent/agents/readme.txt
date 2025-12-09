这是一个非常庞大且架构严谨的系统。为了一步步完成这个模块的实现，我们将按照依赖关系自底向上构建：

核心与基类 (base.py, core/)：定义状态、数据模型和 Agent 基类。
简单 Agent 实现 (naive, graph, hybrid, deep)：实现基础功能的 Agent。
多智能体基础组件 (multi_agent/tools, multi_agent/core)：工具和数据结构。
多智能体功能模块 (planner, executor, reporter)：具体的业务逻辑。
多智能体集成 (integration)：编排器和门面。
高级 Agent (fusion_agent.py)：集大成者。
以下是具体的代码实现步骤。

第 1 步：核心数据模型与 Agent 基类
首先定义多智能体所需的核心状态和数据模型，以及所有 Agent 的父类。

第 2 步：简单 Agent 实现
实现 Naive、Graph 和 Hybrid Agent。


# graphrag_agent/agents/multi_agent/planner/base_planner.py

# graphrag_agent/agents/multi_agent/planner/task_decomposer.py

# graphrag_agent/agents/multi_agent/executor/worker_coordinator.py


# graphrag_agent/agents/multi_agent/reporter/section_writer.py

# graphrag_agent/agents/multi_agent/reporter/report_assembler.py

第 5 步：集成层与 Facade (integration)
这是将多智能体模块封装为 LangGraph 可调用对象的关键。

第 6 步：Fusion GraphRAG Agent (fusion_agent.py)
最后，实现最复杂的 Agent，它使用上述的 MultiAgentFacade。

Python

# graphrag_agent/agents/fusion_agent.py
from typing import AsyncGenerator
from .base import BaseAgent
from .multi_agent.integration.legacy_facade import MultiAgentFacade

class FusionGraphRAGAgent(BaseAgent):
    """
    Fusion GraphRAG Agent
    基于多 Agent 协作架构的增强型 GraphRAG Agent
    """

    def __init__(self):
        # 设置缓存目录
        self.cache_dir = "./cache/fusion_graphrag"

        # 初始化父类（虽然 Fusion 主要代理给 multi_agent，但保留基类特性）
        super().__init__(cache_dir=self.cache_dir)

        # 创建多智能体编排入口
        # 假设有个 CacheManager 类
        self.multi_agent = MultiAgentFacade(cache_manager=None)

    def _setup_graph(self):
        """
        Fusion Agent 覆盖此方法，因为它可能有完全不同的图结构，
        或者在此处将 self.graph 设置为 self.multi_agent.graph
        """
        self.graph = self.multi_agent.graph

    async def ask_stream(self, query: str, thread_id: str = "default",
                         recursion_limit: int = 5, show_thinking: bool = False) -> AsyncGenerator[str, None]:
        """
        覆盖流式方法以适配多智能体栈的事件输出
        """
        # 1. 快速缓存检查
        fast_result = self.check_fast_cache(query, thread_id)
        if fast_result:
            yield fast_result
            return

        # 2. 调用多智能体图
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}

        # 注意：这里的数据结构要匹配 PlanExecuteState
        initial_state = {"input": query, "past_steps": []}

        async for event in self.graph.astream_events(initial_state, config, version="v1"):
            # 处理多智能体的复杂事件流
            kind = event["event"]

            # 例如：显示规划阶段
            if kind == "on_chain_start" and event["name"] == "planner":
                if show_thinking:
                    yield "\nThinking: Creating plan...\n"

            # 显示生成的 Token
            elif kind == "on_chat_model_stream":
                 content = event["data"]["chunk"].content
                 if content:
                     yield content
第 7 步：模块导出 (__init__.py)
Python

# graphrag_agent/agents/__init__.py
from .base import BaseAgent
from .naive_rag_agent import NaiveRagAgent
from .graph_agent import GraphAgent
from .hybrid_agent import HybridAgent
from .deep_research_agent import DeepResearchAgent
from .fusion_agent import FusionGraphRAGAgent

__all__ = [
    "BaseAgent",
    "NaiveRagAgent",
    "GraphAgent",
    "HybridAgent",
    "DeepResearchAgent",
    "FusionGraphRAGAgent"
]
总结
这个实现方案遵循了如下核心思想：

利用继承 (BaseAgent)：减少了代码重复，统一了流式输出 (ask_stream) 和基础图构建逻辑。
LangGraph 驱动：所有的 Agent 本质上都是 StateGraph，这使得状态管理和持久化（Memory）变得标准化。
组合模式 (Facade)：FusionGraphRAGAgent 没有把所有逻辑写在一个文件里，而是作为 MultiAgentFacade 的包装器，后者协调了 Planner、Executor 和 Reporter。
关注点分离：
planner 负责把 Query 变成 Plan。
executor 负责具体的工具调用和证据收集。
reporter 负责文本生成和引用处理。
这套结构为复杂的知识检索和推理任务提供了极高的扩展性。