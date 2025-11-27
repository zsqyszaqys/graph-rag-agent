import time
import concurrent.futures
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import Neo4jVector

from graphrag_agent.models.get_models import get_embeddings_model
from graphrag_agent.graph.core import BaseIndexer, connection_manager
from graphrag_agent.config.settings import CHUNK_BATCH_SIZE, MAX_WORKERS as DEFAULT_MAX_WORKERS

class ChunkIndexManager(BaseIndexer):
    """
     Chunk索引管理器，负责在Neo4j数据库中创建和管理文本块的向量索引。
    处理__Chunk__节点的embedding向量计算和索引创建，支持后续基于向量相似度的RAG查询。
    """

    def __init__(self, refresh_schema:bool = True, batch_size:int = 100, max_workers:int = 4):
        """
        初始化Chunk索引管理器

        :param refresh_schema:是否刷新Neo4j图数据库的schema
        :param batch_size:批处理大小
        :param max_workers:并行工作线程数
        """
        batch_size = batch_size or CHUNK_BATCH_SIZE
        max_workers = max_workers or DEFAULT_MAX_WORKERS

        super().__init__(batch_size, max_workers)

        # 初始化图数据库连接
        self.graph = connection_manager.get_connection()

        # 初始化嵌入模型
        self.embeddings = get_embeddings_model()

        # 创建索引
        self._create_indexes()

    def _create_indexes(self)->None:
        """创建必要的索引以优化查询性能"""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (c:`__Chunk__`) ON (c.id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:`__Chunk__`) ON (c.fileName)",
            "CREATE INDEX IF NOT EXISTS FOR (c:`__Chunk__`) ON (c.position)"
        ]

        connection_manager.create_multiple_indexes(index_queries)

    def clear_existing_index(self) -> None:
        """清除已存在的普通索引"""
        connection_manager.drop_index("chunk_embedding")

    def create_chunk_index(self,
                           node_label:str = '__Chunk__',
                           text_property:str = 'text',
                           embedding_property:str = 'embedding'
                           )->Optional[Neo4jVector]:
        """
        为文本块节点生成embeddings并创建向量存储接口
        :param node_label:文本块节点的标签
        :param text_property:用于计算embedding的文本属性
        :param embedding_property:存储embedding的属性名
        :return: Neo4jVector: 创建的向量存储对象
        """

        start_time = time.time()
        self.clear_existing_index()

        # 获取所有需要处理的文本块节点
        chunks = self.graph.query(
            f"""
           MATCH (c:`{node_label}`)
           WHERE c.{text_property} IS NOT NULL AND c.{embedding_property} IS NULL
           RETURN id(c) AS neo4j_id, c.id AS chunk_id
           """
        )

        if not chunks:
            print("没有找到需要处理的文本块节点")
            # 尝试建立向量储存接口
            try:
                vector_store = Neo4jVector.from_existing_graph(
                    self.embeddings,
                    node_label=node_label,
                    text_node_properties=[text_property],
                    embedding_node_property=embedding_property
                )
                print("成功连接到现有向量索引")
                return vector_store
            except Exception as e:
                print(f"连接到向量存储时出错: {e}")
                return None

        print(f"开始为 {len(chunks)} 个文本块生成embeddings")

        # 批量处理所有的文本
        self._process_embeddings_in_batches(chunks, node_label, text_property, embedding_property)

        # 不尝试创建新的向量索引，只连接到现有的
        try:
            # 创建向量存储对象
            vector_store = Neo4jVector.from_existing_graph(
                self.embeddings,
                node_label=node_label,
                text_node_properties=[text_property],
                embedding_node_property=embedding_property
            )

            end_time = time.time()
            print(f"索引创建成功，总耗时: {end_time - start_time:.2f}秒")
            print(f"其中: embedding计算: {self.embedding_time:.2f}秒, 数据库操作: {self.db_time:.2f}秒")

            return vector_store
        except Exception as e:
            print(f"创建向量存储时出错: {e}")
            return None



    def _process_embeddings_in_batches(self, chunks:List[Dict[str, Any]],
                                        node_label:str,
                                        text_property:str,
                                        embedding_property:str
                                       )->None:
        """
        批量处理文本块embedding的生成
        :param chunks:文本块列表
        :param node_label:节点标签
        :param text_property:文本属性
        :param embedding_property:embedding属性名
        """

        # 获取最优批处理大小
        chunk_count = len(chunks)
        optimal_batch_size = self.get_optimal_batch_size(chunk_count)

        def process_batch(batch, batch_index):
            # 获取批次内所有文本块的文本
            chunk_texts = self._get_chunk_texts_batch(batch, text_property)

            # 计算embedding
            embedding_start = time.time()
            embeddings = self._compute_embeddings_batch(chunk_texts)
            embedding_end = time.time()
            self.embedding_time += (embedding_end - embedding_start)

            # 更新数据库
            db_start = time.time()
            self._update_embeddings_batch(batch, embeddings, embedding_property)
            db_end = time.time()
            self.db_time += (db_end - db_start)

        self.batch_process_with_progress(
            chunks,
            process_batch,
            self.batch_size,
            "处理文本块embedding"
        )

    def _update_embeddings_batch(self, chunks:List[Dict[str, Any]],
                                 embeddings:List[List[float]],
                                 embedding_property:str
                                 )->None:
        """
        批量更新文本块embeddings
        :param chunks:文本块列表
        :param embeddings:对应的embedding列表
        :param embedding_property: embedding属性名
        """
        # 构建更新数据
        update_data = []
        for i, chunk in enumerate(chunks):
            if i < len(embeddings) and embeddings[i] is not None:
                update_data.append({
                    "id": chunk['neo4j_id'],
                    "embedding": embeddings[i]
                })

        # 批量更新
        if update_data:
            try:
                query = f"""
                    UNWIND $updates AS update
                    MATCH (c) WHERE id(c) = update.id
                    SET c.{embedding_property} = update.embedding
                """
                self.graph.query(query, params={"updates": update_data})
            except Exception as e:
                print(f"批量更新embeddings失败: {e}")
                for update in update_data:
                    try:
                        single_query = f"""
                            MATCH (c) WHERE id(c) = $id
                            SET c.{embedding_property} = $embedding
                        """
                        self.graph.query(single_query, params={
                            "id": update["id"],
                            "embedding": update["embedding"]
                        })
                    except Exception as e2:
                        print(f"单个embedding更新失败 (ID: {update['id']}): {e2}")


    def _get_chunk_texts_batch(self, chunks:List[Dict[str, Any]], text_property:str)->List[str]:
        """
         获取批量文本块的文本内容
        :param chunks:文本块列表
        :param text_property:文本属性
        :return:List[str]: 文本块文本列表
        """
        # 构建查询参数
        chunk_ids = [chunk['neo4j_id'] for chunk in chunks]

        # 使用高效的文本提取查询
        query = f"""
               UNWIND $chunk_ids AS id
               MATCH (c) WHERE id(c) = id
               RETURN id, c.{text_property} AS chunk_text
           """

        results = self.graph.query(query, params={"chunk_ids": chunk_ids})

        # 提取文本内容
        chunk_texts = []
        for row in results:
            text = row.get("chunk_text", "")
            # 确保文本不为空
            if not text:
                text = f"chunk_{row['id']}"

            chunk_texts.append(text)

        return chunk_texts

    def _compute_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        计算一批文本的embedding向量
        """
        embeddings = []

        # 预处理文本
        clean_texts = []
        for text in texts:
            safe_text = text if text and text.strip() else "empty chunk"
            clean_texts.append(safe_text)

        # 分析批处理的最佳大小
        embed_batch_size = min(32, len(clean_texts))

        # 批量执行
        for i in range(0, len(clean_texts), embed_batch_size):
            sub_batch = clean_texts[i:i + embed_batch_size]

            # 临时列表，用于存放当前小批次的结果
            sub_batch_embeddings = []

            try:
                # 优先方案：使用官方的批量接口（通常最快且自动保证顺序）
                if hasattr(self.embeddings, 'embed_documents'):
                    sub_batch_embeddings = self.embeddings.embed_documents(sub_batch)
                else:
                    # 备选方案：没有批量接口，必须手动循环
                    # 使用 map 可以严格保证返回顺序与输入顺序一致
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        results = executor.map(self.embeddings.embed_query, sub_batch)
                        sub_batch_embeddings = list(results)

            except Exception as e:
                print(f"批量处理失败: {e}，尝试逐个处理...")
                # 兜底方案：串行处理，确保万无一失
                sub_batch_embeddings = []
                embedding_dim = getattr(self.embeddings, 'embedding_size', 1536)  # 尝试动态获取维度

                for text in sub_batch:
                    try:
                        vec = self.embeddings.embed_query(text)
                        sub_batch_embeddings.append(vec)
                    except Exception as e2:
                        print(f"单个文本计算失败: {e2}")
                        # 填充零向量，保持列表长度对齐
                        sub_batch_embeddings.append([0.0] * embedding_dim)

            # 将这一批结果加入总列表
            embeddings.extend(sub_batch_embeddings)

        return embeddings
