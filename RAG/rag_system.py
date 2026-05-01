# %%
import os
from dataclasses import dataclass
import dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from typing import List, TypedDict, Dict, Optional
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import START, END
from langgraph.graph import StateGraph

dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('ZHIPUAI_API_KEY')
os.environ['OPENAI_BASE_URL'] = os.getenv('ZHIPUAI_BASE_URL')
model = ChatOpenAI(
    model="glm-4-flash",
    temperature=0.3,  # 控制生成随机性，值越低输出越确定（0-1范围）
    max_tokens=1000  # 限制模型单次生成的最大token数量，防止过长响应
)


# %%
def get_embeddings():
    if os.getenv('ZHIPUAI_API_KEY'):
        try:
            from langchain_openai import OpenAIEmbeddings
            print("使用 OpenAI Embeddings")
            return OpenAIEmbeddings(model="embedding-2")
        except:
            print("langchain_openai使用错误。")


# %%
@dataclass
class RAGConfig:
    # 生成参数
    temperature: float = 0.2  # 控制生成文本的随机性，值越低输出越确定、保守
    max_tokens: int = 1000  # 限制模型生成的最大 token 数量，防止输出过长

    # 文档处理参数
    chunk_size: int = 500  # 文档切片的大小（字符数），影响检索精度与上下文完整性
    chunk_overlap: int = 100  # 切片之间的重叠字符数，保证上下文连贯性

    # 检索参数
    top_k: int = 3  # 检索时返回的最相关文档片段数量
    search_type: str = "similarity"  # 检索方式，如 "similarity"（相似度）或 "mmr"（最大边际相关性）


# %%
class RAGState(TypedDict):
    query: str  # 用户当前的查询内容
    chat_history: List[Dict[str, str]]  # 对话历史，按时间顺序存储用户与系统的交互记录
    documents: List[Document]  # 检索到的相关文档列表，每个元素为 Document 对象
    context: str  # 由检索文档生成的上下文信息，用于模型推理
    answer: str  # 模型基于上下文生成的回答
    sources: List[Dict[str, str]]  # 引用的文档来源信息，包含标题、URL 等元数据
    confidence: float  # 模型对生成答案的置信度评分（0~1）


# %%
class DocumentProcessor:
    """
    文档处理器，负责将原始文本转换为向量存储。
    包含文档加载、分割、向量化和存储功能。
    """

    def __init__(self, config: RAGConfig):
        """
        初始化文档处理器。

        :param config: RAG配置对象，包含chunk_size、chunk_overlap等参数
        """
        self.config = config
        # 创建文本分割器，按指定大小和重叠量切分文本
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        # 获取嵌入模型用于向量化
        self.embeddings = get_embeddings()
        # 初始化向量存储
        self.vector_store = None

    def load_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[Document]:
        """
        将文本列表转换为Document对象列表。
        """
        documents = []
        for i, text in enumerate(texts):
            # 如果提供元数据则使用，否则生成默认source
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {'source': f"{i}"}
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档列表按配置切分成小块。

        :param documents: Document对象列表
        :return: 切分后的Document对象列表
        """
        return self.text_splitter.split_documents(documents)

    def create_vector_store(self, documents: List[Document]) -> InMemoryVectorStore:
        """
        创建向量存储，将文档向量化并存储。

        :param documents: 切分后的Document对象列表
        :return: 向量存储对象
        """
        self.vector_store = InMemoryVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )
        return self.vector_store

    def process(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> InMemoryVectorStore:
        """
        完整处理流程：加载->分割->向量化->存储。

        :param texts: 文本内容列表
        :param metadatas: 可选的元数据列表
        :return: 向量存储对象
        """
        print("加载文档")
        documents = self.load_documents(texts, metadatas)
        print(f"加载了{len(documents)}个文档")

        print("分割文件")
        chunks = self.split_documents(documents)
        print(f"生成了{len(chunks)}个文本块")

        print('创建向量存储')
        vectors_store = self.create_vector_store(chunks)
        print("向量存储创建完成")

        return vectors_store


# %%
class Retriever:
    """
    检索器，负责从向量存储中检索与查询最相关的文档。
    支持返回文档及其相关性分数。
    """

    def __init__(self, vector_store: InMemoryVectorStore, config: RAGConfig):
        """
        初始化检索器。

        :param vector_store: 向量存储对象，用于执行相似度搜索
        :param config: RAG配置对象，包含检索参数如top_k
        """
        self.vector_store = vector_store
        self.config = config

    def retrieve(self, query: str) -> List[Document]:
        """
        执行相似度搜索，返回最相关的文档列表。

        :param query: 用户查询字符串
        :return: 最相关的文档列表，数量由config.top_k控制
        """
        return self.vector_store.similarity_search(
            query=query,
            k=self.config.top_k,
        )

    def retrieve_with_scores(self, query: str) -> List[tuple]:
        """
        执行相似度搜索，返回文档及其相关性分数。

        :param query: 用户查询字符串
        :return: 包含文档和分数的元组列表，数量由config.top_k控制
        """
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=self.config.top_k,
        )


# %%
class Generator:
    """
    生成器，负责基于检索到的上下文生成回答、重写查询和评估回答质量。
    是RAG系统的核心组件，处理用户查询并生成最终输出。
    """

    def __init__(self, config: RAGConfig):
        """
        初始化生成器。
        """
        self.config = config
        self.llm = model  # 语言模型实例

        # RAG核心提示模板，定义了如何基于上下文回答问题
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的问答助手，请基于提供的上下文信息回答用户的问题。
            重要规则：
            1. 只使用提供的上下文信息来回答问题。
            2. 如果上下文中没有相关信息，请诚实地说“根据提供的信息，我的回答可能不准确。”，但你可以自行回答这个问题
            3. 回答要准确、简洁、有条理。
            4. 在回答末尾标注信息来源。

            上下文信息：
            {context}
            """),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{query}")
        ])

        # 查询重写提示模板，用于优化用户查询
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个查询优化专家。请根据对话历史，将用户的问题改写为一个独立、完整的查询。

            如果问题本身已经很清晰完整，直接返回原问题。
            只返回改写后的查询，不要添加任何解释。"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "原始问题：{query}\n\n请改写为独立完整的查询：")
        ])

    def rewrite_query(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        重写用户查询，使其成为独立完整的查询。

        :param query: 用户原始查询
        :param chat_history: 对话历史，用于理解上下文
        :return: 重写后的独立完整查询
        """
        if not chat_history:
            return query

        # 仅使用最近4轮对话历史进行重写
        messages = []
        for msg in chat_history[-4:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        chain = self.rewrite_prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "chat_history": messages})

    def generate(self, query: str, context: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        生成基于上下文的回答。

        :param query: 用户查询
        :param context: 检索到的上下文信息
        :param chat_history: 对话历史（可选）
        :return: 生成的回答文本
        """
        messages = []
        if chat_history:
            # 仅使用最近4轮对话历史
            for msg in chat_history[-4:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

        chain = self.rag_prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "query": query,
            "chat_history": messages,
            "context": context
        })

    def evaluate(self, query: str, context: str, answer: str) -> float:
        """
        评估生成回答的质量和置信度。

        :param query: 用户查询
        :param context: 检索到的上下文
        :param answer: 生成的回答
        :return: 0-1之间的置信度分数
        """
        # 评估提示模板，用于判断回答质量
        eval_prompt = ChatPromptTemplate.from_messages([
            ("system", """评估以下回答的置信度。考虑：
            1. 回答是否基于提供的上下文
            2. 信息的相关性和准确性
            3. 回答的完整性

            只返回一个0到1之间的数字，表示置信度。"""),
            ("human", """上下文：{context}

            问题：{query}

            回答：{answer}

            """)
        ])
        chain = eval_prompt | self.llm | StrOutputParser()
        try:
            # 解析并限制置信度在0-1范围内
            score = float(chain.invoke({
                "context": context,
                "query": query,
                "answer": answer
            }).strip())
            return min(max(score, 0), 1.0)
        except:
            return 0.5  # 默认置信度


# %%
class RAGChain:
    """
    RAG核心工作流，整合文档处理、检索和生成组件，实现完整的问答系统。
    通过状态图管理处理流程，确保各组件协同工作。
    """

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统。

        :param config: RAG配置对象，包含系统参数
        """
        self.config = config
        self.processor = DocumentProcessor(self.config)
        self.retriever = None
        self.generator = Generator(self.config)
        self.graph = None

    def index_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """
        索引文档，创建向量存储并初始化检索器。

        :param texts: 文本内容列表
        :param metadatas: 可选的元数据列表
        """
        vector_store = self.processor.process(texts, metadatas)
        self.retriever = Retriever(vector_store, self.config)
        self._build_graph()

    def _build_graph(self):
        """
        构建处理流程图，定义各节点和边。
        流程：查询处理 -> 文档检索 -> 答案生成 -> 置信度评估
        """

        def process_query(state: RAGState) -> RAGState:
            """处理查询，可能需要重写以适应上下文"""
            query = state["query"]
            chat_history = state.get("chat_history", [])

            if chat_history:
                rewritten = self.generator.rewrite_query(query, chat_history)
                print(f"查询改写：{query} -> {rewritten}")
                state["query"] = rewritten

            return state

        def retrieve_documents(state: RAGState) -> RAGState:
            """检索与查询最相关的文档"""
            query = state["query"]
            docs = self.retriever.retrieve(query)
            print(f"检索到{len(docs)}个相关文档")

            state["documents"] = docs

            # 构建上下文和来源信息
            context_parts = []
            sources = []
            for i, doc in enumerate(docs):
                context_parts.append(f"[文档{i + 1}] {doc.page_content}")
                sources.append({
                    "index": i + 1,
                    "source": doc.metadata.get("source", "unknown"),
                    "content_preview": doc.page_content[:100] + "..."
                })

            state["context"] = "\n\n".join(context_parts)
            state["sources"] = sources

            return state

        def generate_answer(state: RAGState) -> RAGState:
            """基于检索到的上下文生成回答"""
            answer = self.generator.generate(
                query=state["query"],
                context=state["context"],
                chat_history=state.get("chat_history", []), )
            state["answer"] = answer
            print("生成回答完成")
            return state

        def evaluate_response(state: RAGState) -> RAGState:
            """评估生成回答的置信度"""
            confidence = self.generator.evaluate(
                query=state["query"],
                context=state["context"],
                answer=state["answer"]
            )
            state["confidence"] = confidence
            print(f"置信度评估：{confidence:.2f}")
            return state

        # 构建状态图
        graph = StateGraph(RAGState)

        # 添加处理节点
        graph.add_node("process_query", process_query)
        graph.add_node("retrieve", retrieve_documents)
        graph.add_node("generate", generate_answer)
        graph.add_node("evaluate", evaluate_response)

        # 定义处理流程
        graph.add_edge(START, "process_query")
        graph.add_edge("process_query", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "evaluate")
        graph.add_edge("evaluate", END)

        self.graph = graph.compile()

    def query(self, question: str, chat_history: List[Dict] = None) -> Dict:
        """
        处理查询。
        :param question: 用户问题
        :param chat_history: 外部传入的对话历史。如果为 None，则使用空列表。
        :return: 结果字典
        """
        if not self.retriever:
            raise ValueError("请先调用 index_documents() 索引文档")

        # 如果没有传入历史，初始化为空列表
        history = chat_history or []

        # ... (原有的 invoke 逻辑)
        # 注意：在 State 构建时，使用传入的 history
        initial_state = {
            "query": question,
            "chat_history": history,  # 这里使用外部传入的
            # ...
        }

        result = self.graph.invoke(initial_state)

        # 关键：将本次的交互追加到历史中，供下一次调用使用
        # 但不要直接修改传入的列表，以免副作用
        new_history = history.copy()
        new_history.append({"role": "user", "content": question})
        new_history.append({"role": "ai", "content": result["answer"]})

        # 返回结果时，附带更新后的 history（或者由 FastAPI 层处理存储）
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "updated_history": new_history  # 供 FastAPI 存储
        }

# %%


SAMPLE_DOCUMENTS = [
    {
        "text": """LangChain 简介

LangChain 是一个用于开发大型语言模型（LLM）应用的开源框架。它提供了一套标准化的接口和工具，
帮助开发者快速构建基于 LLM 的应用程序。

主要特点：
1. 模块化设计：所有组件都可以独立使用或组合使用
2. 链式调用：支持将多个组件链接在一起形成复杂的工作流
3. 记忆管理：内置多种记忆类型，支持对话历史管理
4. 工具集成：可以轻松集成外部工具和 API

LangChain 1.0 于 2025 年 10 月发布，带来了重大改进：
- 更清晰的 API 设计
- 更好的类型提示支持
- 改进的错误处理
- 与 LangGraph 的深度集成

使用场景包括：聊天机器人、问答系统、文档分析、代码生成等。""",
        "metadata": {"source": "langchain_intro.txt", "topic": "introduction"}
    },
    {
        "text": """LangGraph 介绍

LangGraph 是 LangChain 生态系统中的一个重要组件，专门用于构建有状态的、多步骤的 AI 应用。
它基于图结构来定义工作流，使得复杂的 AI 流程变得清晰和可控。

核心概念：
1. 状态（State）：使用 TypedDict 定义应用状态，在节点间传递
2. 节点（Node）：处理状态的函数，执行具体的业务逻辑
3. 边（Edge）：定义节点之间的连接和流转规则
4. 条件边：根据状态动态决定下一个节点

LangGraph 的优势：
- 可视化流程：图结构使工作流一目了然
- 状态管理：自动处理状态的传递和更新
- 检查点：支持中间状态的保存和恢复
- 人机协作：支持 human-in-the-loop 模式

典型应用场景：
- 多步骤推理
- 多代理协作
- 复杂决策流程
- 带有循环的工作流""",
        "metadata": {"source": "langgraph_intro.txt", "topic": "langgraph"}
    },
    {
        "text": """RAG（检索增强生成）原理

RAG 是一种结合检索和生成的技术，通过从知识库中检索相关信息来增强 LLM 的回答质量。

工作流程：
1. 文档处理：将文档分割成小块，并转换为向量表示
2. 向量存储：将文档向量存入向量数据库
3. 查询检索：用户提问时，检索最相关的文档块
4. 上下文增强：将检索到的内容作为上下文提供给 LLM
5. 回答生成：LLM 基于上下文生成准确的回答

RAG 的优势：
- 减少幻觉：基于真实文档生成回答
- 知识更新：无需重新训练模型即可更新知识
- 来源可追溯：可以引用具体的信息来源
- 成本效益：比微调模型更经济

最佳实践：
- 选择合适的分块策略
- 优化检索算法
- 设计有效的提示模板
- 实现结果重排序""",
        "metadata": {"source": "rag_principles.txt", "topic": "rag"}
    },
    {
        "text": """向量数据库介绍

向量数据库是专门用于存储和检索向量数据的数据库系统，是 RAG 系统的核心组件之一。

主要特点：
1. 高效相似度搜索：支持快速的近似最近邻（ANN）搜索
2. 可扩展性：能够处理数百万甚至数十亿级别的向量
3. 实时更新：支持动态添加和删除向量
4. 元数据过滤：支持基于元数据的过滤查询

常见的向量数据库：
- Chroma：轻量级，适合开发和原型
- Pinecone：云原生，完全托管
- Milvus：开源，高性能
- Weaviate：支持混合搜索
- FAISS：Facebook 开发，适合研究

选择建议：
- 开发阶段：使用 Chroma 或内存向量存储
- 生产环境：根据规模选择 Pinecone 或 Milvus
- 需要混合搜索：考虑 Weaviate

性能优化：
- 选择合适的索引类型
- 调整搜索参数
- 使用批量操作""",
        "metadata": {"source": "vector_db.txt", "topic": "database"}
    }
]


