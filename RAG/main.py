from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
import uvicorn
import dotenv

# 导入你的RAG系统
from rag_system import RAGChain, RAGConfig, SAMPLE_DOCUMENTS

# ==================== 配置 ====================

dotenv.load_dotenv()

# ==================== 全局状态 ====================

# RAG系统实例（单例）
rag_system: Optional[RAGChain] = None
rag_config: RAGConfig = RAGConfig()

# 统计信息
stats = {
    "total_queries": 0,
    "total_documents": 0,
    "last_query_time": None
}

# ==================== Lifespan 事件处理器 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器
    """
    global rag_system

    # ========== 启动时执行 ==========
    print("\n" + "=" * 60)
    print("RAG问答系统API启动中...")
    print("=" * 60)

    # 自动初始化RAG系统
    try:
        print("初始化RAG系统...")
        rag_system = RAGChain(config=rag_config)

        # 索引示例文档
        print("\n索引示例文档...")
        texts = [doc["text"] for doc in SAMPLE_DOCUMENTS]
        metadatas = [doc["metadata"] for doc in SAMPLE_DOCUMENTS]

        rag_system.index_documents(texts, metadatas)
        stats["total_documents"] = len(SAMPLE_DOCUMENTS)

        print(f"✓ RAG系统初始化完成，已索引 {len(SAMPLE_DOCUMENTS)} 个文档")
        print("\n可用API文档:")
        print("  - Swagger UI: http://localhost:8000/docs")
        print("  - ReDoc: http://localhost:8000/redoc")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"✗ 系统初始化失败: {e}")
        raise

    yield  # 这里是应用运行的地方

    # ========== 关闭时执行 ==========
    print("\n" + "=" * 60)
    print("RAG问答系统API关闭中...")
    print("=" * 60)
    print(f"本次运行统计:")
    print(f"   - 总查询次数: {stats['total_queries']}")
    print(f"   - 总文档数: {stats['total_documents']}")
    print("=" * 60 + "\n")

# 创建FastAPI应用，使用lifespan
app = FastAPI(
    title="RAG问答系统API",
    description="基于LangChain和LangGraph的智能问答系统",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
    openapi_url="/openapi.json",
    lifespan=lifespan  # 使用新的lifespan事件处理器
)

# CORS配置（允许前端访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 数据模型 ====================

class DocumentInput(BaseModel):
    """文档输入模型"""
    text: str = Field(..., description="文档内容")
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="文档元数据（如source、topic等）"
    )

class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    use_history: bool = Field(default=True, description="是否使用对话历史")

class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str = Field(..., description="AI生成的回答")
    sources: List[Dict] = Field(..., description="引用的文档来源")
    confidence: float = Field(..., description="置信度评分（0-1）", ge=0.0, le=1.0)
    timestamp: str = Field(..., description="响应时间戳")

class IndexRequest(BaseModel):
    """索引请求模型"""
    documents: List[DocumentInput] = Field(..., description="要索引的文档列表")

class IndexResponse(BaseModel):
    """索引响应模型"""
    success: bool = Field(..., description="索引是否成功")
    document_count: int = Field(..., description="索引的文档数量")
    chunk_count: int = Field(..., description="生成的文本块数量")
    message: str = Field(..., description="操作信息")

class HistoryResponse(BaseModel):
    """历史记录响应模型"""
    chat_history: List[Dict[str, str]] = Field(..., description="对话历史")
    count: int = Field(..., description="历史记录数量")

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="系统状态")
    rag_initialized: bool = Field(..., description="RAG系统是否已初始化")
    document_count: int = Field(..., description="已索引的文档数量")
    timestamp: str = Field(..., description="检查时间戳")

class StatsResponse(BaseModel):
    """统计信息响应模型"""
    total_queries: int = Field(..., description="总查询次数")
    average_confidence: float = Field(..., description="平均置信度")
    last_query_time: Optional[str] = Field(None, description="最后查询时间")

# ==================== 辅助函数 ====================

def initialize_rag():
    """初始化RAG系统（如果未初始化）"""
    global rag_system

    if rag_system is None:
        print("初始化RAG系统...")
        rag_system = RAGChain(config=rag_config)

    return rag_system

# ==================== API端点 ====================

# ---- 健康检查 ----
@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查端点"""
    return HealthResponse(
        status="healthy",
        rag_initialized=rag_system is not None,
        document_count=stats["total_documents"],
        timestamp=datetime.now().isoformat()
    )

# ---- 统计信息 ----
@app.get("/stats", response_model=StatsResponse, tags=["系统"])
async def get_stats():
    """获取系统统计信息"""
    total_queries = stats["total_queries"]
    avg_confidence = 0.0

    # 如果有查询记录，计算平均置信度
    if hasattr(rag_system, 'get_last_confidences'):
        confidences = rag_system.get_last_confidences()
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
    return StatsResponse(
        total_queries=total_queries,
        average_confidence=round(avg_confidence, 2),
        last_query_time=stats["last_query_time"]
    )

# ---- 索引文档 ----
@app.post("/index", response_model=IndexResponse, tags=["文档管理"])
async def index_documents(request: IndexRequest):
    """索引新文档到RAG系统"""
    global rag_system

    try:
        # 初始化RAG系统（如果未初始化）
        if rag_system is None:
            rag_system = initialize_rag()

        # 提取文本和元数据
        texts = [doc.text for doc in request.documents]
        metadatas = [doc.metadata for doc in request.documents if doc.metadata]

        # 索引文档
        rag_system.index_documents(texts, metadatas if metadatas else None)

        # 更新统计
        stats["total_documents"] += len(request.documents)

        return IndexResponse(
            success=True,
            document_count=len(request.documents),
            chunk_count=len(request.documents) * 2,  # 估算
            message=f"成功索引 {len(request.documents)} 个文档"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"索引文档失败: {str(e)}"
        )

# ---- 批量索引示例文档 ----
@app.post("/index/sample", response_model=IndexResponse, tags=["文档管理"])
async def index_sample_documents():
    """索引内置的示例文档"""
    global rag_system

    try:
        if rag_system is None:
            rag_system = initialize_rag()
        else:
            # 清除现有历史（重新索引）
            rag_system.clear_history()

        texts = [doc["text"] for doc in SAMPLE_DOCUMENTS]
        metadatas = [doc["metadata"] for doc in SAMPLE_DOCUMENTS]

        rag_system.index_documents(texts, metadatas)
        stats["total_documents"] = len(SAMPLE_DOCUMENTS)

        return IndexResponse(
            success=True,
            document_count=len(SAMPLE_DOCUMENTS),
            chunk_count=len(SAMPLE_DOCUMENTS) * 3,  # 估算
            message=f"成功索引 {len(SAMPLE_DOCUMENTS)} 个示例文档"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"索引示例文档失败: {str(e)}"
        )

# ---- 问答查询 ----
@app.post("/query", response_model=QueryResponse, tags=["问答"])
async def query_rag(request: QueryRequest):
    """向RAG系统提问"""
    global rag_system

    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="RAG系统未初始化，请先索引文档"
        )

    try:
        # 如果不需要使用历史，临时清除
        original_history = None
        if not request.use_history:
            original_history = rag_system.chat_history.copy()
            rag_system.clear_history()

        # 执行查询
        result = rag_system.query(request.question)

        # 恢复历史（如果之前清除了）
        if original_history is not None:
            rag_system.chat_history = original_history

        # 更新统计
        stats["total_queries"] += 1
        stats["last_query_time"] = datetime.now().isoformat()

        # 记录置信度（用于统计）
        if not hasattr(rag_system, '_last_confidences'):
            rag_system._last_confidences = []
        rag_system._last_confidences.append(result["confidence"])
        if len(rag_system._last_confidences) > 100:  # 保留最近100次
            rag_system._last_confidences.pop(0)

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查询失败: {str(e)}"
        )

# ---- 获取对话历史 ----
@app.get("/history", response_model=HistoryResponse, tags=["对话管理"])
async def get_chat_history():
    """获取当前对话历史"""
    global rag_system

    if rag_system is None:
        return HistoryResponse(chat_history=[], count=0)

    return HistoryResponse(
        chat_history=rag_system.chat_history,
        count=len(rag_system.chat_history)
    )

# ---- 清除对话历史 ----
@app.delete("/history", tags=["对话管理"])
async def clear_chat_history():
    """清除对话历史"""
    global rag_system

    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="RAG系统未初始化"
        )

    rag_system.clear_history()

    return {"message": "对话历史已清除", "success": True}

# ---- 重置系统 ----
@app.post("/reset", tags=["系统"])
async def reset_system():
    """重置整个RAG系统（清除所有数据）"""
    global rag_system, stats

    rag_system = None
    stats = {
        "total_queries": 0,
        "total_documents": 0,
        "last_query_time": None
    }

    return {"message": "系统已重置", "success": True}

# ==================== 主程序 ====================

if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # 允许外部访问
        port=8000,
        reload=True,  # 开发模式：代码修改自动重启
        log_level="info"
    )