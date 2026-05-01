"""
多代理智能客服系统 - FastAPI后端
这是一个基于LangChain和LangGraph的智能客服系统API
功能：意图识别、多代理处理、质量评估、对话历史管理、统计信息等
"""

# ==================== 导入必要的库 ====================

# 基础类型注解
from typing import List, Dict, Optional, Any, Literal  # 用于类型提示：列表、字典、可选值

# 异步上下文管理器（用于应用生命周期管理）
from contextlib import asynccontextmanager

# FastAPI相关
from fastapi import FastAPI, HTTPException, status  # FastAPI框架、异常处理、HTTP状态码
from fastapi.middleware.cors import CORSMiddleware  # 跨域资源共享中间件

# 数据验证
from pydantic import BaseModel, Field  # 数据模型和字段验证

# 时间处理
from datetime import datetime  # 日期时间处理

# Web服务器
import uvicorn  # ASGI服务器，用于运行FastAPI应用

# 环境变量
import dotenv  # 从.env文件加载环境变量

# 导入客服系统
from multi_agent import (
    CustomerServiceSystem,  # 客服系统核心类
    MOCK_ORDERS,           # 测试订单数据
    MOCK_PRODUCTS,         # 测试产品数据
    FAQ_DATABASE           # 常见问题数据库
)

# ==================== 配置 ====================

# 加载环境变量（从.env文件读取API密钥等配置）
dotenv.load_dotenv()

# ==================== 全局状态 ====================


# 客服系统实例（单例模式：整个应用只创建一个实例）
customer_service: Optional[CustomerServiceSystem] = None  # 客服系统对象，初始为None

# 统计信息（记录系统运行数据）
stats = {
    "total_queries": 0,              # 总查询次数
    "queries_by_intent": {},         # 按意图分类的查询次数
    "escalations": 0,                # 升级到人工的次数
    "avg_confidence": 0.0,           # 平均置信度
    "avg_quality_score": 0.0,        # 平均质量评分
    "last_query_time": None,         # 最后一次查询时间
    "uptime": None                   # 系统启动时间
}

# ==================== Lifespan 事件处理器 ====================
# 说明：这是FastAPI的新特性，用于管理应用的启动和关闭事件

@asynccontextmanager  # 异步上下文管理器装饰器
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器
    - 启动时：初始化系统、加载数据
    - 关闭时：清理资源、打印统计
    """
    global customer_service, stats  # 声明使用全局变量

    # ========== 启动时执行（应用启动时运行一次）==========
    print("\n" + "=" * 60)
    print("多代理智能客服系统API启动中...")
    print("=" * 60)

    # 自动初始化客服系统
    try:
        print("初始化客服系统...")
        customer_service = CustomerServiceSystem()  # 创建客服系统实例

        # 加载测试数据
        print("\n加载测试数据...")
        print(f"   - 订单数据: {len(MOCK_ORDERS)} 条")
        print(f"   - 产品数据: {len(MOCK_PRODUCTS)} 个")
        print(f"   - FAQ数据: {len(FAQ_DATABASE)} 条")

        # 记录启动时间
        stats["uptime"] = datetime.now().isoformat()

        print(f"\n✓ 客服系统初始化完成！")
        print("\n可用API文档:")
        print("  - Swagger UI: http://localhost:8000/docs")  # 交互式API文档
        print("  - ReDoc: http://localhost:8000/redoc")      # 另一种API文档
        print("=" * 60 + "\n")
    except Exception as e:
        # 如果初始化失败，打印错误并抛出异常
        print(f"✗ 系统初始化失败: {e}")
        raise  # 重新抛出异常，让FastAPI处理

    yield  # 暂停执行，让应用开始运行（这里是应用运行的地方）

    # ========== 关闭时执行（应用关闭时运行一次）==========
    print("\n" + "=" * 60)
    print("多代理智能客服系统API关闭中...")
    print("=" * 60)
    
    # 计算运行时长
    if stats["uptime"]:
        uptime_duration = datetime.now() - datetime.fromisoformat(stats["uptime"])
        print(f"运行时长: {uptime_duration}")

    print(f"本次运行统计:")
    print(f"   - 总查询次数: {stats['total_queries']}")
    print(f"   - 升级到人工: {stats['escalations']} 次")
    print(f"   - 平均置信度: {stats['avg_confidence']:.2f}")
    print(f"   - 平均质量评分: {stats['avg_quality_score']:.2f}")
    
    # 按意图分类统计
    if stats["queries_by_intent"]:
        print(f"\n   - 按意图分类:")
        for intent, count in stats["queries_by_intent"].items():
            print(f"     • {intent}: {count} 次")
    
    print("=" * 60 + "\n")

# ==================== 创建FastAPI应用 ====================

# 创建FastAPI应用实例，配置基本信息
app = FastAPI(
    title="多代理智能客服系统API",  # API标题
    description="基于LangChain和LangGraph的智能客服系统",  # 描述
    version="1.0.0",  # 版本号
    docs_url="/docs",  # Swagger UI文档路径
    redoc_url="/redoc",  # ReDoc文档路径
    openapi_url="/openapi.json",  # OpenAPI规范路径
    lifespan=lifespan  # 使用上面定义的lifespan事件处理器
)

# ==================== CORS配置 ====================
# 说明：CORS（跨域资源共享）允许浏览器从不同域名访问API

app.add_middleware(
    CORSMiddleware,  # 添加CORS中间件
    allow_origins=["*"],  # 允许所有来源访问（生产环境应改为具体域名）
    allow_credentials=True,  # 允许携带认证信息（如cookies）
    allow_methods=["*"],  # 允许所有HTTP方法（GET、POST、PUT、DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)

# ==================== 数据模型（Pydantic） ====================
# 说明：Pydantic用于数据验证和序列化，确保API输入输出格式正确

class MessageRequest(BaseModel):
    """消息请求模型 - 用于接收用户消息"""
    message: str = Field(
        ...,  # 必填
        description="用户消息内容",
        min_length=1,  # 最小长度1个字符
        max_length=2000  # 最大长度2000个字符
    )
    use_history: bool = Field(
        default=True,  # 默认使用对话历史
        description="是否使用对话历史"
    )

class MessageResponse(BaseModel):
    """消息响应模型 - 用于返回客服回复"""
    response: str = Field(..., description="客服生成的回复")
    intent: str = Field(..., description="识别的意图类型")
    confidence: float = Field(
        ...,
        description="意图识别置信度（0-1）",
        ge=0.0,  # 大于等于0.0
        le=1.0   # 小于等于1.0
    )
    quality_score: float = Field(
        ...,
        description="回复质量评分（0-1）",
        ge=0.0,
        le=1.0
    )
    escalated: bool = Field(..., description="是否升级到人工客服")
    timestamp: str = Field(..., description="响应时间戳")

class IntentStats(BaseModel):
    """意图统计模型"""
    intent: str = Field(..., description="意图类型")
    count: int = Field(..., description="查询次数")

class StatsResponse(BaseModel):
    """统计信息响应模型 - 用于返回系统统计"""
    total_queries: int = Field(..., description="总查询次数")
    escalations: int = Field(..., description="升级到人工次数")
    avg_confidence: float = Field(..., description="平均置信度")
    avg_quality_score: float = Field(..., description="平均质量评分")
    queries_by_intent: List[IntentStats] = Field(..., description="按意图分类的统计")
    uptime: Optional[str] = Field(None, description="系统启动时间")
    last_query_time: Optional[str] = Field(None, description="最后查询时间")

class HealthResponse(BaseModel):
    """健康检查响应模型 - 用于返回系统状态"""
    status: str = Field(..., description="系统状态")
    service_initialized: bool = Field(..., description="客服系统是否已初始化")
    orders_count: int = Field(..., description="订单数据数量")
    products_count: int = Field(..., description="产品数据数量")
    faq_count: int = Field(..., description="FAQ数据数量")
    timestamp: str = Field(..., description="检查时间戳")

class HistoryResponse(BaseModel):
    """历史记录响应模型 - 用于返回对话历史"""
    chat_history: List[Dict[str, str]] = Field(..., description="对话历史")
    count: int = Field(..., description="历史记录数量")

class OrderInfo(BaseModel):
    """订单信息模型"""
    order_id: str = Field(..., description="订单号")
    status: str = Field(..., description="订单状态")
    product: str = Field(..., description="产品名称")
    price: float = Field(..., description="价格")
    shipping: str = Field(..., description="物流信息")
    tracking: Optional[str] = Field(None, description="物流单号")
    estimated_delivery: Optional[str] = Field(None, description="预计送达时间")

class ProductInfo(BaseModel):
    """产品信息模型"""
    name: str = Field(..., description="产品名称")
    price: float = Field(..., description="价格")
    features: List[str] = Field(..., description="产品特性")
    stock: int = Field(..., description="库存数量")
    rating: float = Field(..., description="评分")

class DatabaseInfoResponse(BaseModel):
    """数据库信息响应模型"""
    orders: List[OrderInfo] = Field(..., description="订单列表")
    products: List[ProductInfo] = Field(..., description="产品列表")
    faq_topics: List[str] = Field(..., description="FAQ主题列表")

# ==================== 辅助函数 ====================

def initialize_customer_service():
    """
    初始化客服系统（如果未初始化）
    作用：确保客服系统已创建，避免重复初始化
    """
    global customer_service  # 使用全局变量

    # 如果客服系统未创建，则创建它
    if customer_service is None:
        print("初始化客服系统...")
        customer_service = CustomerServiceSystem()

    return customer_service  # 返回客服系统实例

# ==================== API端点（路由） ====================
# 说明：API端点定义了客户端可以访问的URL路径和处理逻辑

# ---- 健康检查端点 ----
@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """
    健康检查端点
    URL: GET /health
    用途：检查API是否正常运行
    返回：系统状态信息
    """
    return HealthResponse(
        status="healthy",  # 系统状态：健康
        service_initialized=customer_service is not None,  # 客服系统是否已初始化
        orders_count=len(MOCK_ORDERS),  # 订单数据数量
        products_count=len(MOCK_PRODUCTS),  # 产品数据数量
        faq_count=len(FAQ_DATABASE),  # FAQ数据数量
        timestamp=datetime.now().isoformat()  # 当前时间
    )

# ---- 统计信息端点 ----
@app.get("/stats", response_model=StatsResponse, tags=["系统"])
async def get_stats():
    """
    获取系统统计信息
    URL: GET /stats
    用途：查看系统运行统计
    返回：查询次数、置信度、质量评分等
    """
    # 构建意图统计列表
    intent_stats = [
        IntentStats(intent=intent, count=count)
        for intent, count in stats["queries_by_intent"].items()
    ]

    # 返回统计信息
    return StatsResponse(
        total_queries=stats["total_queries"],
        escalations=stats["escalations"],
        avg_confidence=round(stats["avg_confidence"], 2),
        avg_quality_score=round(stats["avg_quality_score"], 2),
        queries_by_intent=intent_stats,
        uptime=stats["uptime"],
        last_query_time=stats["last_query_time"]
    )

# ---- 发送消息端点 ----
@app.post("/chat", response_model=MessageResponse, tags=["客服"])
async def chat_with_service(request: MessageRequest):
    """
    与客服系统对话
    URL: POST /chat
    请求体：MessageRequest（包含用户消息）
    用途：获取智能客服的回复
    返回：回复内容、意图、置信度、质量评分等
    """
    global customer_service  # 使用全局变量

    # 检查客服系统是否已初始化
    if customer_service is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,  # 400错误
            detail="客服系统未初始化"
        )

    try:
        # 如果不需要使用历史，临时清除
        original_history = None
        if not request.use_history:
            # 保存原始历史
            original_history = customer_service.chat_history.copy()
            # 清除历史
            customer_service.clear_history()

        # 执行对话（调用客服系统处理消息）
        result = customer_service.handle_message(request.message)

        # 恢复历史（如果之前清除了）
        if original_history is not None:
            customer_service.chat_history = original_history

        # 更新统计信息
        stats["total_queries"] += 1  # 查询次数+1
        stats["last_query_time"] = datetime.now().isoformat()  # 记录时间

        # 更新意图统计
        intent = result["intent"]
        if intent not in stats["queries_by_intent"]:
            stats["queries_by_intent"][intent] = 0
        stats["queries_by_intent"][intent] += 1

        # 更新升级统计
        if result["escalated"]:
            stats["escalations"] += 1

        # 更新平均置信度（移动平均）
        current_avg = stats["avg_confidence"]
        stats["avg_confidence"] = (
            current_avg * (stats["total_queries"] - 1) + result["confidence"]
        ) / stats["total_queries"]

        # 更新平均质量评分（移动平均）
        current_avg_quality = stats["avg_quality_score"]
        stats["avg_quality_score"] = (
            current_avg_quality * (stats["total_queries"] - 1) + result["quality_score"]
        ) / stats["total_queries"]

        # 返回对话结果
        return MessageResponse(
            response=result["response"],
            intent=result["intent"],
            confidence=result["confidence"],
            quality_score=result["quality_score"],
            escalated=result["escalated"],
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        # 如果发生错误，抛出HTTP异常
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理消息失败: {str(e)}"
        )

# ---- 获取对话历史端点 ----
@app.get("/history", response_model=HistoryResponse, tags=["对话管理"])
async def get_chat_history():
    """
    获取当前对话历史
    URL: GET /history
    用途：查看之前的对话记录
    返回：对话历史列表
    """
    global customer_service  # 使用全局变量

    # 如果客服系统未初始化，返回空历史
    if customer_service is None:
        return HistoryResponse(chat_history=[], count=0)

    # 返回对话历史
    return HistoryResponse(
        chat_history=customer_service.chat_history,  # 对话历史
        count=len(customer_service.chat_history)  # 历史记录数量
    )

# ---- 清除对话历史端点 ----
@app.delete("/history", tags=["对话管理"])
async def clear_chat_history():
    """
    清除对话历史
    URL: DELETE /history
    用途：清空之前的对话记录
    返回：操作结果
    """
    global customer_service  # 使用全局变量

    # 检查客服系统是否已初始化
    if customer_service is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="客服系统未初始化"
        )

    # 清除对话历史
    customer_service.clear_history()

    # 返回成功响应
    return {"message": "对话历史已清除", "success": True}

# ---- 获取数据库信息端点 ----
@app.get("/database", response_model=DatabaseInfoResponse, tags=["数据管理"])
async def get_database_info():
    """
    获取系统数据库信息
    URL: GET /database
    用途：查看订单、产品、FAQ等数据
    返回：数据库内容
    """
    # 转换订单数据
    orders = [
        OrderInfo(
            order_id=order_id,
            status=info["status"],
            product=info["product"],
            price=info["price"],
            shipping=info["shipping"],
            tracking=info.get("tracking"),
            estimated_delivery=info.get("estimated_delivery")
        )
        for order_id, info in MOCK_ORDERS.items()
    ]

    # 转换产品数据
    products = [
        ProductInfo(
            name=name,
            price=info["price"],
            features=info["features"],
            stock=info["stock"],
            rating=info["rating"]
        )
        for name, info in MOCK_PRODUCTS.items()
    ]

    # 返回数据库信息
    return DatabaseInfoResponse(
        orders=orders,
        products=products,
        faq_topics=list(FAQ_DATABASE.keys())
    )

# ---- 重置系统端点 ----
@app.post("/reset", tags=["系统"])
async def reset_system():
    """
    重置整个客服系统（清除所有数据）
    URL: POST /reset
    用途：重新开始，清除所有对话历史
    返回：操作结果
    """
    global customer_service, stats  # 使用全局变量

    # 重置客服系统和统计信息
    if customer_service:
        customer_service.clear_history()
    
    stats = {
        "total_queries": 0,
        "queries_by_intent": {},
        "escalations": 0,
        "avg_confidence": 0.0,
        "avg_quality_score": 0.0,
        "last_query_time": None,
        "uptime": datetime.now().isoformat()
    }

    # 返回成功响应
    return {"message": "系统已重置", "success": True}

# ==================== 主程序 ====================

if __name__ == "__main__":
    # 说明：当直接运行此文件时执行（而不是作为模块导入时）

    # 启动服务器
    uvicorn.run(
        "main:app",  # 模块名:应用实例
        host="0.0.0.0",  # 监听所有网络接口（允许外部访问）
        port=8000,  # 监听端口8000
        reload=True,  # 开发模式：代码修改自动重启
        log_level="info"  # 日志级别：信息
    )