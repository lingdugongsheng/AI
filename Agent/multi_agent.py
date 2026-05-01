# %%
import os
import json
from typing import List, Dict, Any, TypedDict, Literal
from datetime import datetime
import dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent

dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('ZHIPUAI_API_KEY')
os.environ['OPENAI_BASE_URL'] = os.getenv('ZHIPUAI_BASE_URL')
model = ChatOpenAI(
    model="glm-4-flash",
    temperature=0.3,
    max_tokens=1000
)
# %%
MOCK_ORDERS = {
    "ORD001": {
        "status": "已发货",
        "product": "智能手表 Pro",
        "price": 1299,
        "shipping": "顺丰快递",
        "tracking": "SF1234567890",
        "estimated_delivery": "2024-12-20"
    },
    "ORD002": {
        "status": "处理中",
        "product": "无线耳机 Max",
        "price": 899,
        "shipping": "待发货",
        "tracking": None,
        "estimated_delivery": "2024-12-22"
    },
    "ORD003": {
        "status": "已完成",
        "product": "便携充电宝",
        "price": 199,
        "shipping": "已签收",
        "tracking": "YT9876543210",
        "estimated_delivery": "2024-12-15"
    }
}

MOCK_PRODUCTS = {
    "智能手表 Pro": {
        "price": 1299,
        "features": ["心率监测", "GPS定位", "防水50米", "7天续航"],
        "stock": 50,
        "rating": 4.8
    },
    "无线耳机 Max": {
        "price": 899,
        "features": ["主动降噪", "40小时续航", "蓝牙5.3", "通话降噪"],
        "stock": 120,
        "rating": 4.6
    },
    "便携充电宝": {
        "price": 199,
        "features": ["20000mAh", "快充支持", "双USB输出", "LED显示"],
        "stock": 200,
        "rating": 4.5
    },
    "智能音箱": {
        "price": 499,
        "features": ["语音控制", "多房间音频", "智能家居联动", "Hi-Fi音质"],
        "stock": 80,
        "rating": 4.7
    }
}

FAQ_DATABASE = {
    "连接问题": "请尝试以下步骤：1) 重启设备 2) 检查蓝牙是否开启 3) 删除配对记录后重新配对 4) 确保设备电量充足",
    "充电问题": "建议使用原装充电器，检查充电线是否损坏。如果问题持续，可能需要更换电池或送修。",
    "软件更新": "打开设备对应的APP，进入设置-关于-检查更新，按提示操作即可完成更新。",
    "退货政策": "我们支持7天无理由退货，30天内有质量问题可换货。请保留好购买凭证和完整包装。"
}


# %%
@tool
def query_order(order_id: str) -> str:
    """查询订单信息

    Args:
        order_id: 订单号，格式如 ORD001

    Returns:
        订单详情的JSON字符串
    """
    order = MOCK_ORDERS.get(order_id.upper())
    if order:
        return json.dumps(order, ensure_ascii=False, indent=2)
    return f"未找到订单{order_id}"


@tool
def track_shipping(tracking_number: str) -> str:
    """查询物流信息

    Args:
        tracking_number: 物流单号

    Returns:
        物流状态信息
    """
    if tracking_number.startswith("SF"):
        return f"顺丰快递{tracking_number}:包裹已到达配送站，预计今日送达"
    elif tracking_number.startswith("YT"):
        return f"圆通快递{tracking_number}:已签收"
    return f"未找到物流信息{tracking_number}"


@tool
def search_product(keyword: str) -> str:
    """搜索产品信息

    Args:
        keyword: 产品关键词

    Returns:
        匹配产品的信息
    """
    results = []
    for name, info in MOCK_PRODUCTS.items():
        if keyword.lower() in name.lower():
            results.append({
                "name": name,
                "price": info["price"],
                "features": info["features"],
                "rating": info["rating"]
            })

    if results:
        return json.dumps(results, ensure_ascii=False, indent=2)
    return f"未找到包含{keyword}的产品"


@tool
def get_product_recommendations(budget: int, category: str = "全部") -> str:
    """根据预算推荐产品

    Args:
        budget: 预算金额
        category: 产品类别（可选）

    Returns:
        推荐产品列表
    """
    recommendations = []
    for name, info in MOCK_PRODUCTS.items():
        if info['price'] <= budget:
            recommendations.append({
                "name": name,
                "price": info['price'],
                "rating": info['rating']
            })

    recommendations.sort(key=lambda x: x["price"], reverse=True)
    if recommendations:
        return json.dumps(recommendations[:3], ensure_ascii=False, indent=2)
    return f"在预算{budget}内暂无推荐产品"


@tool
def search_faq(problem_type: str) -> str:
    """搜索常见问题解答

    Args:
        problem_type: 问题类型关键词

    Returns:
        相关FAQ答案
    """
    for key, answer in FAQ_DATABASE.items():
        if problem_type in key or key in problem_type:
            return f"【{key}】\n{answer}"
    return "未找到相关FAQ,建议联系人工客服或缺更多帮助。"


# %%
class CustomerServiceState(TypedDict):
    """客服系统状态"""
    user_message: str  # 用户消息
    chat_history: List[Dict[str, str]]  # 对话历史
    intent: str  # 识别的意图
    confidence: float  # 意图置信度
    agent_response: str  # 代理回复
    needs_escalation: bool  # 是否需要升级
    escalation_reason: str  # 升级原因
    quality_score: float  # 质量评分
    metadata: Dict[str, Any]  # 元数据


# %%
def safe_parse_json(text: str, default: dict = None) -> dict:
    """
    安全解析可能包含在Markdown代码块中的JSON字符串

    Args:
        text: 输入文本，可能包含JSON内容
        default: 解析失败时的默认返回值（默认为空字典）

    Returns:
        解析后的JSON对象或默认值
    """
    # 初始化默认值为空字典（如果未提供）
    if default is None:
        default = {}

    # 去除输入文本两端的空白字符
    content = text.strip()

    # 检查文本是否包含Markdown格式的JSON代码块（```json）
    if "```json" in content:
        try:
            # 提取JSON代码块内容（去除```json和```标记）
            content = content.split("```json")[1].split("```")[0]
        except IndexError:
            # 如果分割后索引超出范围，保持原始内容
            pass
    # 检查文本是否包含普通Markdown代码块（```）
    elif "```" in content:
        try:
            # 提取代码块内容（去除```标记）
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
        except IndexError:
            # 如果分割后索引超出范围，保持原始内容
            pass

    # 去除提取内容两端的空白字符
    content = content.strip()

    # 尝试解析JSON内容
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # 捕获JSON解析错误并打印详细信息
        print(f"JSON 解析失败: {e}")
        # 返回默认值（避免程序中断）
        return default


# %%
class IntentClassifier:
    """意图分类器，用于识别用户消息的业务意图类型"""

    def __init__(self):
        self.llm = model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个意图分类专家，分析用户消息并返回意图分类。
            可选意图：
            - tech_support: 技术问题、故障排除、使用帮助
            - order_service: 订单查询、物流跟踪、退换货
            - product_consult: 产品咨询、价格询问、功能介绍
            - escalate：投诉、无法理解、需要人工

            返回格式(JSON):
            {{"intent": "意图类型","confidence": 0.0-1.0, "reason": "分类原因“}}

            只返回JSON, 不要其他内容。"""),
            ("human", "{message}")
        ])

    def classify(self, message: str) -> Dict[str, Any]:
        chain = self.prompt | self.llm | StrOutputParser()
        result = chain.invoke({"message": message})

        default_result = {"intent": "escalate", "confidence": 0.5, "reason": "解析失败"}
        parsed = safe_parse_json(result, default_result)

        if "intent" in parsed:
            return parsed
        return default_result


# %%
class TechSupportAgent:
    """技术支持Agent，处理技术问题和故障排除"""

    def __init__(self):
        self.llm = model
        self.tools = [search_faq]
        self.system_prompt = """你是一个专业的技术支持工程师，你的职责是：
        1.分析用户遇到的技术问题
        2.提供清晰的故障排除步骤
        3.使用search_faq工具查找相关解决方案
        4.如果问题超出能力范围，建议升级到人工支持
        回复要求：
        - 语气友好专业
        - 步骤清晰有序
        - 提供多个可能解决方案"""

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    def handle(self, message: str, chat_history: List = None) -> str:
        messages = []
        if chat_history:
            for msg in chat_history[-6:]:  # 只保留最近6条，避免token过多
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        messages.append({"role": "user", "content": message})
        result = self.agent.invoke({"messages": messages})
        if result["messages"]:
            return result["messages"][-1].content
        return "抱歉，我暂时无法处理您的问题。建议联系人工客服"


# %%
class OrderServiceAgent:
    """订单服务Agent，处理订单查询和物流跟踪"""

    def __init__(self):
        self.llm = model
        self.tools = [query_order, track_shipping]
        self.system_prompt = """你是一个专业的订单服务专员。你的职责是：
        1. 帮助用户查询订单状态
        2. 提供物流跟踪信息
        3. 解答退换货相关问题
        4. 使用工具获取准确信息

        回复要求：
        - 信息准确完整
        - 主动提供相关信息
        - 如果需要订单号，礼貌询问"""
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt
        )

    def handle(self, message: str, chat_history: List = None) -> str:
        messages = []
        if chat_history:
            for msg in chat_history[-6:]:  # 只保留最近6条，避免token过多
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        messages.append({"role": "user", "content": message})
        result = self.agent.invoke({"messages": messages})
        if result["messages"]:
            return result["messages"][-1].content
        return "抱歉，订单查询服务暂时不可用，请稍后再试。"


# %%
class ProductConsultAgent:
    """产品咨询Agent，提供产品推荐和咨询服务"""

    def __init__(self):
        self.llm = model
        self.tools = [search_product, get_product_recommendations]
        self.system_prompt = """你是一个热情的产品顾问。你的职责是：
        1. 介绍产品功能和特点
        2. 根据用户需求推荐合适的产品
        3. 解答价格和库存问题
        4. 使用工具获取最新产品信息

        回复要求：
        - 热情有亲和力
        - 突出产品优势
        - 根据用户需求推荐
        - 不要过度推销"""
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt
        )

    def handle(self, message: str, chat_history: List = None) -> str:
        messages = []
        if chat_history:
            for msg in chat_history[-6:]:  # 只保留最近6条，避免token过多
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        messages.append({"role": "user", "content": message})
        result = self.agent.invoke({"messages": messages})
        if result["messages"]:
            return result["messages"][-1].content
        return "抱歉，产品信息查询暂时不可用。请稍后再试。"


# %%
class QualityChecker:
    # 质量检查器，评估客服回复质量
    def __init__(self):
        self.llm = model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是客服质量检查专家。评估客服回复的质量。

        评估维度：
        1. 相关性（0-25分）：回复是否针对用户问题
        2. 完整性（0-25分）：是否提供了足够的信息
        3. 专业性（0-25分）：语言是否专业得体
        4. 有用性（0-25分）：是否真正帮助到用户

        返回格式（JSON）：
        {{"total_score": 0-100, "needs_escalation": True/False, "reason": "评估说明"}}

        只返回JSON。"""),
            ("human", """用户问题：{user_message}
        客服回复：{agent_response}

        请评估：""")
        ])

    def check(self, user_message: str, agent_response: str) -> Dict[str, Any]:
        """检查回复质量"""
        chain = self.prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "user_message": user_message,
            "agent_response": agent_response
        })

        # 使用安全的 JSON 解析
        default_result = {"total_score": 60, "needs_escalation": False, "reason": "评估完成"}
        return safe_parse_json(result, default_result)


# %%
class CustomerServiceSystem:
    """智能客服系统核心类，整合意图分类、专业处理和质量检查功能"""

    def __init__(self):
        """初始化客服系统各组件"""
        # 初始化意图分类器，用于识别用户消息的业务意图
        self.classifier = IntentClassifier()
        # 初始化各专业领域处理Agent
        self.tech_agent = TechSupportAgent()  # 技术支持Agent
        self.order_agent = OrderServiceAgent()  # 订单服务Agent
        self.product_agent = ProductConsultAgent()  # 产品咨询Agent
        # 初始化质量检查器，用于评估客服回复质量
        self.quality_checker = QualityChecker()
        # 构建系统工作流状态图
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建客服系统工作流状态图

        工作流程：
        1. 分析用户意图
        2. 根据意图路由到相应处理Agent
        3. 执行质量检查
        4. 根据质量评分决定是否升级到人工客服
        5. 返回最终响应

        Returns:
            编译后的状态图实例
        """

        # 定义工作流中各节点的处理函数

        def classify_intent(state: CustomerServiceState) -> CustomerServiceState:
            """意图分类节点：分析用户消息的业务意图

            Args:
                state: 当前系统状态

            Returns:
                更新后的系统状态，包含意图和置信度
            """
            print("分析用户意图")
            # 调用意图分类器分析用户消息
            result = self.classifier.classify(state["user_message"])
            # 更新状态中的意图和置信度
            state["intent"] = result.get("intent", "escalate")
            state["confidence"] = result.get("confidence", 0.3)

            print(f"意图：{state['intent']}({state['confidence']:.2f})")
            return state

        def route_to_agent(state: CustomerServiceState) -> Literal[
            "tech_support", "order_service", "product_consult", "escalate"]:
            """路由节点：根据意图和置信度决定处理路径

            Args:
                state: 当前系统状态

            Returns:
                目标处理节点的名称
            """
            intent = state["intent"]
            confidence = state["confidence"]

            # 置信度低于阈值时升级到人工客服
            if confidence < 0.6:
                return "escalate"
            # 根据意图类型路由到相应处理节点
            if intent == "tech_support":
                return "tech_support"
            elif intent == "order_service":
                return "order_service"
            elif intent == "product_consult":
                return "product_consult"
            else:
                return "escalate"

        def tech_support_handler(state: CustomerServiceState) -> CustomerServiceState:
            """技术支持处理节点：处理技术问题

            Args:
                state: 当前系统状态

            Returns:
                更新后的系统状态，包含技术支持回复
            """
            print("技术支持代理处理中")
            # 调用技术支持Agent处理用户消息
            response = self.tech_agent.handle(state["user_message"], state["chat_history"])
            state["agent_response"] = response
            return state

        def order_service_handler(state: CustomerServiceState) -> CustomerServiceState:
            """订单服务处理节点：处理订单相关问题

            Args:
                state: 当前系统状态

            Returns:
                更新后的系统状态，包含订单服务回复
            """
            print("订单服务代理处理中")
            # 调用订单服务Agent处理用户消息
            response = self.order_agent.handle(state["user_message"], state["chat_history"])
            state["agent_response"] = response
            return state

        def product_consult_handler(state: CustomerServiceState) -> CustomerServiceState:
            """产品咨询处理节点：处理产品相关问题

            Args:
                state: 当前系统状态

            Returns:
                更新后的系统状态，包含产品咨询回复
            """
            print("产品咨询代理处理中")
            # 调用产品咨询Agent处理用户消息
            response = self.product_agent.handle(state["user_message"], state["chat_history"])
            state["agent_response"] = response
            return state

        def escalate_handler(state: CustomerServiceState) -> CustomerServiceState:
            """升级到人工客服节点：处理需要人工介入的情况

            Args:
                state: 当前系统状态

            Returns:
                更新后的系统状态，包含升级提示信息
            """
            print("升级到人工客服")
            # 设置需要升级的标志
            state["needs_escalation"] = True
            state["escalation_reason"] = "意图识别置信度低或用户要求人工服务"
            # 提供升级提示和替代联系方式
            state["agent_response"] = """非常抱歉，您的问题需要人工客服来处理。
            我已经为您转接人工客服，请稍后···

            在等待期间，你也可以：
            1. 拨打客服热线：400-xxx-xxxx
            2. 发送邮件至：support@example.com
            3. 工作日 9:00-18:00 在线客服响应更快

            感谢您的耐心等待！"""
            return state

        def quality_check(state: CustomerServiceState) -> CustomerServiceState:
            """质量检查节点：评估客服回复质量

            Args:
                state: 当前系统状态

            Returns:
                更新后的系统状态，包含质量评分
            """
            print("执行质量检查")
            # 调用质量检查器评估回复质量
            result = self.quality_checker.check(state["user_message"], state["agent_response"])
            # 更新质量评分
            state["quality_score"] = result.get("total_score", 0) / 100

            # 如果质量评分低于阈值或需要升级，设置相应标志
            if result.get("needs_escalation", False) or state["quality_score"] < 0.6:
                state["needs_escalation"] = True
                state["escalation_reason"] = result.get("reason", "质量检查未通过")

            print(f"质量评分：{state['quality_score']:.2f}")
            return state

        def should_escalate(state: CustomerServiceState) -> Literal["escalate_final", "respond"]:
            """判断是否需要最终升级节点：根据质量检查结果决定

            Args:
                state: 当前系统状态

            Returns:
                "escalate_final"或"respond"，决定后续处理路径
            """
            if state.get("needs_escalation", False):
                return "escalate_final"
            return "respond"

        def final_escalate(state: CustomerServiceState) -> CustomerServiceState:
            """最终升级节点：在回复中添加升级提示

            Args:
                state: 当前系统状态

            Returns:
                更新后的系统状态，包含添加了升级提示的回复
            """
            original_response = state["agent_response"]
            # 在原始回复基础上添加系统升级提示
            state["agent_response"] = f"""{original_response}
            系统提示：由于此问题可能需要更专业的处理，我们建议您联系人工客服以获得更好的服务。"""
            return state

        def respond(state: CustomerServiceState) -> CustomerServiceState:
            """正常响应节点：返回最终回复

            Args:
                state: 当前系统状态

            Returns:
                未修改的系统状态
            """
            return state

        # 创建状态图实例
        graph = StateGraph(CustomerServiceState)

        # 添加工作流节点
        graph.add_node("classify", classify_intent)
        graph.add_node("tech_support", tech_support_handler)
        graph.add_node("order_service", order_service_handler)
        graph.add_node("product_consult", product_consult_handler)
        graph.add_node("escalate", escalate_handler)
        graph.add_node("quality_check", quality_check)
        graph.add_node("escalate_final", final_escalate)
        graph.add_node("respond", respond)

        # 定义节点间的连接关系
        graph.add_edge(START, "classify")
        # 根据路由结果连接到相应处理节点
        graph.add_conditional_edges(
            "classify",
            route_to_agent,
            {
                "tech_support": "tech_support",
                "order_service": "order_service",
                "product_consult": "product_consult",
                "escalate": "escalate"
            }
        )
        # 所有处理节点后都连接到质量检查节点
        graph.add_edge("tech_support", "quality_check")
        graph.add_edge("order_service", "quality_check")
        graph.add_edge("product_consult", "quality_check")
        graph.add_edge("escalate", "quality_check")

        # 根据质量检查结果决定是否需要升级
        graph.add_conditional_edges(
            "quality_check",
            should_escalate,
            {
                "escalate_final": "escalate_final",
                "respond": "respond"
            }
        )

        # 定义最终节点连接
        graph.add_edge("escalate_final", END)
        graph.add_edge("respond", END)

        # 编译并返回状态图
        return graph.compile()

    def handle_message(self, message: str, chat_history: [Dict] = None) -> Dict[str, Any]:
        """处理用户消息的入口方法

        Args:
            message: 用户输入的消息
            chat_history: 聊天历史记录（可选）

        Returns:
            包含回复内容、意图、置信度、质量评分和是否升级的信息字典
        """
        # 打印用户消息
        print(f"\n{'=' * 60}")
        print(f"用户: {message}")
        print('=' * 60)

        # 初始化系统状态
        initial_state = {
            "user_message": message,
            "chat_history": chat_history or [],
            "intent": "",
            "confidence": 0.0,
            "agent_response": "",
            "needs_escalation": False,
            "escalation_reason": "",
            "quality_score": 0.0,
            "metadata": {"timestamp": datetime.now().isoformat()}
        }

        # 执行工作流，获取处理结果
        result = self.graph.invoke(initial_state)

        # 返回格式化的结果
        return {
            "response": result["agent_response"],
            "intent": result["intent"],
            "confidence": result["confidence"],
            "quality_score": result["quality_score"],
            "escalated": result["needs_escalation"]
        }


# %%
def main():
    """演示多代理客服系统"""

    print("=" * 60)
    print("多代理智能客服系统演示")
    print("=" * 60)

    # 初始化系统
    print("\n初始化客服系统...")
    system = CustomerServiceSystem()
    print("系统初始化完成！")

    # 测试场景
    test_cases = [
        # 技术支持场景
        {
            "category": "技术支持",
            "messages": [
                "我的蓝牙耳机连接不上手机怎么办？",
                "手表充电很慢，是不是坏了？"
            ]
        },
        # 订单服务场景
        {
            "category": "订单服务",
            "messages": [
                "帮我查一下订单 ORD001 的物流状态",
                "我的订单什么时候能到？订单号是 ORD002"
            ]
        },
        # 产品咨询场景
        {
            "category": "产品咨询",
            "messages": [
                "你们有什么智能手表推荐吗？预算1500左右",
                "无线耳机有什么功能？"
            ]
        },
        # 升级场景
        {
            "category": "人工升级",
            "messages": [
                "我要投诉！这是第三次出问题了！",
                "我想和你们经理谈谈"
            ]
        }
    ]

    # 运行测试
    for test in test_cases:
        print(f"\n{'=' * 60}")
        print(f"测试类别: {test['category']}")
        print('=' * 60)

        chat_history = []  # ✅ 初始化对话历史

        for message in test["messages"]:
            # ✅ 传递 chat_history
            result = system.handle_message(message, chat_history)

            print("\n客服回复:")
            print(f"{result['response']}")
            print("\n处理信息:")
            print(f"   - 意图: {result['intent']}")
            print(f"   - 置信度: {result['confidence']:.2f}")
            print(f"   - 质量评分: {result['quality_score']:.2f}")
            print(f"   - 是否升级: {'是' if result['escalated'] else '否'}")
            print("-" * 60)

            # ✅ 更新对话历史
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": result['response']})

    # 交互式演示
    print("\n" + "=" * 60)
    print("交互式对话演示")
    print("=" * 60)
    print("提示: 输入 'quit' 退出")

    chat_history = []

    while True:
        user_input = input("\n您: ").strip()

        if user_input.lower() == 'quit':
            print("\n感谢使用智能客服系统，再见！")
            break

        if not user_input:
            continue

        result = system.handle_message(user_input, chat_history)
        print(f"\n客服: {result['response']}")

        # ✅ 更新对话历史
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": result['response']})


# %%
if __name__ == "__main__":
    main()
# %%
