🧠 RAG 智能问答系统

基于 LangChain + LangGraph 的检索增强生成（RAG）问答系统，支持多轮对话、来源追溯和置信度评估。

🌟 特性

智能问答：基于知识库的准确回答，减少模型幻觉
多轮对话：自动维护对话历史，支持上下文理解
来源追溯：显示答案引用的具体文档来源
置信度评估：AI自动评估回答的可信度（0-1分）
响应式界面：美观的聊天界面，支持移动端访问
RESTful API：完整的API接口，支持Swagger文档
Docker支持：一键部署，环境隔离

🚀 快速开始

环境要求
Python 3.10+
智谱AI API Key（或其他兼容OpenAI的LLM服务）

安装依赖
pip install -r requirements.txt

配置环境变量
创建 .env 文件：
ZHIPUAI_API_KEY=your_zhipu_api_key
ZHIPUAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
TAVILY_API_KEY=your_tavily_api_key  # 可选，用于网络搜索

启动服务
开发模式（自动重载）
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

或直接运行
python main.py

访问应用
前端界面: http://localhost:8000
API文档: http://localhost:8000/docs
健康检查: http://localhost:8000/health

🐳 Docker 部署

构建镜像
docker build -t rag-qa-system .

运行容器
docker run -p 8000:8000 \
  -v $(pwd)/.env:/app/.env \
  rag-qa-system

使用 Docker Compose（推荐）
docker-compose up --build

📖 API 文档

系统提供完整的 RESTful API：
端点   方法   描述
/health   GET   健康检查

/query   POST   提交问题获取答案

/index   POST   索引新文档

/history   GET   获取对话历史

/stats   GET   获取系统统计

详细API文档请访问：http://localhost:8000/docs

🏗️ 项目结构

├── main.py              # FastAPI 主应用
├── rag_system.py        # RAG 核心逻辑
├── index.html           # 前端聊天界面
├── requirements.txt     # 依赖包列表
├── Dockerfile           # Docker 配置
├── .env                 # 环境变量配置
└── README.md            # 项目说明

🔧 核心技术栈

后端框架: FastAPI
AI框架: LangChain + LangGraph
LLM服务: 智谱AI (GLM-4-Flash)
向量存储: InMemoryVectorStore
前端: HTML5 + CSS3 + JavaScript
部署: Docker

📝 使用示例

单轮问答
用户: 什么是 LangChain？
助手: LangChain 是一个用于开发大型语言模型（LLM）应用的开源框架...
来源: [langchain_intro.txt]
置信度: 95%

多轮对话
用户: LangGraph 是什么？
助手: LangGraph 是 LangChain 生态系统中的一个重要组件...

用户: 它的核心概念有哪些？
助手: LangGraph 的核心概念包括：1. 状态（State）2. 节点（Node）...

⚙️ 配置选项

在 RAGConfig 中可以调整以下参数：

temperature: 生成随机性 (默认: 0.2)
max_tokens: 最大输出长度 (默认: 1000)
chunk_size: 文档分块大小 (默认: 500)
chunk_overlap: 分块重叠 (默认: 100)
top_k: 检索文档数量 (默认: 3)

🤝 贡献指南

欢迎提交 Issue 和 Pull Request！
Fork 本项目
创建特性分支 (git checkout -b feature/AmazingFeature)
提交更改 (git commit -m 'Add some AmazingFeature')
推送到分支 (git push origin feature/AmazingFeature)
创建 Pull Request

📄 许可证

本项目采用 MIT 许可证 - 详情请查看 LICENSE 文件。

注意: 请勿在公共仓库中提交真实的 API Key。使用 .env 文件并将其添加到 .gitignore 中。

Happy Coding! 🚀
