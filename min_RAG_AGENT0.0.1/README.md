# min_RAG_AGENT 0.0.1

一个**最小可用的 RAG Agent**：Milvus 向量检索 + DeepSeek 工具调用 Agent，针对「售后客服 FAQ」场景。

设计目标不是功能多，而是把 RAG Agent 的**每一条链路都摊开**：文档切分 → 向量化 → 入库 → 检索 → 工具封装 → Agent 决策 → 对外服务。
每一层都是独立文件、独立函数，可以单独运行和调试。

---

## 1. 技术栈（本机实测版本）

| 组件 | 版本 / 说明 |
| --- | --- |
| Python | 3.13.12（conda 环境 `langchain1.2`） |
| LangChain | 1.2.12（`langchain-core` 1.2.18，`langchain-classic` 1.0.2） |
| 向量数据库 | Milvus Standalone **v3.0.0**（Docker，客户端 `pymilvus` 2.6.12），`http://localhost:19530` |
| 对话模型 | DeepSeek（`init_chat_model`，OpenAI 兼容协议） |
| 嵌入模型 | `Qwen/Qwen3-Embedding-0.6B`（SiliconFlow，1024 维） |
| Web 框架 | FastAPI 0.135.1 + Uvicorn 0.46.0 |
| 配置管理 | python-dotenv 1.2.1 + Pydantic 2.12.5 |

> 依赖清单见 `requirements.txt`。



## 2. 目录结构

```
min_RAG_AGENT0.0.1/
├── .env                      # 密钥与地址（已被 .gitignore 忽略，切勿提交）
├── requirements_full.txt     # 环境全量快照（UTF-16LE 编码）
├── main.py                   # 入口一：命令行交互（REPL）
├── api.py                    # 入口二：FastAPI HTTP 服务
├── agent.py                  # Agent 组装：模型 + 提示词 + 工具 → AgentExecutor
├── tools.py                  # 工具层：把检索能力包装成 LLM 可调用的 @tool
├── data/
│   └── customer_faq.txt      # 知识库原文：10 条售后 FAQ
└── rag/                      # RAG 内核，自底向上 6 层
    ├── config.py             # 第 1 层：集中配置（路径、Milvus、维度、度量）
    ├── loader.py             # 第 2 层：加载 + 切分文档
    ├── embedding.py          # 第 3 层：嵌入模型（懒加载 + 单例）
    ├── vectorstore.py        # 第 4 层：建库建表建索引（一次性脚本）
    ├── ingest.py             # 第 5 层：向量化并写入 Milvus
    └── retriever.py          # 第 6 层：检索 + 拼装上下文 Prompt
```

依赖方向严格单向：`config ← {loader, embedding, vectorstore} ← ingest ← retriever ← tools ← agent ← {main, api}`。

---

## 3. 请求链路

```
用户输入
  ├── main.py   (终端 REPL)  ─┐
  └── api.py    (POST /chat) ─┴──► agent_exe.invoke({"input": "..."})
                                          │
                                          ▼
                             DeepSeek 模型（决定是否调用工具）
                                          │
                                          ▼
                              tools.rag_faq(query="原样用户输入")
                                          │
                                          ▼
                         rag/retriever.generate_answer(query)
                            ├─ embedding_model.embed_query()   ← SiliconFlow
                            ├─ MilvusClient.search(COSINE, top-5)
                            └─ 拼装「问题 + 检索到的上下文」
                                          │
                                          ▼
                             工具返回值作为上下文交回模型
                                          │
                                          ▼
                        最终答案 ──► answer["output"]
```

**一个容易误读的点**：`generate_answer()` 并**不生成**答案，它返回的是「问题 + 上下文」拼好的 prompt 文本。因为工具返回值会被 AgentExecutor 作为消息回填给模型，由模型完成最后一步生成。命名与职责不符，是后续要改的地方（见桌面升级指南 v0.1.0）。

---

## 4. 环境准备

### 4.1 Python 环境

```bash
conda activate langchain1.2     # 本项目实际使用的环境
python --version                # 应为 3.13.x
```

如果换机器重建：

```bash
conda create -n langchain1.2 python=3.13 -y
conda activate langchain1.2
pip install -r requirements_full.txt
```

> `requirements_full.txt` 是 UTF-16LE + CRLF，跨平台工具链容易踩坑，建议尽快转成 UTF-8 的精简 `requirements.txt`。

### 4.2 启动 Milvus（必须先做）

向量库需要独立服务，**不随本项目启动**。用 Milvus 官方 Standalone 方式（Docker Compose 或 Windows 版脚本）拉起 19530 端口即可。

本机已验证可用的组合（三个容器全部 healthy）：

| 容器 | 镜像 |
| --- | --- |
| `milvus-standalone` | `milvusdb/milvus:v3.0.0`（19530 / 9091） |
| `milvus-etcd` | `quay.io/coreos/etcd:v3.5.25` |
| `milvus-minio` | `minio/minio:RELEASE.2024-05-28T17-19-04Z`（9000 / 9001） |

```bash
# 确认三个容器都是 Up (healthy)，且 19530 端口通
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'

# 验证端口是否可连（连接失败时所有检索都会报错）
python -c "import socket;s=socket.socket();s.settimeout(2);print(s.connect_ex(('127.0.0.1',19530)))"
# 输出 0 表示通，非 0 表示 Milvus 没起来
```

### 4.3 配置 `.env`

项目根目录的 `.env` 需要 4 个变量（`.env` 已被 git 忽略，换机器需重新填写）：

```dotenv
# DeepSeek 官方控制台申请，用于对话模型
DEEPSEEK_API_KEY=sk-xxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com

# SiliconFlow（硅基流动）申请，用于嵌入模型
SILICONFLOW_API_KEY=sk-xxxxxxxx
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
```

`agent.py` 与 `rag/embedding.py` 均用 `load_dotenv(override=True)` 读取，即 **`.env` 会覆盖系统已有的同名环境变量**，调试时注意。

---

## 5. 快速开始（首次运行必须按顺序执行）

```bash
# 0. 进入项目根目录并激活环境
cd E:/RAG_agent/min_RAG_AGENT0.0.1
conda activate langchain1.2

# 1. 启动 Milvus，确认 19530 可连（见 4.2）

# 2. 检查切分结果：确认 10 条 FAQ 被切成了什么样子
python -m rag.loader

# 3. 建库、建集合、建索引
#    ⚠️ 这一步会 drop 掉已有集合（当前库里已有 10 条数据）
#    只想验证检索效果的话，跳过 3 和 4，直接做 5
python -m rag.vectorstore

# 4. 向量化并写入 Milvus
python -m rag.ingest

# 5. 验证检索效果：打印命中的片段和相似度分数
python -m rag.retriever     # ⚠️ 当前此脚本有 bug，见第 8 节
```

跑通 5 步之后，可以选任意一种方式使用：

```bash
python agent.py      # 单次调用，验证 Agent 全链路
python main.py       # 终端多轮对话（输入 exit / quit 退出）
python api.py        # 启动 HTTP 服务，默认 8000 端口
```

---

## 6. API 说明

启动：`python api.py`（文档：http://127.0.0.1:8000/docs）

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

### `POST /chat`

请求：`{"query": "下单后什么时候发货"}`，响应：`{"answer": "..."}`

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"下单后什么时候发货"}'
```

- `query` 为空白字符串时返回 `400`
- Agent 内部异常统一返回 `500`，`detail` 为异常信息

---

## 7. 关键设计决策

| 决策 | 原因 |
| --- | --- |
| 提示词强制「原样传入用户输入」 | 检索质量对 query 改写极其敏感。改写后的短句常常丢掉「发货/退货/发票」等关键词，导致召回错误片段。宁可牺牲一点泛化能力，也要保证召回准确。 |
| 度量类型选 `COSINE` | 文本语义检索的标准选择；`L2` 适合图像/通用特征，`IP` 需要向量已归一化。 |
| 嵌入模型做成「懒加载 + 类级单例」 | 嵌入模型是受控的全局资源，避免到处 `new` 造成重复初始化和额外费用。 |
| 集合开 `enable_dynamic_field=True` | `source` 等元数据没写进 schema 也能存进来，方便迭代阶段加字段而不改表。 |
| 三层入口但共用同一个 `agent_exe` | CLI / API / 脚本三种用法共享同一份 Agent 实例，保证行为一致。 |

---

## 8. 已知问题与限制

以下是当前 0.0.1 版本的**真实缺陷**，动手升级前请先看一遍：

| # | 位置 | 问题 |
| --- | --- | --- |
| 1 | `rag/retriever.py:66` | `__main__` 里用了未定义的 `client`（`client` 只存在于 `retrieve()` 函数内）→ `NameError`，`python -m rag.retriever` 无法运行 |
| 2 | `rag/vectorstore.py` | 建表逻辑裸露在模块顶层且无 `if __name__ == "__main__"` 保护，任何 `import` 都会 **drop 掉整个集合**，风险极高 |
| 3 | `rag/loader.py:1` | `from rag.config import Config as config` 引入的是**类**，其它文件引入的是**实例** `config`，风格不统一 |
| 4 | `rag/loader.py:27` | `print(f"共{i}条chunk")` 复用了循环变量：**实测显示"共9条chunk"，实际有 10 条**（打印的是末位索引）；切片为空时还会 `NameError` |
| 5 | `rag/retriever.py:16` | `output_fields` 含 `chunk_id`，但 schema 未声明该字段，取不到值 |
| 6 | `rag/retriever.py` | **没有相关性阈值**，无论多不相关都硬返回 top-5。更要命的是：实测分数与语义相关性**反向**（无关提问分数反而更高），所以阈值在当前嵌入配置下**根本用不了** —— 详见第 9 节实测记录 |
| 7 | `rag/retriever.py` | `generate_answer()` 名不副实，返回的是 prompt 而非答案 |
| 8 | `agent.py` | 无多轮记忆，每次 `invoke` 都是独立会话 |
| 9 | `rag/loader.py` | 切分**碰巧**是对的：`chunk_size=80` 恰好装得下每条 FAQ（实测 10 条切片，每条 51–68 字）。但它依赖"每条 QA ≤ 80 字"+"空行分隔"两个隐式假设，语料一变长就会立刻切坏；`chunk_overlap=20` 在此语料下从未生效 |
| 10 | `rag/loader.py` | `metadata.source` 是**本机绝对路径**（`E:\RAG_agent\...`），10 条切片全都一样，既不可移植也无法用于溯源 |
| 11 | `agent.py:46` | `verbose=True` 常开 |
| 12 | 全局 | 无任何测试用例（`.pytest_cache` 存在但用例为空） |
| 13 | `api.py` | 无鉴权、无超时、无并发控制，`/chat` 全同步阻塞 |
| 14 | `main.py:18` | 重复的 `if not user_input: continue` |
| 15 | `requirements_full.txt` | 通用环境全量快照，非项目最小依赖 |

---

## 9. 实测记录：检索质量验证（2026-09-21）

环境：Docker 中 `milvusdb/milvus:v3.0.0` standalone（19530 已就绪），`pymilvus` 2.6.12 可正常连接并查询。

### 10.1 链路是通的 —— 工程层没有 bug

| 检查项 | 结果 |
| --- | --- |
| `python -m rag.loader` | 10 条切片，每条一问一答（数显显示"共9条"，是第 4 条 bug 的 off-by-one） |
| `python -m rag.vectorstore` | 集合 `question_and_answer` 已存在，10 条数据 |
| `python -m rag.ingest` | 已入库过（`source` 指向旧路径 `E:\MIN_RAG_AGENT\`，项目搬过家） |
| `python agent.py` | **跑通**，返回正确答案 |
| 库内向量 vs 当前模型新嵌入同一文本 | 余弦 **0.9997 ~ 0.9999**，向量新鲜、模型一致 |
| Milvus 返回的排名 vs 本地重算余弦排名 | **完全一致**，Milvus 没有算错 |

结论：切分、嵌入、入库、检索、Agent 组装**每一层都工作正常**。问题不在工程实现。





