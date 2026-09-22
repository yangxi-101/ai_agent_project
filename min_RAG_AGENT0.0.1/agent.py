import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from tools import rag_faq


#  基本配置
load_dotenv(override=True)
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 引入模型
model = init_chat_model(
    model="deepseek:deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    extra_body={"thinking":{'type':'disabled'}}
)

# 定义提示词
prompt = ChatPromptTemplate.from_messages([
    ('system',"""
    你是一个售后客服助手。

## 核心规则（必须严格遵守）
1. 调用工具 `rag_faq` 时，**必须原样使用用户输入的 `input` 作为 query，禁止改写、概括、翻译、扩写**。
2. 如果用户输入是“什么时候发货”，传给 rag_faq 的 query 必须是“什么时候发货”，一字不差。
3. 只能使用 `rag_faq` 工具返回的内容回答。
4. 如果工具返回“我暂时无法回答”，你必须原样返回，不得编造。
5. 禁止自行总结、推理、补充知识库以外的内容。
    """),
    ('human','{input}'),
    ('placeholder',"{agent_scratchpad}")
])

# 创造agent
# 专门为DeepSeek和ChatGPT大模型准备的框架
agent = create_tool_calling_agent(llm=model,tools=[rag_faq,],
    prompt=prompt)

agent_exe = AgentExecutor(
    agent = agent,
    tools= [rag_faq,],
    verbose=True,
    handle_parsing_errors=True,
)

if __name__ == "__main__":
    content = "下单后什么时候发货"
    answer = agent_exe.invoke(
        {'input':content}
    )
    final_answer = answer["output"]
    print(final_answer)
