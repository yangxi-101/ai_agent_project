from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from agent import agent_exe


app = FastAPI(
    title="售后客服:min_agent",
    description="基于LangChain+DeepSeek的RAG agent",
    version="0.0.1",
)

# ----------- 请求/响应模型 ---------

class ChatRequest(BaseModel):
    query : str

class ChatResponse(BaseModel):
    answer : str

# ----------- 健康检查 ---------

@app.get('/health')
def health():
    return {'status':'ok'}

# ----------- 核心接口 ---------

@app.post('/chat',response_model=ChatResponse)
def chat(req:ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400,detail="query 不能为空")
    try:
        result = agent_exe.invoke({'input':req.query.strip()})
        return ChatResponse(answer=result['output'])
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app",reload=True)