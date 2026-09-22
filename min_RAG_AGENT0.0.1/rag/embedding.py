import os
from dotenv import load_dotenv
from langchain.embeddings import init_embeddings
from rag.config import config
load_dotenv(override=True)
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

# 把 embedding 模型变成一个『受控的全局资源』，而不是一个到处乱 new 的临时变量。
# 初始化嵌入模型
class EmbeddingManager:
    _embedding = None # 类级别缓存

    @classmethod  #这个方法是给‘类’用的，不是给‘对象’用的
    def embedding_model(cls):
        """
        对外唯一入口：获取嵌入模型
        懒加载 + 单例
        """
        if cls._embedding is None:
            cls._embedding = init_embeddings(
                model=config.EMBED_MODEL,
                provider=config.PROVIDER,
                api_key=SILICONFLOW_API_KEY,
                base_url=SILICONFLOW_BASE_URL
            )
        return cls._embedding

#  给一个对外的embedding_model接口
embedding_model = EmbeddingManager.embedding_model()
