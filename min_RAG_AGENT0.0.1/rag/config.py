from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent # BASE_DIR:以当前 config.py 为起点，向上回溯两级，得到的项目根目录的绝对路径。
    BASE_PATH = BASE_DIR/"data"/"customer_faq.txt"
    MILVUS_URL =  "http://localhost:19530"
    DB_NAME = 'MIN_RAG_AGENT_DB'
    COLLECTION_NAME = "question_and_answer"
    EMBED_DIM = 1024 # 根据嵌入向量模型的条件，维度为1024
    METRIC_TYPE = 'COSINE' # 余弦相似度，适用于文本、语义检索 越大越好
  # METRIC_TYPE = 'L2' # 欧式距离，适用于图像、通用特征
  # METRIC_TYPE = 'IP' # 内积，推荐、召回(需要归一化)
  # RAG / 文本 Embedding → 99% 用 COSINE

    #  嵌入模型名称
    EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    #  嵌入模型提供商
    PROVIDER = 'openai'

config = Config()