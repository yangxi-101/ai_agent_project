from typing import List
from langchain_core.documents import Document
from pymilvus import MilvusClient
from rag.embedding import embedding_model
from rag.loader import loader_faq_document
from rag.config import config


def build_insert_data(chunks:List[Document]) -> List[dict]:
    # 加载文档切片
    text = [chunk.page_content for chunk in chunks] # 生成一个列表内含 Document 类型的对象

    #  向量化
    vectors = embedding_model.embed_documents(text)

    #  构建数据
    #  将切片与向量相关联，形成配对
    return [
        {
            'vector': vectors[i],
            'text':chunks[i].page_content,
            'source': chunks[i].metadata.get("source", "") # metadata 里拿 source，如果有就拿；没有，就给一个空字符串，别报错。
        }
        for i in range(len(chunks))
    ]

def ingest():
    chunks = loader_faq_document()
    data = build_insert_data(chunks)
    #  连接Milvus,并使用数据库
    client = MilvusClient(config.MILVUS_URL)
    client.use_database(db_name=config.DB_NAME)

    # 写入
    insert_res = client.insert(
        collection_name=config.COLLECTION_NAME,
        data=data
    )
    client.flush(collection_name=config.COLLECTION_NAME)

    print(f"向量化完成，文本与向量相匹配，并且存入 Milvus 。写入条数：{insert_res['insert_count']}")


if __name__ == "__main__":
    ingest()
  # # 验证
  #   def testify_insert_message(client):
  #       stats = client.get_collection_stats(collection_name=config.COLLECTION_NAME)
  #       print("✅ Collection stats:", stats)
  #
  #       result = client.query(
  #           collection_name=config.COLLECTION_NAME,
  #           filter="id >= 0",
  #           output_fields=["id", "text", "source"]
  #       )
  #
  #       print(f"✅ 成功写入 {len(result)} 条数据")
  #       for r in result:
  #           print(r)


