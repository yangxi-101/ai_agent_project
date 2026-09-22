from rag.embedding import embedding_model
from rag.config import config
from pymilvus import MilvusClient
# 定义一个检索函数
def retrieve(query:str,limit:int=3 ):
    # 1. 将问题向量化
    query_vector = embedding_model.embed_query(str(query))

    # 从向量数据库中检索数据
    client=MilvusClient(config.MILVUS_URL)
    client.use_database(config.DB_NAME)
    results = client.search(
        collection_name=config.COLLECTION_NAME,
        data=[query_vector],
        limit=limit,
        output_fields=['text','chunk_id','source'] #  search 完以后，除了 vector 和 distance，还要把哪些字段一并返回
    )
    return results[0]

def generate_answer(query:str):
    #  1. 检索到的数据
    hits = retrieve(query,limit=5)
    #  2. 格式化的操作
    context_blocks = []

    #  print('=== 检索结果 ===')
    for i,hit in enumerate(hits,1):  #  从1开始记数
        text = hit['entity']['text']
        source = hit['entity'].get("source",'unknow')
        #  chunk_id = hit['entity'].get('id','unknow') # 如果你选择自动增加关键字，那么这行代码将显示不存在
        score = hit['distance']

        print(f"[{i}] score={score:.4f}"
              f"source:{source}")
        print(text)
        print()

        # 拼接编号和元数据的规范上下文块
        context_blocks.append(f"[片段{i} "
                              f"source={source} ] | score = {score}\n {text}")

    context = "\n\n".join(context_blocks)

    # 3. 构造 Prompt
    user_prompt = f"""问题:
{query}
上下文:
{context}

请你根据上述的问题和上下文内容回答问题，如果没有相关信息，请说明：“知识库中没有相关信息”

"""
    # 查看检索命中情况
    # print("=== 检索命中数 ===", len(hits))
    # for hit in hits:
    #     print("score:", hit.score)
    #     print("text:", hit["entity"].get("text"))
    return user_prompt

if __name__ == "__main__":

    content = "下单后什么时候发货"
    generate_answer(content)
    print("=== RETRIEVER DB INFO ===")
    print("collection:", config.COLLECTION_NAME)
    print("num_entities:", client.get_collection_stats(config.COLLECTION_NAME))