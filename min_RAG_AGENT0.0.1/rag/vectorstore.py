from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema
from rag.config import config

# 只跑一次

#  1. milvus
client = MilvusClient(config.MILVUS_URL)

#  2. 切库
existed_database = client.list_databases()
if config.DB_NAME not in existed_database: # 没有数据库就先创建
    client.create_database(db_name=config.DB_NAME)
client.use_database(db_name=config.DB_NAME)

#  3. 开发阶段：删旧表
if client.has_collection(collection_name=config.COLLECTION_NAME):
    client.drop_collection(collection_name=config.COLLECTION_NAME)

#  4. 定义表结构;表逻辑
fields = [
    FieldSchema(
        name='id',
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True
    ),
    FieldSchema(
        name='vector',
        dtype=DataType.FLOAT_VECTOR,
        dim=config.EMBED_DIM
    ),
    FieldSchema(
        name='text',
        dtype=DataType.VARCHAR,
        max_length=4000,
    ),
]

schema = CollectionSchema(
    fields=fields,
    description="RAG FAQ collection",
    enable_dynamic_field=True # 如果 upsert 进来的数据，带了 schema 里没声明的字段，允许进来，不乱报错。
)

#  5.建索引 --> 快速定位相似向量
index_params = client.prepare_index_params()
index_params.add_index(
    field_name= 'vector',
    index_type='IVF_FLAT',
    metric_type='COSINE',
    params={'nlist':128}
)

#  6.创建指定格式的collection
client.create_collection(
    collection_name=config.COLLECTION_NAME,
    schema=schema,
    index_params=index_params
                         )

# ===== 验证 schema =====
info = client.describe_collection(config.COLLECTION_NAME)

print("=== COLLECTION SCHEMA ===")
print("Collection:", info["collection_name"])
print("AutoID:", info.get("auto_id"))
for f in info["fields"]:
    print(f"- {f['name']}: auto_id={f.get('auto_id')}")