from rag.config import Config as config
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def loader_faq_document():

    # 文档加载
    loader = TextLoader(file_path= config.BASE_PATH,encoding='utf-8')
    data = loader.load()

    # 文本切分策略
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=20,
        separators=[ # 切分策略
            "\n\n",
            "\n",
        ]
    )

    return splitter.split_documents(documents=data)

def check_information(chunks):
    for i,chunk in enumerate(chunks):
        print(f"\n{i}条chunk。","内容为: ",chunk.page_content)
    print(f"共{i}条chunk")

# ========== 仅限本体调试 ==========
if __name__ == "__main__":
    chunks = loader_faq_document()
    check_info =check_information(chunks)
    print(check_info)

