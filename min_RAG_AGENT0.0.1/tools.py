from langchain.tools import tool
from rag.retriever import generate_answer

@tool(parse_docstring=True)
def rag_faq(query:str)->str:
    """
    检索售后 FAQ 知识库，回答用户关于售后政策的问题。

    Args:
        query:用户的原始问题原文。
               为保证检索准确率，请原样传入用户输入，不要改写。

    Return:
           知识库中匹配到的售后政策说明；
        若无相关信息，返回“我暂时无法回答”。
    """
    return generate_answer(query)