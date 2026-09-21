from typing import List
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# from component.prompt import load_and_format_prompt

import jieba

def get_rag_tools(module_name, module_document, embeddings):
    # 所有QAs整合与分块
    documents = [Document(metadata={}, page_content=module_document)]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 10,
        chunk_overlap  = 0,
        length_function = len,
        separators=["Question:"] 
    )

    docs = text_splitter.split_documents(documents)
    # print(docs)

    texts = [doc.page_content for doc in docs]
    # print(len(docs))

    # 关键词检索
    texts_processed = [preprocessing_func(t) for t in texts]
    vectorizer = BM25Okapi(texts_processed)

    # # 向量检索---语义
    db = FAISS.load_local( 
        folder_path= "./build/db_saves-condition/" + module_name,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorizer, texts, db


def preprocessing_func(text: str) -> List[str]:
    """
    jieba分词
    """
    return list(jieba.cut(text))


def rrf(vector_results: List[str], text_results: List[str], k: int=5, m_vector: int = 20, m_text: int = 100):
    """
    使用RRF算法对两组检索结果进行重排序
        k(int): 排序后返回前k个
        m (int): 超参数
    """
    doc_scores = {}
    
    # 遍历两组结果,计算每个文档的融合分数
    for rank, doc_id in enumerate(vector_results):
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (rank + m_vector)
    for rank, doc_id in enumerate(text_results):
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (rank + m_text)
    
    # 将结果按融合分数排序
    sorted_results = [d for d, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]]

    return sorted_results