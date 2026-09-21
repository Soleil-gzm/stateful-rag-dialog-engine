import logging

from ..generation.formatter import load_and_format_prompt
from ..retrieval.rag import get_rag_tools, preprocessing_func, rrf

logger = logging.getLogger(__name__)


class Module:
    '''
    定义模块
    '''
    def __init__(self, name, document_name, embeddings):
        self.name = name
        self.file_prefix = document_name
        self.split_document(document_name, embeddings)

    def split_document(self, document_name, embeddings):
        with open("./build/txt-condition/" + document_name + ".txt", 'r', encoding='utf-8') as f:
            text = f.read()

        lines = text.split("\n")
        qa_pairs = []
        for i in range(0, len(lines), 3):
            if i+2 < len(lines):
                qa_pairs.append((lines[i], lines[i+1], lines[i+2]))

        self.questions = [q for q, a, l in qa_pairs]
        self.answers = [a for q, a, l in qa_pairs]
        self.labels = [l for q, a, l in qa_pairs]

        self.vectorizer, self.texts, self.db = get_rag_tools(document_name, ''.join(self.questions), embeddings)

    def generate_rag(self, components, query, task_id, case_info):
        target = "Question: " + query
        bm25_res = self.vectorizer.get_top_n(preprocessing_func(target), self.texts, n=25)
        vector_res = self.db.similarity_search(target, k=25)

        text_results = [i for i in bm25_res]
        vector_results = [i.page_content for i in vector_res]

        logger.debug("文本检索结果：%s", text_results)
        logger.debug("向量检索结果：%s", vector_results)

        # 取前k个
        rrf_res = rrf(vector_results, text_results, k=5)
        logger.debug("RRF 合并结果：%s", rrf_res)

        # 搜索增强，使用小模型，后续可以变成只在模块3使用？
        # id = self.generate_llm(components["template"], rrf_res, target, components['model_rag'], components['tokenizer_rag'])
        # question = rrf_res[id]
        question = rrf_res[0]
        logger.debug("选中问题：%s", question)

        response = self.answers[self.questions.index(question)].split('Answer: ')[1].strip('\n')
        # response = rrf_res[0].split('Answer: ')[1].strip('\n')
        # print(response)

        label = self.labels[self.questions.index(question)].split('Label: ')[1].strip('\n')

        response = load_and_format_prompt(response, case_info)

        # 不再直接写 stdout —— 协议输出统一由 Pathway 负责
        return response, label