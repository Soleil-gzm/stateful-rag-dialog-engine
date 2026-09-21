import torch
import json
import sys
# from threading import Thread
import time
import random
import re

from engine.component.prompt import build_prompt, load_and_format_prompt
from engine.retrieval.rag import get_rag_tools, preprocessing_func, rrf 
from engine.tracking.trackers import QueryTracker


class Pathway:
    '''
    定义路径Pathway为相对独立的Modules
    '''
    def __init__(self):
        self.pathway = {}

    def _extract_count(self, file_prefix):
        """从文件名前缀中提取 count 数字，用于排序"""
        match = re.search(r'count(\d+)', file_prefix)
        if match:
            return int(match.group(1))
        return 0

    def get_case_modules(self, args, case_info, Modules_all):
        marketing_not_null = False if case_info.get('sellingpoint') == "" else True
        quota_not_null = False if case_info.get('quota') == "" else True
        rate_not_null = False if case_info.get('interest') == "" else True

        marketing_cond = '营销优惠卖点非空' if marketing_not_null else '营销优惠卖点为空'
        quota_cond = '额度非空' if quota_not_null else '额度为空'
        rate_cond = '预计借款利率区间非空' if rate_not_null else '预计借款利率区间为空'
        combo_str = f"{marketing_cond}_{quota_cond}_{rate_cond}"

        selected_modules = {}
        # 定义需要按条件筛选的类别
        conditional_categories = {'确认', '答疑'}
    
        for category, module_list in Modules_all.items():
            if category in conditional_categories:
                # 从当前类别中筛选出文件前缀包含 combo_str 的模块
                matched_modules = [mod for mod in module_list if combo_str in mod.file_prefix]
                # 按文件名中的 count 数字排序（确保 repeat 索引对应正确的 count 顺序）
                # 假设文件名中 count 后跟数字，例如 "count1"、"count2"...
                matched_modules.sort(key=lambda x: self._extract_count(x.file_prefix))
                selected_modules[category] = matched_modules
                # 可选：打印警告
                if len(matched_modules) == 0:
                    print(f"警告：未找到 {category} 类别且满足条件 {combo_str} 的模块")
            else:
                selected_modules[category] = module_list
    
        return selected_modules     # 筛选符合客户的话术模块吗？

    def runLLM(self, input_data, args, components, tracker_check, tracker_willing, Modules_all):
        json_data = json.loads(input_data)

        task_id = json_data['task_id']
        chat = json_data['chatHistory']
        prompt = json_data['prompt']

        chatHistory = chat['history']
        query = chatHistory[-2]['message']
        chatHistory = chatHistory[:-2]

        state = chat['state']			# default = 1
        node = chat['node']             # default = "continue"
        repeat = chat['repeat']         # default = 0
        
        # # 读取案例信息
        # case_file_path = args.case_folder + "/case-" + customer_id + ".json"
        # with open(case_file_path, 'r', encoding='utf-8') as file:
        #     case_info = json.load(file)
        case_info = prompt
        
        # -----------
        check_count = chat['check_count'] 	# default = -1
        query_node = chat['query_node']		# default = "continue"      # 这个不需要
        # -----------

        # 每个线程都创建？
        # Modules = Modules_all
        Modules = self.get_case_modules(args, case_info, Modules_all)

        # if node == 'end':
        #     sys.stdout.write(json.dumps({"task_id": task_id, "response": "再见。"}, ensure_ascii=False) + "\n")
        #     sys.stdout.flush()
        #     time.sleep(0.01)
        #     sys.stdout.write(json.dumps({"task_id": task_id, "response": f"<END_OF_STREAMING_SIGNAL>{state}|{node}|{repeat}|{check_count}|{query_node}"}, ensure_ascii=False) + "\n")
        #     sys.stdout.flush()
        # elif node == 'transfer':
        #     sys.stdout.write(json.dumps({"task_id": task_id, "response": "正在为你转接。"}, ensure_ascii=False) + "\n")
        #     sys.stdout.flush()
        #     time.sleep(0.01)
        #     sys.stdout.write(json.dumps({"task_id": task_id, "response": f"<END_OF_STREAMING_SIGNAL>{state}|{node}|{repeat}|{check_count}|{query_node}"}, ensure_ascii=False) + "\n")
        #     sys.stdout.flush()
        # else:
            # if check_count == 3:
            #     # 这里应该用不上了，有了判断repeat最大次数
            #     sys.stdout.write(json.dumps({"task_id": task_id, "response": "很抱歉，如果没法确认是本人接电的话，这边就先不打扰了，再见。"}, ensure_ascii=False) + "\n")
            #     sys.stdout.flush()
            #     time.sleep(0.01)
            #     sys.stdout.write(json.dumps({"task_id": task_id, "response": f"<END_OF_STREAMING_SIGNAL>{state}|{node}|{repeat}|{check_count}|{query_node}"}, ensure_ascii=False) + "\n")
            #     sys.stdout.flush()
        if (state == 1 and repeat > 1) or (state == 2 and repeat > 1) or (state == 3 and repeat > 0) or (state == 4 and repeat > 1) or (state == 5 and repeat > 1) or (state == 6 and repeat > 0) or (state == 7 and repeat > 0):
            # ------------------------判断repeat是否到达最大次数-----------------
            sys.stdout.write(json.dumps({"task_id": task_id, "response": "那您这边稍后有需要的话，可以登录星图金融APP或者在微信上搜索苏宁任性花小程序，在首页点击借款申请就可以了，那这边就先不打扰了，再见。"}, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            time.sleep(0.01)
            sys.stdout.write(json.dumps({"task_id": task_id, "response": f"<END_OF_STREAMING_SIGNAL>{state}|{node}|{repeat}|{check_count}|{query_node}"}, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        elif check_count > 5:       # 控制触发信息问题模块的次数
            sys.stdout.write(json.dumps({"task_id": task_id, "response": "您这边如果有其他问题可以联系咱们的在线客服，或者打95177转3号键咨询，那本次来电的话主要是邀请您参与咱们平台的优惠活动，您这边稍后可以登录星图金融APP或者微信搜索苏宁任性花小程序，在首页点击去借钱，就可以享受本次优惠了，那这边就先不打扰您了，祝您生活愉快，再见。"}, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            time.sleep(0.01)
            sys.stdout.write(json.dumps({"task_id": task_id, "response": f"<END_OF_STREAMING_SIGNAL>{state}|{node}|{repeat}|{check_count}|{query_node}"}, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        elif node == 'stop' and query.strip() == "。":
            sys.stdout.write(json.dumps({"task_id": task_id, "response": "那您这边稍后有需要的话，可以登录星图金融APP或者在微信上搜索苏宁任性花小程序，在首页点击借款申请就可以了，那这边就先不打扰了，再见。"}, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            time.sleep(0.01)
            sys.stdout.write(json.dumps({"task_id": task_id, "response": f"<END_OF_STREAMING_SIGNAL>{state}|{node}|{repeat}|{check_count}|{query_node}"}, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        else:
            # 
            if state == 1 and repeat == 1:
                # repeat>=1，对话进行一轮以上，history可取
                queryhistory = chat['history'][-3:-1]

                track_check = tracker_check.main(args, components["template"], components["model_track"], components["tokenizer_track"], queryhistory)
                query_node = track_check['node']

                if query_node == 'nonidentity':
                    state = 3
                    repeat = 0
                elif query_node == 'message':
                    state = 7
                    repeat = 0
                else:
                    track_willing = tracker_willing.main(args, components["template"], components["model_track"], components["tokenizer_track"], queryhistory)
                    willing_node = track_willing['node']
                    if willing_node == "cc":
                        state = 6
                        repeat = 0
                    else:   # 这里包括node=unknown
                        state = 2
                        repeat = 0
            elif state == 2 and repeat == 1:
                # 产介也判断三方
                queryhistory = chat['history'][-3:-1]

                track_check = tracker_check.main(args, components["template"], components["model_track"], components["tokenizer_track"], queryhistory)
                query_node = track_check['node']

                if query_node == 'nonidentity':
                    state = 3
                    repeat = 0
                elif query_node == 'message':
                    state = 7
                    repeat = 0
                else:
                    track_willing = tracker_willing.main(args, components["template"], components["model_track"], components["tokenizer_track"], queryhistory)
                    willing_node = track_willing['node']
                    if willing_node == "cc":
                        state = 6
                        repeat = 0
                    elif willing_node == "yes":
                        state = 4
                        repeat = 0
                    else:    # 这里包括node=unknown
                        state = 5
                        repeat = 0
            elif state == 4:
                queryhistory = chat['history'][-3:-1]

                track_willing = tracker_willing.main(args, components["template"], components["model_track"], components["tokenizer_track"], queryhistory)
                willing_node = track_willing['node']
                if willing_node == "cc":
                    state = 6
                    repeat = 0
                elif willing_node == "no":
                    state = 5
                    repeat = 0
            elif state == 5:
                queryhistory = chat['history'][-3:-1]
                
                track_willing = tracker_willing.main(args, components["template"], components["model_track"], components["tokenizer_track"], queryhistory)
                willing_node = track_willing['node']
                if willing_node == "cc":
                    state = 6
                    repeat = 0

            # 控制客户不说话的次数
            if (state == 4 or state == 5) and query.strip() == "。":
                if node == 'continue':
                    node = "stop"

            if state == 1:
                current_module = '核实'
                response, label = Modules[current_module][repeat].generate_rag(components, query, task_id, case_info)
                repeat += 1
            elif state == 2:
                current_module = '产介'
                response, label = Modules[current_module][repeat].generate_rag(components, query, task_id, case_info)
                repeat += 1
            elif state == 3:
                current_module = '三方'
                response, label = Modules[current_module][repeat].generate_rag(components, query, task_id, case_info)
                repeat += 1
            elif state == 4:
                current_module = '确认'
                response, label = Modules[current_module][repeat].generate_rag(components, query, task_id, case_info)
                if label != "信息问题":
                    repeat += 1
                else:
                    check_count += 1
            elif state == 5:
                current_module = '答疑'
                response, label = Modules[current_module][repeat].generate_rag(components, query, task_id, case_info)
                if label != "信息问题":
                    repeat += 1
                else:
                    check_count += 1
            elif state == 6:
                current_module = '投诉'
                response, label = Modules[current_module][repeat].generate_rag(components, query, task_id, case_info)
                repeat += 1
            elif state == 7:
                current_module = '留言'
                response, label = Modules[current_module][repeat].generate_rag(components, query, task_id, case_info)
                repeat += 1

            # Dialogue state tracking
            # if state != 0:
            # history_update = chat['history']
            # history_update[-1]['message'] = response

            # history_track = tracker.main(args, components, history_update)
            # node = history_track['node']

            sys.stdout.write(json.dumps({"task_id": task_id, "response": f"<END_OF_STREAMING_SIGNAL>{state}|{node}|{repeat}|{check_count}|{query_node}"}, ensure_ascii=False) + "\n")
            sys.stdout.flush()


    def traverse(self, args, components, Modules_all):
        '''
        遍历路径
        '''
        # ------------
        # tracker_check = QueryTracker(args.prompt_track_check_path)
        # tracker_willing = QueryTracker(args.prompt_track_willing_path)
        # ------------

        class _MockTracker:
            def main(self, *a, **kw):
                return {"node": "unknown"}   # 永远返回 unknown，不触发跳转

        tracker_check = _MockTracker()
        tracker_willing = _MockTracker()

        k = 0
        with torch.no_grad():
            while True:
                if k > 20:
                    torch.cuda.empty_cache()
                    k = 0

                # Format of input_data: 
                # {"task_id":"13827","chatHistory":{"history":[{"role":"user","message":"喂"},{"role":"assistant","message":" "}],"state":1,"node":"continue","repeat":0,"customer_id":1,"check_count":-1,"query_node":"continue"}}
                input_data = sys.stdin.readline().strip()
                self.runLLM(input_data, args, components, tracker_check, tracker_willing, Modules_all)
                
                # thread = Thread(target=self.runLLM, args=(input_data, args, components, tracker, querytracker, Modules_all))
                # thread.start()
                k = k + 1


class Module:
    '''
    定义模块
    '''
    def __init__(self, name, document_name, embeddings):
        self.name = name
        self.file_prefix = document_name
        self.split_document(document_name, embeddings)

    def split_document(self, document_name, embeddings):
        with open("./src/templates/txt-condition/" + document_name + ".txt", 'r', encoding='utf-8') as f:
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

        print("文本：", text_results)
        print("向量：", vector_results)

        # 取前k个
        rrf_res = rrf(vector_results, text_results, k=5)
        print("合并：", rrf_res)

        # 搜索增强，使用小模型，后续可以变成只在模块3使用？
        # id = self.generate_llm(components["template"], rrf_res, target, components['model_rag'], components['tokenizer_rag'])
        # question = rrf_res[id]
        question = rrf_res[0]
        print("选择：", question)

        response = self.answers[self.questions.index(question)].split('Answer: ')[1].strip('\n')
        # response = rrf_res[0].split('Answer: ')[1].strip('\n')
        # print(response)

        label = self.labels[self.questions.index(question)].split('Label: ')[1].strip('\n')

        response = load_and_format_prompt(response, case_info)

        output = {"task_id": task_id, "response": response}
        sys.stdout.write(f"{json.dumps(output, ensure_ascii=False)}\n")
        sys.stdout.flush()

        return response, label