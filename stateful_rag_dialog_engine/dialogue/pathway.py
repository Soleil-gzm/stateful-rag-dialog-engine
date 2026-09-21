import torch
import json
import sys
import logging
# from threading import Thread
import time
import random

from ..modules.registry import ModuleRegistry
from ..modules.selector import select_specs
from ..modules.specs import ModuleSpec
from ..tracking.trackers import QueryTracker

logger = logging.getLogger(__name__)


class Pathway:
    '''
    定义路径Pathway为相对独立的Modules
    '''
    def __init__(self, registry: ModuleRegistry):
        self.pathway = {}
        self.registry = registry

    # ---- 协议输出：所有 stdout 写入都集中在这两个方法 ----

    def _write_response(self, task_id, response):
        """写一条对话话术"""
        sys.stdout.write(json.dumps({"task_id": task_id, "response": response}, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def _write_signal(self, task_id, state, node, repeat, check_count, query_node):
        """写状态信号（驱动外部状态机）"""
        signal = f"<END_OF_STREAMING_SIGNAL>{state}|{node}|{repeat}|{check_count}|{query_node}"
        sys.stdout.write(json.dumps({"task_id": task_id, "response": signal}, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    # ---------------------------------------------------

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
        selected_specs = select_specs(Modules_all, case_info)

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
            self._write_response(task_id, "那您这边稍后有需要的话，可以登录星图金融APP或者在微信上搜索苏宁任性花小程序，在首页点击借款申请就可以了，那这边就先不打扰了，再见。")
            time.sleep(0.01)
            self._write_signal(task_id, state, node, repeat, check_count, query_node)
        elif check_count > 5:       # 控制触发信息问题模块的次数
            self._write_response(task_id, "您这边如果有其他问题可以联系咱们的在线客服，或者打95177转3号键咨询，那本次来电的话主要是邀请您参与咱们平台的优惠活动，您这边稍后可以登录星图金融APP或者微信搜索苏宁任性花小程序，在首页点击去借钱，就可以享受本次优惠了，那这边就先不打扰您了，祝您生活愉快，再见。")
            time.sleep(0.01)
            self._write_signal(task_id, state, node, repeat, check_count, query_node)
        elif node == 'stop' and query.strip() == "。":
            self._write_response(task_id, "那您这边稍后有需要的话，可以登录星图金融APP或者在微信上搜索苏宁任性花小程序，在首页点击借款申请就可以了，那这边就先不打扰了，再见。")
            time.sleep(0.01)
            self._write_signal(task_id, state, node, repeat, check_count, query_node)
        else:
            response = None
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
                module = self.registry.get(selected_specs[current_module][repeat])
                response, label = module.generate_rag(components, query, task_id, case_info)
                repeat += 1
            elif state == 2:
                current_module = '产介'
                module = self.registry.get(selected_specs[current_module][repeat])
                response, label = module.generate_rag(components, query, task_id, case_info)
                repeat += 1
            elif state == 3:
                current_module = '三方'
                module = self.registry.get(selected_specs[current_module][repeat])
                response, label = module.generate_rag(components, query, task_id, case_info)
                repeat += 1
            elif state == 4:
                current_module = '确认'
                module = self.registry.get(selected_specs[current_module][repeat])
                response, label = module.generate_rag(components, query, task_id, case_info)
                if label != "信息问题":
                    repeat += 1
                else:
                    check_count += 1
            elif state == 5:
                current_module = '答疑'
                module = self.registry.get(selected_specs[current_module][repeat])
                response, label = module.generate_rag(components, query, task_id, case_info)
                if label != "信息问题":
                    repeat += 1
                else:
                    check_count += 1
            elif state == 6:
                current_module = '投诉'
                module = self.registry.get(selected_specs[current_module][repeat])
                response, label = module.generate_rag(components, query, task_id, case_info)
                repeat += 1
            elif state == 7:
                current_module = '留言'
                module = self.registry.get(selected_specs[current_module][repeat])
                response, label = module.generate_rag(components, query, task_id, case_info)
                repeat += 1
            # Dialogue state tracking
            # if state != 0:
            # history_update = chat['history']
            # history_update[-1]['message'] = response

            # history_track = tracker.main(args, components, history_update)
            # node = history_track['node']

            # 先写话术，再写状态信号（与拆分前顺序一致）
            if response is not None:                        # ← 新增判断
                self._write_response(task_id, response)
            self._write_signal(task_id, state, node, repeat, check_count, query_node)


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
                line = sys.stdin.readline()
                if not line:
                    # EOF：外部系统关闭了 stdin，正常退出
                    break
                input_data = line.strip()
                if not input_data:
                    # 空行：跳过
                    continue
                self.runLLM(input_data, args, components, tracker_check, tracker_willing,Modules_all)

                # thread = Thread(target=self.runLLM, args=(input_data, args, components, tracker, querytracker, Modules_all))
                # thread.start()
                k = k + 1