import json
import torch
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError

from engine.component.prompt import build_prompt

class State(BaseModel):
    node: str

class AssistantTracker():
    def __init__(self, prompt_path):
        self.prompt_path = prompt_path

    def load_and_format_prompt(self, prompt_path, history):
        """
        导入DST prompt
        history需要重新格式化
        """
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        history_reformat = ""
        # 只判断最后一轮对话的内容
        for item in history[-2:]:
            role, message = item['role'], item['message']
            if role == 'user':
                history_reformat += "客户：" + message + '\n'
            else:
                history_reformat += "专员：" + message + '\n'

        prompt_template_load = PromptTemplate.from_template(prompt_template)
        prompt_case = prompt_template_load.format(
            json_format = State.model_json_schema(),
            history = history_reformat,
        )

        return prompt_case

    def extract_json(self, text):
        try:
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                return '{"node": "unknown"}'
            json_content = text[json_start:json_end].replace("\\_", "_")
            return json_content
        except Exception as e:
            # return f"Error extracting JSON: {e}"
            return '{"node": "unknown"}'

    # Generate the response under current state
    def generate(self, args, template, model, tokenizer, prompt):
        input_ids = build_prompt(template, tokenizer, query='', history=[], prompt=prompt)

        input_length = input_ids.size(1)
        stop_token_id = tokenizer.convert_tokens_to_ids(template.stop_word)

        outputs = model.generate(
            batch_input_ids=input_ids,
            max_new_tokens=64,
            end_id=stop_token_id,
            pad_id=stop_token_id,
            num_return_sequences=1,
            streaming=False,
            output_sequence_lengths=True,
            output_generation_logits=False,
            return_dict=True,
            return_all_generated_tokens=False
        )
        torch.cuda.synchronize()

        output_ids = outputs["output_ids"]
        sequence_length = outputs["sequence_lengths"][0][0]
        
        # input_text = tokenizer.decode(input_ids[0].tolist())
        # print(f'Input: \"{input_text}\"')
        
        response = tokenizer.decode(output_ids[0][0][input_length:sequence_length].tolist())

        return response

    def main(self, args, template, model, tokenizer, history):
        prompt = self.load_and_format_prompt(self.prompt_path, history)
        # print(prompt)
        """ 
        print(prompt) 
        ## json schema格式如下：
        {json_format}
        注意，如果按以下方式多一个tab就会报错，原因不知
        ## json schema格式如下：
            {json_format}
        """
        response = self.generate(args, template, model, tokenizer, prompt=prompt)

        try:
            res_item = State.model_validate_json(self.extract_json(response))
            res_json = res_item.model_dump_json()
            return json.loads(res_json)
        except ValidationError:
            # 记录日志，返回默认状态
            # logging.error(f"Validation failed for response: {response}")
            default_state = State(node="unknown")
            return json.loads(default_state.model_dump_json())


class QueryTracker():
    def __init__(self, prompt_path):
        self.prompt_path = prompt_path

    def load_and_format_prompt(self, prompt_path, history):
        """
        导入DST prompt
        history需要重新格式化
        """
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        history_reformat = ""
        # 只判断最后一轮对话的内容
        for item in history[-2:]:
            role, message = item['role'], item['message']
            if role == 'assistant':
                history_reformat += "专员：" + message + '\n'
            else:
                history_reformat += "客户：" + message + '\n'

        prompt_template_load = PromptTemplate.from_template(prompt_template)
        prompt_case = prompt_template_load.format(
            json_format = State.model_json_schema(),
            history = history_reformat,
        )

        return prompt_case

    def extract_json(self, text):
        try:
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                return '{"node": "unknown"}'
            json_content = text[json_start:json_end].replace("\\_", "_")
            return json_content
        except Exception as e:
            # return f"Error extracting JSON: {e}"
            return '{"node": "unknown"}'

    # Generate the response under current state
    def generate(self, args, template, model, tokenizer, prompt):
        input_ids = build_prompt(template, tokenizer, query='', history=[], prompt=prompt)

        input_length = input_ids.size(1)
        stop_token_id = tokenizer.convert_tokens_to_ids(template.stop_word)

        outputs = model.generate(
            batch_input_ids=input_ids,
            max_new_tokens=64,
            end_id=stop_token_id,
            pad_id=stop_token_id,
            num_return_sequences=1,
            streaming=False,
            output_sequence_lengths=True,
            output_generation_logits=False,
            return_dict=True,
            return_all_generated_tokens=False
        )
        torch.cuda.synchronize()

        output_ids = outputs["output_ids"]
        sequence_length = outputs["sequence_lengths"][0][0]
        
        # input_text = tokenizer.decode(input_ids[0].tolist())
        # print(f'Input: \"{input_text}\"')
        
        response = tokenizer.decode(output_ids[0][0][input_length:sequence_length].tolist())

        return response

    def main(self, args, template, model, tokenizer, history):
        prompt = self.load_and_format_prompt(self.prompt_path, history)
        # print(prompt)
        """ 
        print(prompt) 
        ## json schema格式如下：
        {json_format}
        注意，如果按以下方式多一个tab就会报错，原因不知
        ## json schema格式如下：
            {json_format}
        """
        response = self.generate(args, template, model, tokenizer, prompt=prompt)
        
        try:
            res_item = State.model_validate_json(self.extract_json(response))
            res_json = res_item.model_dump_json()
            return json.loads(res_json)
        except ValidationError:
            # 记录日志，返回默认状态
            # logging.error(f"Validation failed for response: {response}")
            default_state = State(node="unknown")
            return json.loads(default_state.model_dump_json())


if __name__ == "__main__":
    history = []
    with open('../output/history.jsonl', "r") as f:
        for line in f:
            history.append(json.loads(line))

    prompt_path = "../prompts/dialogue_track.txt"

    # prompt = load_and_format_prompt(prompt_path, history)
    # print(prompt)
