import torch


def build_prompt(template, tokenizer, query, history, prompt=None):
    """
    构建LLM prompt, 调节system prompt与history的相对位置
    """
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format

    system_text = system_format.format(content=prompt)
    input_ids = tokenizer.encode(system_text, add_special_tokens=False)

    # concat conversation
    history.append({"role": 'user', 'message': query})
    for item in history:
        role, message = item['role'], item['message']
        if role == 'user':
            message = user_format.format(content=message, stop_token=tokenizer.eos_token)
        else:
            message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
        tokens = tokenizer.encode(message, add_special_tokens=False)
        input_ids += tokens

    # add the assistant start if using the standard format
    input_ids += tokenizer.encode('<|im_start|>assistant\n')
    # print("当前Prompt：" + tokenizer.decode(input_ids))
    # print(input_ids)

    input_ids = torch.tensor([input_ids], dtype=torch.long)

    return input_ids