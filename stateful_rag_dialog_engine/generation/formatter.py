from langchain_core.prompts import PromptTemplate
from ..utils.amount import amount_to_chinese


def load_and_format_prompt(prompt, data): 
    """
    加载prompt template, 将对应的case标签替换为具体信息
    """
    # with open(prompt_path, 'r', encoding='utf-8') as f: 
    #     prompt_template = f.read()  

    # check the loading file
    # print(prompt_path)

    prompt_template_load = PromptTemplate.from_template(prompt)
    prompt_case = prompt_template_load.format(
        # 专员工号 = data["专员工号"],
    
        # 姓名 = data["姓名"],
        # 性别 = data["性别"],
        # 产品名称 = data["产品名称"],
        # 营销优惠卖点 = data["营销优惠卖点"],

        专员工号 = data["jobnumber"],
    
        姓名 = data["info_name"],
        性别 = data["info_gender"],
        产品名称 = data["product"],
        营销优惠卖点 = data["sellingpoint"],
        额度 = amount_to_chinese(data["quota"]),
        预计借款利率区间 = data["interest"],
    )
    return prompt_case