import re

def convert_less_than_10000(num: int, is_highest_group: bool = False, prefer_liang: bool = True) -> str:
    """
    将 0~9999 的整数转换为中文读法。
    :param num: 0~9999 的整数
    :param is_highest_group: 是否为最高位组（影响 10~19 的读法，最高组读“十”而非“一十”）
    :return: 中文数字字符串
    """
    if num == 0:
        return "零"

    digits_cn = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    units = ['千', '百', '十', '']  # 分别对应四位中的千、百、十、个位
    
    # 将数字转为四位字符串，不足四位前面补零，例如 20 -> "0020"
    num_str = f"{num:04d}"
    
    result = ""
    zero_pending = False  # 标记是否需要在遇到下一个非零数字时补“零”
    
    for idx, ch in enumerate(num_str):
        if ch == '0':
            # 只有已经有过非零数字后，才需要标记可能补零
            if result != "":
                zero_pending = True
        else:
            # 如果之前有零需要补，则添加“零”
            if zero_pending:
                result += "零"
                zero_pending = False
            
            digit = int(ch)
            
            # 处理十位为 1 的特殊情况：最高组且只有十位和个位时，省略“一”
            if idx == 2 and digit == 1 and is_highest_group and result == "":
                # 例如 10 读“十”，11 读“十一”，12 读“十二”
                result += "十"
            else:
                if prefer_liang and digit == 2 and units[idx] in ('千', '百') and result == "":
                    result += "两" + units[idx]  # 修复：带上单位
                else:
                    result += digits_cn[digit] + units[idx]

    return result 


def convert_integer(int_str: str, prefer_liang: bool = True) -> str:
    """
    将整数字符串（可能包含前导零）转换为中文读法。
    :param int_str: 整数字符串，例如 "100000"
    :return: 中文数字字符串 
    """
    # 去除前导零
    int_str = int_str.lstrip('0')
    if not int_str:
        return "零"
    
    # 从低位到高位每4位分组
    reversed_str = int_str[::-1]
    groups = []
    for i in range(0, len(reversed_str), 4):
        chunk = reversed_str[i:i+4][::-1]  # 恢复为正常顺序
        groups.append(chunk)
    
    # 此时 groups 是从低位到高位的列表，反转得到从高位到低位
    groups = groups[::-1]
    n = len(groups)
    
    # 单位：从低位组开始下标 0 对应个位，1 对应万，2 对应亿，3 对应万亿
    units = ['', '万', '亿', '万亿']
    
    result = ""
    zero_pending = False  # 标记是否遇到全零组，需要在下一个非零组前补“零”
    
    for i, group_str in enumerate(groups):
        num = int(group_str)
        # 该组对应的单位（从低位算起）
        low_idx = n - 1 - i
        unit = units[low_idx] if low_idx < len(units) else ""
        
        if num == 0:
            # 如果前面已经有内容，且不是最后一组，则可能需要在后面补零
            if result != "" and i < n - 1:
                zero_pending = True
            continue
        
        # 处理零的补充
        if zero_pending and result != "" and not result.endswith("零"):
            result += "零"
            zero_pending = False
        # 如果当前组不足四位（即千位为0），且前面有非零组，且结果末尾不是“零”，则需要补零
        elif result != "" and i > 0 and not result.endswith("零") and num < 1000:
            result += "零"
        
        # 组内转换
        group_cn = convert_less_than_10000(num, is_highest_group=(i == 0), prefer_liang=prefer_liang)
        
        # 组级单位（万、亿）前，如果组数值为2，口语化读“两”而非“二”
        if prefer_liang and num == 2 and unit != "":
            group_cn = "两"

        result += group_cn + unit
    
    return result


def amount_to_chinese(amount_str: str, prefer_liang: bool = True) -> str:
    """
    将额度字符串转换为中文数字读法。
    支持格式：纯数字、带逗号、带货币符号（¥、$）、带“元”字、带小数（小数部分忽略，除非有非零小数，按“点X”读）。
    例如：'100,000' -> '十万'；'100000.50' -> '十万点五'
    :param amount_str: 额度字符串
    :return: 中文读法字符串
    """
    if not isinstance(amount_str, str):
        amount_str = str(amount_str)
    
    original = amount_str.strip()
    
    # 提取数字及小数点，忽略其他字符（如逗号、货币符号、空格、元等）
    match = re.search(r'\d+(?:\.\d+)?', original)
    if not match:
        return "无法识别"
    
    number_str = match.group(0)
    
    # 分离整数部分和小数部分
    if '.' in number_str:
        int_part, decimal_part = number_str.split('.')
    else:
        int_part, decimal_part = number_str, ""
    
    # 转换整数部分
    int_cn = convert_integer(int_part, prefer_liang=prefer_liang)
    
    # 处理小数部分，只有非全零才读 
    if decimal_part and decimal_part.strip('0') != '':   # 去掉所有0后非空说明有有效数字
        digits_cn = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
        decimal_cn = "点" + "".join(digits_cn[int(d)] for d in decimal_part)
        return int_cn + decimal_cn
    else:
        return int_cn


if __name__ == "__main__":
    # 测试用例
    test_cases = [
        "100000",
        "100,000",
        "¥100000",
        "100000元",
        "100000.00",
        "100000.50",
        "100000000",
        "100000001",
        "100010000",
        "100200300",
        "110000",
        "1000000",
        "10000000",
        "200000",
        "0",
        "500",
        "1005",
        "1010",
        "1100",
        "1000000000",  # 十亿
        "35000",
        "506420",
        "500503",
        "2000",
        "22000",
        "222200",
        "200"
    ]
    
    for case in test_cases:
        print(f"{case:>15} -> {amount_to_chinese(case)}")