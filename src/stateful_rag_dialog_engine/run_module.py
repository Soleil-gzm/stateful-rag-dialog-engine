'''
Usage:
python run_module.py --args_file=args/qwen.json
python run_module.py

Parameter configuration in 'args/qwen.json'
'''

import argparse

from .arguments.arguments import CustomizedArguments
from .component.template import template_dict
from .component.structure_frontend_v2 import Pathway, Module

from langchain_huggingface import HuggingFaceEmbeddings

from transformers import (
    set_seed,
    HfArgumentParser,
    AutoTokenizer
)
import jieba
import itertools

# import tensorrt_llm
# from tensorrt_llm.runtime import ModelRunnerCpp
# from tensorrt_llm.logger import logger


def setup_arguments():
    '''
    读取arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--args_file", type=str, default='src/args/qwen.json', help="")

    parser.add_argument("--kv_cache_free_gpu_memory_fraction", type=float, default=None)

    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    
    # 读取参数配置
    hf_parser = HfArgumentParser(CustomizedArguments)
    json_args, = hf_parser.parse_json_file(json_file=args.args_file)
    # logger.info("args:{}".format(args))

    if args.kv_cache_free_gpu_memory_fraction is not None:
        json_args.kv_cache_free_gpu_memory_fraction = args.kv_cache_free_gpu_memory_fraction

    # 设置随机种子
    set_seed(json_args.seed)

    return json_args


def load_tokenizer(model_name_or_path):
    '''
    加载tokenizer
    '''
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=False,
        use_fast=False      # 指示是否强制使用 fast tokenizer，即使其不支持特定模型的功能。默认为 True。
    )

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    logger.info(f'vocab_size of tokenizer: {tokenizer.vocab_size}')
    
    return tokenizer


def init_components(args):
    '''
    初始化template, model, tokenizer
    '''
    template = template_dict[args.template_name]

    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model_path, 
        model_kwargs = {'device': 'cpu'}
    )

    # runtime_rank = tensorrt_llm.mpi_rank()

    # tokenizer_track = load_tokenizer(args.model_track_name_or_path)

    # runner_track_kwargs = dict(
    #     engine_dir=args.engine_track_dir,
    #     rank=runtime_rank,
    #     max_output_len=args.max_new_tokens,
    #     enable_context_fmha_fp32_acc=False,
    #     max_batch_size=args.max_batch_size,
    #     max_input_len=args.max_input_length,
    #     kv_cache_free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
    #     cuda_graph_mode=False,
    #     gather_generation_logits=False,
    # )
    # model_track = ModelRunnerCpp.from_dir(**runner_track_kwargs)

    # if template.stop_word is None:
    #     template.stop_word = tokenizer.eos_token
    # stop_token_id = tokenizer.convert_tokens_to_ids(template.stop_word)

    # jieba词库加载
    jieba.initialize()

    return {
        'template': template,
        'model_track': None,
        'tokenizer_track': None,
        'embeddings': embeddings
    }


if __name__ == "__main__":
    # 加载arguments
    args = setup_arguments()

    # 模型Initialize
    components = init_components(args)
    
    # 定义模块配置：(类别, 基础名称, 最大count, 是否有条件)
    module_configs = [
        ('核实', '模块1-确认身份', 1, False),
        ('产介', '模块2-产介', 1, False),
        ('三方', '模块3-三方', 1, False),
        ('确认', '模块4-意愿确认', 2, True),   # 有条件，count从1到2
        ('答疑', '模块5-异议处理', 2, True),   # 有条件，count从1到2
        ('投诉', '投诉倾向', 1, False),
        ('留言', '语音留言', 1, False),
        # ('信息', '信息问题', 1, True)
    ]
    
    # 三对二值条件
    condition_pairs = [
        ('营销优惠卖点非空', '营销优惠卖点为空'),
        ('额度非空', '额度为空'),
        ('预计借款利率区间非空', '预计借款利率区间为空'),
    ]
    
    Modules_all = {}
    
    for category, base_name, max_count, has_condition in module_configs:
        modules = []
        if not has_condition:
            # 无条件的模块：仅一个（count固定为1）
            file_prefix = f"output_{base_name}_1"
            modules.append(Module(category, file_prefix, components['embeddings']))
        else:
            # 有条件的模块：按count和8种组合生成
            for count in range(1, max_count + 1):
                combo_id = 1
                for combo in itertools.product(*condition_pairs):
                    combo_str = '_'.join(combo)
                    file_prefix = f"output_{base_name}_count{count}_combo{combo_id}_{combo_str}"
                    modules.append(Module(category, file_prefix, components['embeddings']))
                    combo_id += 1
        Modules_all[category] = modules
    
    # 可选：打印检查每个类别的模块数量
    for cat, mods in Modules_all.items():
        print(f"{cat}: {len(mods)} 个模块")

    # 构建Pathway
    p = Pathway()
    p.traverse(args, components, Modules_all)