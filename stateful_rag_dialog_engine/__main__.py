'''
Usage:
python -m stateful_rag_dialog_engine --args_file=config/args/qwen.json
python -m stateful_rag_dialog_engine

Parameter configuration in 'config/args/qwen.json'
'''

import argparse
import logging
import os
import sys

from .config.arguments import CustomizedArguments
from .config.templates import template_dict
from .dialogue.pathway import Pathway
from .modules.registry import ModuleRegistry
from .modules.specs import build_all_specs
from .utils.logger import setup_logging

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

logger = logging.getLogger(__name__)

def setup_arguments():
    '''
    读取arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--args_file", type=str, default='config/args/qwen.json', help="")

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
    # 初始化日志：文件 + stderr，级别由环境变量 LOG_LEVEL 控制
    log_file = setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
    logger.info("日志文件: %s", log_file)

    # 加载arguments
    args = setup_arguments()

    # 模型Initialize
    components = init_components(args)

    # 构建所有模块的 spec（纯配置，不加载 FAISS）
    specs_all = build_all_specs()

    # 可选：打印检查每个类别的模块数量
    for cat, specs in specs_all.items():
        logger.info("%s: %d 个模块", cat, len(specs))

    # ModuleRegistry：按需构造 Module
    registry = ModuleRegistry(components['embeddings'])

    # 构建Pathway
    p = Pathway(registry)
    p.traverse(args, components, specs_all)