from dataclasses import dataclass, field
# from typing import Optional


@dataclass
class CustomizedArguments:
    """
    一些自定义参数
    """
    case_folder: str = field(metadata={"help": "案例信息存放路径"})
    prompts_path: str = field(metadata={"help": "prompt目录路径"})
    model_track_name_or_path: str = field(metadata={"help": "询问追踪模型权重路径"})
    engine_track_dir: str = field(metadata={"help": "追踪模型trt引擎路径"})

    prompt_track_check_path: str = field(metadata={"help": "询问追踪模型-核实身份prompt权重路径"})
    prompt_track_willing_path: str = field(metadata={"help": "询问追踪模型-意愿prompt权重路径"})

    embedding_model_path: str = field(metadata={"help": "向量模型路径"})
    template_name: str = field(default="", metadata={"help": "数据格式"})
    kv_cache_free_gpu_memory_fraction: float = field(default=0.6, metadata={"help": "kv_cache占用的显存比例"})
    max_batch_size: int = field(default=1, metadata={"help": "模型支持的最大batch size"})
    max_input_length: int = field(default=2048, metadata={"help": "模型输入的最大长度"})
    max_new_tokens: int = field(default=256, metadata={"help": "模型输出的最大长度"})
    top_p: float = field(default=0.9, metadata={"help": "模型输出top_p设置"})
    temperature: float = field(default=0.3, metadata={"help": "模型输出temperature设置"})
    repetition_penalty: float = field(default=1.2, metadata={"help": "模型输出repetition_penalty设置"})
    seed: int = field(default=10, metadata={"help": "随机数种子"})
