# Copyright (c) OpenMMLab. All rights reserved.
import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
from xtuner.engine.hooks import (DatasetInfoHook, EvaluateChatHook,
                                 VarlenAttnArgsToMessageHubHook)
from xtuner.engine.runner import TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.parallel.sequence import SequenceParallelSampler
from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#
# 模型的基本设置，如预训练模型的选择、数据集信息和训练过程的基本参数
#######################################################################
# Model
pretrained_model_name_or_path = '/home/wuren123/weiy/LLMs/Well_Master/models/internlm2-chat-1_8b'   # 设置LLM路径或huggingface hub id
use_varlen_attn = False

# Data
alpaca_en_path = '/home/wuren123/weiy/LLMs/Well_Master/data/wenlv_data_converted.json'     # 微调数据路径HuggingFace Hub ID，以用于 datasets.load_dataset

# prompt_template = PROMPT_TEMPLATE.default   # 模板提示，用于定义生成文本的格式或结构
prompt_template = PROMPT_TEMPLATE.internlm2_chat

max_length = 2048   # 单条数据最大 token 数，超过则截断
pack_to_max_length = True   # 是否将多条短数据拼接到 max_length 的一条长度的样本, 提高GPU 利用率 

# parallel
sequence_parallel_size = 1  # 并行序列处理的大小，用于模型训练时的序列并行

# Scheduler & Optimizer
batch_size = 4  # per_device 批量大小
accumulative_counts = 16    # 梯度累积，每多少次 backward 更新一次参数
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 0   # 数据加载器中工作进程的数量
max_epochs = 500  # 最大训练轮数
optim_type = AdamW  # 优化器类型
lr = 2e-4   # 学习率
betas = (0.9, 0.999)    # 优化器中beta参数，控制动量和平方梯度的移动平均
weight_decay = 0    # 权重衰减系数，用于正则化和避免过拟合
max_norm = 1  # grad clip 梯度裁剪的最大范数，用于避免梯度爆炸
warmup_ratio = 0.03  # 预热比例，学习率在这个比率的训练过程中线性增加到初始学习率

# Save
save_steps = 2000    # 保存模型的步数间隔（iter数）
save_total_limit = 3  # Maximum checkpoints to keep (-1 means unlimited) 保存模型的总数限制，超过限制时删除旧的模型，-1表示没限制

# Evaluate the generation performance during the training
evaluation_freq = 500
# SYSTEM = SYSTEM_TEMPLATE.alpaca  # 验证对话效果的 system 字段 
SYSTEM = '' 

# 可以设置多个问题，来确保模型在训练过程中的变化是朝着我们想要的方向前进的
evaluation_inputs = [
    # '请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai'
    '请介绍一下你自己', 'Please introduce yourself',
    '去江西旅游你有什么攻略', 'What travel tips do you have for visiting Jiangxi',
    '4月底去江西的武功山合适吗', 'Is it suitable to visit Wugong Mountain in Jiangxi at the end of April'
]

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#
# 指定用于训练的模型的和分词器的具体类型和配置，包括预训练模型的路径和是否启动
# 特定功能（如可变长度注意力），这是模型训练的核心组成部分
#######################################################################
tokenizer = dict(  # 分词器
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(  # 构建model
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    
    llm=dict(  # 构建 LLM
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        
        quantization_config=dict(  
            # 技巧：量化配置（保留则为 4 比特，删除则为正常浮点）；
            # 也就是使用QLoRA，将 quantization_config 设置为 None，就切换成了 LoRA 微调
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    
    lora=dict(  
        # 技巧：LoRA 配置（保留则使用LoRA微淘，删除则使用全量微调）
        # 将quantization_config 和 lora 都设置为None，就切换到全参数训练模式
        type=LoraConfig,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        # modules_to_save=['embed_tokens', 'lm_head']  # 引入 embed_tokens 和 lm_head 的训练（会增大显存需求），进而支持任选对话模版
        task_type='CAUSAL_LM'))

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
# 
# 输出处理的细节，包括如何加载数据集、预处理以及批处理等，确保模型能够收到正确
# 格式和质量的数据
#######################################################################
alpaca_en = dict(  # 构建训练数据集
    type=process_hf_dataset,
    # dataset=dict(type=load_dataset, path=alpaca_en_path),  # 调用 datasets.load_dataset 接口
    dataset=dict(type=load_dataset, path='json', data_files=dict(train=alpaca_en_path)),
    
    tokenizer=tokenizer,
    max_length=max_length,
    # dataset_map_fn=alpaca_map_fn,  # 选择匹配的数据集，这里是map_fn
    dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)

sampler = SequenceParallelSampler \
    if sequence_parallel_size > 1 else DefaultSampler
    
train_dataloader = dict(  # 构造训练数据集的数据加载器
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=alpaca_en,
    sampler=dict(type=sampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#
# 配置优化过程的关键参数，如学习率调整策略和优化器的选择，这些是影响模型训练效
# 果和训练速度的主要因素
#######################################################################
# optimizer
optim_wrapper = dict(  # 构建优化器
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [  # 设置学习率 scheduler
    dict(
        type=LinearLR,  # warup 阶段
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,  # cosine 学习率衰减阶段
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)  # 设置训练迭代代数

#######################################################################
#                           PART 5  Runtime                           #
#
# 定义训练过程的额外配置，如日志、模型保存策略，自定义钩子等，以支持训练流程的
# 监控、调试和结果的保存
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),  # 在训练前打印数据样本
    dict(
        type=EvaluateChatHook,  # 在训练时测试对话效果
        tokenizer=tokenizer,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        system=SYSTEM,
        prompt_template=prompt_template)
]

if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]  # vallen_attention 依赖的 Hook

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
# visualizer = None
from mmengine.visualization import Visualizer, TensorboardVisBackend, WandbVisBackend
# visualizer = dict(type=Visualizer, vis_backends=[dict(type=TensorboardVisBackend)])  # 启动tensorboard
visualizer = dict(type=Visualizer,vis_backends=[dict(type=WandbVisBackend, init_kwargs=dict(project='internlm-1.8b-chat-wenlv'))])  # 启动wandb

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
