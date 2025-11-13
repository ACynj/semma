"""
EnhancedUltra 消融实验运行脚本

此脚本用于运行消融实验，测试各个组件的贡献。
消融实验配置：
1. Baseline (SEMMA): 无任何增强
2. +SimilarityEnhancer: 只使用相似度增强
3. +AdaptiveGate: 相似度增强 + 自适应门控
4. +PromptGraph: 只使用提示图增强
5. +SimilarityEnhancer + PromptGraph: 相似度增强 + 提示图增强
6. Full (EnhancedUltra): 所有组件（相似度增强 + 自适应门控 + 提示图增强）
"""

import os
import sys
import csv
import time
import argparse
import random
import yaml

import torch
import torch_geometric as pyg
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util, parse
from ultra.models import Ultra
from ultra.enhanced_models import EnhancedUltra
from script.run import train_and_validate, test

mydir = os.getcwd()

# 消融实验配置
ABLATION_CONFIGS = {
    "baseline": {
        "name": "Baseline (SEMMA)",
        "run": "semma",
        "use_similarity_enhancer": False,
        "use_prompt_enhancer": False,
        "use_adaptive_gate": False,
        "description": "无任何增强，作为基线"
    },
    "similarity_only": {
        "name": "+SimilarityEnhancer",
        "run": "EnhancedUltra",
        "use_similarity_enhancer": True,
        "use_prompt_enhancer": False,
        "use_adaptive_gate": False,
        "description": "只使用相似度增强"
    },
    "similarity_gate": {
        "name": "+SimilarityEnhancer+Gate",
        "run": "EnhancedUltra",
        "use_similarity_enhancer": True,
        "use_prompt_enhancer": False,
        "use_adaptive_gate": True,
        "description": "相似度增强 + 自适应门控"
    },
    "prompt_only": {
        "name": "+PromptGraph",
        "run": "EnhancedUltra",
        "use_similarity_enhancer": False,
        "use_prompt_enhancer": True,
        "use_adaptive_gate": False,
        "description": "只使用提示图增强"
    },
    "similarity_prompt": {
        "name": "+SimilarityEnhancer+PromptGraph",
        "run": "EnhancedUltra",
        "use_similarity_enhancer": True,
        "use_prompt_enhancer": True,
        "use_adaptive_gate": False,
        "description": "相似度增强 + 提示图增强"
    },
    "full": {
        "name": "Full (EnhancedUltra)",
        "run": "EnhancedUltra",
        "use_similarity_enhancer": True,
        "use_prompt_enhancer": True,
        "use_adaptive_gate": True,
        "description": "所有组件（相似度增强 + 自适应门控 + 提示图增强）"
    }
}

# 默认训练配置
default_train_config = {
    "CoDExSmall": (10, 1000),
    "CoDExMedium": (10, 1000),
    "CoDExLarge": (10, 1000),
    "FB15k237": (10, 1000),
    "WN18RR": (10, 1000),
    "YAGO310": (10, 2000),
    "DBpedia100k": (10, 1000),
    "AristoV4": (10, 1000),
    "ConceptNet100k": (10, 1000),
    "ATOMIC": (10, 1000),
    "NELL995": (10, 1000),
    "Hetionet": (10, 1000),
    "WDsinger": (10, 1000),
    "FB15k237_10": (10, 1000),
    "FB15k237_20": (10, 1000),
    "FB15k237_50": (10, 1000),
    "NELL23k": (10, 1000),
    "FB15k237Inductive": (10, 'null'),
    "WN18RRInductive": (10, 'null'),
    "NELLInductive": (10, 'null'),
    "ILPC2022SmallInductive": (10, 'null'),
    "ILPC2022LargeInductive": (10, 1000),
    "NLIngram": (10, 'null'),
    "FBIngram": (10, 'null'),
    "WKIngram": (10, 'null'),
    "WikiTopicsMT1": (10, 'null'),
    "WikiTopicsMT2": (10, 'null'),
    "WikiTopicsMT3": (10, 'null'),
    "WikiTopicsMT4": (10, 'null'),
    "Metafam": (10, 'null'),
    "FBNELL": (10, 'null'),
    "HM": (10, 1000)
}

default_finetuning_config = {
    "CoDExSmall": (1, 4000),
    "CoDExMedium": (1, 4000),
    "CoDExLarge": (1, 2000),
    "FB15k237": (1, 'null'),
    "WN18RR": (1, 'null'),
    "YAGO310": (1, 2000),
    "DBpedia100k": (1, 1000),
    "AristoV4": (1, 2000),
    "ConceptNet100k": (1, 2000),
    "ATOMIC": (1, 200),
    "NELL995": (1, 'null'),
    "Hetionet": (1, 4000),
    "WDsinger": (3, 'null'),
    "FB15k237_10": (1, 'null'),
    "FB15k237_20": (1, 'null'),
    "FB15k237_50": (1, 1000),
    "NELL23k": (3, 'null'),
    "FB15k237Inductive": (1, 'null'),
    "WN18RRInductive": (1, 'null'),
    "NELLInductive": (3, 'null'),
    "ILPC2022SmallInductive": (3, 'null'),
    "ILPC2022LargeInductive": (1, 1000),
    "NLIngram": (3, 'null'),
    "FBIngram": (3, 'null'),
    "WKIngram": (3, 'null'),
    "WikiTopicsMT1": (3, 'null'),
    "WikiTopicsMT2": (3, 'null'),
    "WikiTopicsMT3": (3, 'null'),
    "WikiTopicsMT4": (3, 'null'),
    "Metafam": (3, 'null'),
    "FBNELL": (3, 'null'),
    "HM": (1, 100)
}

seeds = [1024, 42, 1337, 512, 256]


def set_seed(seed):
    random.seed(seed + util.get_rank())
    torch.manual_seed(seed + util.get_rank())
    torch.cuda.manual_seed_all(seed + util.get_rank())
    np = __import__('numpy')
    np.random.seed(seed + util.get_rank())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_flags_with_ablation_config(flags_path, ablation_config):
    """更新 flags.yaml 文件以应用消融实验配置"""
    # 读取现有 flags
    with open(flags_path, 'r', encoding='utf-8') as f:
        flags_data = yaml.safe_load(f) or {}
    
    # 更新消融实验相关配置
    flags_data['run'] = ablation_config['run']
    flags_data['use_similarity_enhancer'] = ablation_config['use_similarity_enhancer']
    flags_data['use_prompt_enhancer'] = ablation_config['use_prompt_enhancer']
    flags_data['use_adaptive_gate'] = ablation_config['use_adaptive_gate']
    
    # 写回文件
    with open(flags_path, 'w', encoding='utf-8') as f:
        yaml.dump(flags_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def run_ablation_experiment(dataset, ablation_key, config_file, ckpt=None, 
                           finetune=False, train_from_scratch=False, 
                           repeats=1, seed_idx=0):
    """
    运行单个消融实验
    
    Args:
        dataset: 数据集名称（可能包含版本，如 "FB15k237Inductive:v1"）
        ablation_key: 消融实验配置键（如 "baseline", "similarity_only"）
        config_file: 配置文件路径
        ckpt: 检查点路径（用于微调或推理）
        finetune: 是否微调
        train_from_scratch: 是否从头训练
        repeats: 重复次数
        seed_idx: 种子索引
    """
    ablation_config = ABLATION_CONFIGS[ablation_key]
    
    # 解析数据集名称和版本
    if ':' in dataset:
        ds, version = dataset.split(':', 1)
    else:
        ds, version = dataset, None
    
    print(f"\n{'='*80}")
    print(f"运行消融实验: {ablation_config['name']}")
    print(f"数据集: {dataset}")
    print(f"描述: {ablation_config['description']}")
    print(f"{'='*80}\n")
    
    # 更新 flags.yaml
    flags_path = os.path.join(mydir, "flags.yaml")
    update_flags_with_ablation_config(flags_path, ablation_config)
    
    # 获取训练配置
    if finetune:
        config_dict = default_finetuning_config
        if version == "large":
            epochs, batch_per_epoch = 1, 1000
        else:
            epochs, batch_per_epoch = config_dict.get(ds, (1, 2000))
    elif train_from_scratch:
        config_dict = default_train_config
        epochs, batch_per_epoch = config_dict.get(ds, (10, 1000))
    else:
        epochs, batch_per_epoch = 0, 'null'
    
    # 准备配置变量
    vars = {
        'epochs': epochs,
        'bpe': batch_per_epoch,
        'dataset': ds,
        'gpus': [0]  # 默认使用 GPU 0
    }
    if version is not None:
        vars['version'] = version
    if ckpt is not None:
        vars['ckpt'] = ckpt
    
    # 加载配置
    cfg = parse.load_config(config_file, context=vars, root=mydir)
    
    # 创建工作目录
    root_dir = os.path.expanduser(cfg.output_dir)
    root_dir = os.path.join(mydir, root_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    # 为每个消融实验创建子目录
    ablation_dir = os.path.join(root_dir, f"{ablation_key}_{ds}")
    if version:
        ablation_dir = f"{ablation_dir}_{version}"
    if not os.path.exists(ablation_dir):
        os.makedirs(ablation_dir)
    
    os.chdir(ablation_dir)
    working_dir = util.create_working_directory(cfg)
    
    # 设置随机种子
    seed = seeds[seed_idx] if seed_idx < len(seeds) else random.randint(0, 10000)
    set_seed(seed)
    
    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning(f"消融实验: {ablation_config['name']}")
        logger.warning(f"数据集: {dataset}")
        logger.warning(f"随机种子: {seed}")
        logger.warning(f"配置: {ablation_config}")
    
    # 加载任务和数据集
    task_name = cfg.task["name"]
    dataset_obj = util.build_dataset(cfg)
    device = util.get_device(cfg)
    
    train_data, valid_data, test_data = dataset_obj[0], dataset_obj[1], dataset_obj[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    
    # 根据 flags 创建模型
    flags = parse.load_flags(os.path.join(mydir, "flags.yaml"))
    if flags.run == "semma":
        model = Ultra(
            rel_model_cfg=cfg.model.relation_model,
            entity_model_cfg=cfg.model.entity_model,
            sem_model_cfg=cfg.model.semantic_model,
        )
    elif flags.run == "EnhancedUltra":
        model = EnhancedUltra(
            rel_model_cfg=cfg.model.relation_model,
            entity_model_cfg=cfg.model.entity_model,
            sem_model_cfg=cfg.model.semantic_model,
        )
    else:
        model = Ultra(
            rel_model_cfg=cfg.model.relation_model,
            entity_model_cfg=cfg.model.entity_model,
        )
    
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
    
    model = model.to(device)
    
    # 运行训练或测试
    if train_from_scratch or finetune:
        # 处理过滤数据（用于归纳推理）
        val_filtered_data = None
        if task_name == "InductiveInference":
            val_filtered_data = dataset_obj.valid_filtered_data
        train_and_validate(cfg, model, train_data, valid_data, 
                          filtered_data=val_filtered_data, device=device, 
                          logger=logger, batch_per_epoch=cfg.train.batch_per_epoch)
        
        # 训练后进行评估
        if flags.eval_on_valid:
            if util.get_rank() == 0:
                logger.warning("=" * 80)
                logger.warning("在验证集上评估")
                logger.warning("=" * 80)
            metrics = test(cfg, model, valid_data, device, logger, 
                         filtered_data=val_filtered_data, return_metrics=True)
        else:
            if util.get_rank() == 0:
                logger.warning("=" * 80)
                logger.warning("在测试集上评估")
                logger.warning("=" * 80)
            test_filtered_data = None
            if task_name == "InductiveInference":
                test_filtered_data = dataset_obj.test_filtered_data
            metrics = test(cfg, model, test_data, device, logger, 
                         filtered_data=test_filtered_data, return_metrics=True)
        
        # 保存结果
        results = {
            'ablation': ablation_key,
            'name': ablation_config['name'],
            'dataset': dataset,
            'seed': seed,
            **metrics
        }
        return results
    else:
        # 只进行推理
        test_filtered_data = None
        if task_name == "InductiveInference":
            test_filtered_data = dataset_obj.test_filtered_data
        metrics = test(cfg, model, test_data, device, logger, 
                      filtered_data=test_filtered_data, return_metrics=True)
        
        # 保存结果
        results = {
            'ablation': ablation_key,
            'name': ablation_config['name'],
            'dataset': dataset,
            'seed': seed,
            **metrics
        }
        
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 EnhancedUltra 消融实验")
    parser.add_argument("-c", "--config", help="YAML配置文件", 
                       default="config/ablation/ablation_config.yaml", type=str)
    parser.add_argument("-d", "--datasets", help="目标数据集（逗号分隔）", 
                       default="FB15k237", type=str, required=True)
    parser.add_argument("-a", "--ablations", help="消融实验配置（逗号分隔，或'all'表示全部）", 
                       default="all", type=str)
    parser.add_argument("-ckpt", "--checkpoint", help="检查点路径（用于推理或微调）", 
                       default=None, type=str)
    parser.add_argument("-ft", "--finetune", help="微调检查点", action='store_true')
    parser.add_argument("-tr", "--train", help="从头训练", action='store_true')
    parser.add_argument("-reps", "--repeats", help="每个实验的重复次数", default=1, type=int)
    
    args = parser.parse_args()
    
    # 解析数据集
    datasets = args.datasets.split(",")
    
    # 解析消融实验配置
    if args.ablations.lower() == "all":
        ablation_keys = list(ABLATION_CONFIGS.keys())
    else:
        ablation_keys = [k.strip() for k in args.ablations.split(",")]
    
    # 验证消融实验配置
    for key in ablation_keys:
        if key not in ABLATION_CONFIGS:
            print(f"错误: 未知的消融实验配置 '{key}'")
            print(f"可用的配置: {list(ABLATION_CONFIGS.keys())}")
            sys.exit(1)
    
    # 创建结果文件
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    results_file = os.path.join(mydir, f"ablation_results_{timestamp}.csv")
    
    all_results = []
    
    # 运行所有消融实验
    for dataset in datasets:
        for ablation_key in ablation_keys:
            for rep in range(args.repeats):
                try:
                    result = run_ablation_experiment(
                        dataset=dataset,
                        ablation_key=ablation_key,
                        config_file=args.config,
                        ckpt=args.checkpoint,
                        finetune=args.finetune,
                        train_from_scratch=args.train,
                        repeats=args.repeats,
                        seed_idx=rep
                    )
                    
                    if result:
                        all_results.append(result)
                        
                        # 保存中间结果
                        if all_results:
                            with open(results_file, 'w', newline='', encoding='utf-8') as f:
                                if all_results:
                                    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                                    writer.writeheader()
                                    writer.writerows(all_results)
                    
                except Exception as e:
                    print(f"错误: 运行消融实验失败 ({ablation_key}, {dataset}, rep {rep+1})")
                    print(f"错误信息: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print(f"\n{'='*80}")
    print(f"消融实验完成！结果已保存到: {results_file}")
    print(f"{'='*80}\n")

