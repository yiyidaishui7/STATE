#!/usr/bin/env python
"""
run_mmd_aae_train.py - MMD-AAE 训练入口脚本

使用方法:
    cd ~/state/src
    python run_mmd_aae_train.py
    
    # 或指定配置文件:
    python run_mmd_aae_train.py --config ../configs/mmd_aae_config.yaml
"""
import os
import sys
import argparse
import logging

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="MMD-AAE Training")
    parser.add_argument("--config", type=str, 
                        default="../configs/mmd_aae_config.yaml",
                        help="Path to config YAML file")
    args = parser.parse_args()
    
    # 加载配置
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    if not os.path.exists(config_path):
        log.error(f"配置文件不存在: {config_path}")
        sys.exit(1)
    
    log.info(f"加载配置: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # 验证必要的配置字段
    if not hasattr(cfg, 'embeddings') or not hasattr(cfg.embeddings, 'current'):
        log.error("配置缺少 embeddings.current")
        sys.exit(1)
    
    if not hasattr(cfg, 'dataset') or not hasattr(cfg.dataset, 'current'):
        log.error("配置缺少 dataset.current")
        sys.exit(1)
    
    log.info(f"Embeddings: {cfg.embeddings.current}")
    log.info(f"Dataset: {cfg.dataset.current}")
    
    # 设置环境变量
    if hasattr(cfg.experiment, 'port'):
        os.environ["MASTER_PORT"] = str(cfg.experiment.port)
    if hasattr(cfg.experiment, 'num_gpus_per_node'):
        os.environ["SLURM_NTASKS_PER_NODE"] = str(cfg.experiment.num_gpus_per_node)
    
    # 导入并运行训练
    log.info(f"开始训练: {cfg.experiment.name}")
    
    from state.emb.train.trainer import main as train_main
    train_main(cfg)


if __name__ == "__main__":
    main()
