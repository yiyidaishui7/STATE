#!/bin/bash

#############################################################
# Baseline训练脚本 - 严格Zero-shot版本
# 训练集：K562 + Jurkat + RPE1
# 验证集：HepG2
# 测试集：H1
#############################################################

set -e

# ============================================
# 配置
# ============================================
CONFIG_PATH="configs/baseline_config_new.toml"
OUTPUT_DIR="results/baseline_new_$(date +%Y%m%d_%H%M%S)"
PROJECT_DIR="./"

# ============================================
# 检查环境
# ============================================
echo "=========================================="
echo "Baseline训练 - 严格Zero-shot"
echo "=========================================="

# 检查是否在state目录
cd $PROJECT_DIR
if [ ! -f "pyproject.toml" ]; then
    echo "❌ 错误：不在state项目目录"
    exit 1
fi

# 检查配置文件
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

# 检查数据文件
echo "检查数据文件..."
for file in k562.h5 jurkat.h5 rpe1.h5 hepg2.h5; do
    if [ -f "competition_support_set/$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ❌ $file 不存在"
        exit 1
    fi
done

# 检查H1是否存在
if [ -f "competition_support_set/h1.h5" ]; then
    echo "  ✓ h1.h5 (将作为测试集)"
    HAS_H1=true
else
    echo "  ⚠️  h1.h5 不存在（跳过测试集）"
    HAS_H1=false
fi

# 检查ESM2特征文件
if [ ! -f "competition_support_set/ESM2_pert_features.pt" ]; then
    echo "❌ ESM2特征文件不存在"
    exit 1
fi
echo "  ✓ ESM2_pert_features.pt"

# GPU信息
echo ""
echo "GPU信息:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 备份配置
cp $CONFIG_PATH $OUTPUT_DIR/config_used.toml

# ============================================
# 训练配置
# ============================================
echo ""
echo "=========================================="
echo "训练配置"
echo "输出目录: $OUTPUT_DIR"
echo "开始时间: $(date)"
echo "=========================================="

# ============================================
# 运行训练
# ============================================
echo ""
echo "开始训练..."
echo ""

uv run state tx train \
  data.kwargs.toml_config_path="$CONFIG_PATH" \
  data.kwargs.num_workers=8 \
  data.kwargs.batch_col="batch_var" \
  data.kwargs.pert_col="target_gene" \
  data.kwargs.cell_type_key="cell_type" \
  data.kwargs.control_pert="non-targeting" \
  data.kwargs.perturbation_features_file="competition_support_set/ESM2_pert_features.pt" \
  training.max_steps=40000 \
  training.val_freq=200 \
  training.ckpt_every_n_steps=2000 \
  model=state_sm \
  wandb.tags="[baseline_new,zero_shot,dg_experiment]" \
  wandb.project=baseline_dg_new \
  output_dir="$OUTPUT_DIR" \
  name="baseline_new" \
  2>&1 | tee $OUTPUT_DIR/training.log

TRAIN_EXIT_CODE=$?


# ============================================
# 训练完成
# ============================================
echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ 训练完成！"
    
    # 生成可视化图表
    echo ""
    echo "生成可视化图表..."
    python scripts/plot_training.py --log-dir $OUTPUT_DIR
    
    echo ""
    echo "✓ 图表已保存到: $OUTPUT_DIR/plots/"
else
    echo "❌ 训练失败，退出码: $TRAIN_EXIT_CODE"
fi

echo ""
echo "结束时间: $(date)"
echo "结果保存在: $OUTPUT_DIR"
echo "=========================================="

exit $TRAIN_EXIT_CODE