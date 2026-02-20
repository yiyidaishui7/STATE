#!/bin/bash
# run_and_eval.sh - 训练多组实验 + 自动量化评估
#
# 使用: cd ~/state/src && bash run_and_eval.sh
#
# 每组实验训练完成后自动运行评估，最后统一对比
# // turbo-all

echo "========================================"
echo "MMD-AAE 批量训练 + 量化评估"
echo "========================================"

cd /media/mldadmin/home/s125mdg34_03/state/src

# 实验配置: exp_name lambda_recon lambda_mmd lambda_adv epochs latent_dim [extra_args]
# lambda_mmd=0 表示使用自动校准, mmd_target_ratio 控制 MMD 相对权重
EXPERIMENTS=(
    "v3_auto_r05    1.0  0  0.1  50  64  --mmd_target_ratio 0.5"
    "v3_auto_r1     1.0  0  0.1  50  64  --mmd_target_ratio 1.0"
    "v3_auto_r2     1.0  0  0.1  50  64  --mmd_target_ratio 2.0"
    "v3_auto_r5     1.0  0  0.1  50  64  --mmd_target_ratio 5.0"
    "v3_auto_r10    1.0  0  0.1  50  64  --mmd_target_ratio 10.0"
    "v3_auto_l32    1.0  0  0.1  50  32  --mmd_target_ratio 1.0"
    "v3_auto_l128   1.0  0  0.1  50  128 --mmd_target_ratio 1.0"
    "v3_auto_a05    1.0  0  0.5  50  64  --mmd_target_ratio 1.0"
)

TOTAL=${#EXPERIMENTS[@]}
echo "共 $TOTAL 组实验"
echo ""

for i in "${!EXPERIMENTS[@]}"; do
    # 解析前 6 个位置参数 + 剩余额外参数
    PARTS=(${EXPERIMENTS[$i]})
    EXP_NAME="${PARTS[0]}"
    LR="${PARTS[1]}"
    LM="${PARTS[2]}"
    LA="${PARTS[3]}"
    EPOCHS="${PARTS[4]}"
    LD="${PARTS[5]}"
    EXTRA="${PARTS[@]:6}"  # 额外参数 (如 --mmd_target_ratio 1.0)
    
    NUM=$((i + 1))
    echo "========================================"
    echo "[$NUM/$TOTAL] 实验: $EXP_NAME"
    echo "  λ_recon=$LR, λ_mmd=$LM, λ_adv=$LA"
    echo "  epochs=$EPOCHS, latent_dim=$LD"
    echo "  extra: $EXTRA"
    echo "========================================"
    
    python train_mmd_aae.py \
        --exp_name "$EXP_NAME" \
        --lambda_recon "$LR" \
        --lambda_mmd "$LM" \
        --lambda_adv "$LA" \
        --epochs "$EPOCHS" \
        --latent_dim "$LD" \
        $EXTRA
    
    echo "[$NUM/$TOTAL] 训练完成，运行评估..."
    python evaluate_mmd_aae.py --exp_name "$EXP_NAME" --samples 2000
    echo ""
done

echo ""
echo "========================================"
echo "✅ 所有实验完成! 运行最终对比..."
echo "========================================"

# 最终全部对比
python evaluate_mmd_aae.py --all --samples 2000

echo ""
echo "完成! 查看 evaluations/ 目录下的结果文件"
