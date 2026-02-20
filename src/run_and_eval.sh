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

# 实验配置: exp_name lambda_recon lambda_mmd lambda_adv epochs latent_dim
# 论文推荐: recon=1.0, adv=0.1, mmd=2500.0
EXPERIMENTS=(
    "v3_m1000_l64   1.0  1000.0  0.1  50  64"
    "v3_m2500_l64   1.0  2500.0  0.1  50  64"
    "v3_m5000_l64   1.0  5000.0  0.1  50  64"
    "v3_m10000_l64  1.0  10000.0 0.1  50  64"
    "v3_m2500_l32   1.0  2500.0  0.1  50  32"
    "v3_m2500_l128  1.0  2500.0  0.1  50  128"
    "v3_r05_m2500   0.5  2500.0  0.1  50  64"
    "v3_r2_m2500    2.0  2500.0  0.1  50  64"
)

TOTAL=${#EXPERIMENTS[@]}
echo "共 $TOTAL 组实验"
echo ""

for i in "${!EXPERIMENTS[@]}"; do
    read -r EXP_NAME LR LM LA EPOCHS LD <<< "${EXPERIMENTS[$i]}"
    
    NUM=$((i + 1))
    echo "========================================"
    echo "[$NUM/$TOTAL] 实验: $EXP_NAME"
    echo "  λ_recon=$LR, λ_mmd=$LM, λ_adv=$LA"
    echo "  epochs=$EPOCHS, latent_dim=$LD"
    echo "========================================"
    
    python train_mmd_aae.py \
        --exp_name "$EXP_NAME" \
        --lambda_recon "$LR" \
        --lambda_mmd "$LM" \
        --lambda_adv "$LA" \
        --epochs "$EPOCHS" \
        --latent_dim "$LD"
    
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
