#!/bin/bash
# run_experiments_v2.sh - 架构修复版实验
#
# 使用: cd ~/state/src && bash run_experiments_v2.sh
#
# tmux 操作:
#   tmux ls                    # 查看 session
#   tmux attach -t v2_default  # 进入实验
#   Ctrl+b d                   # 退出 (不关闭)

echo "========================================"
echo "MMD-AAE v2 实验 (架构修复版)"
echo "========================================"

cd /media/mldadmin/home/s125mdg34_03/state/src
ACTIVATE_CMD="source activate state_env"

# 实验 1: 默认配置
SESSION="v2_default"
echo "启动: $SESSION (λ_m=10, epochs=50)"
tmux new-session -d -s $SESSION
tmux send-keys -t $SESSION "$ACTIVATE_CMD" C-m
tmux send-keys -t $SESSION "cd /media/mldadmin/home/s125mdg34_03/state/src" C-m
tmux send-keys -t $SESSION "python train_mmd_aae.py --lambda_mmd 10.0 --epochs 50 --exp_name v2_m10_ep50" C-m

# 实验 2: 更大 MMD
SESSION="v2_m20"
echo "启动: $SESSION (λ_m=20, epochs=50)"
tmux new-session -d -s $SESSION
tmux send-keys -t $SESSION "$ACTIVATE_CMD" C-m
tmux send-keys -t $SESSION "cd /media/mldadmin/home/s125mdg34_03/state/src" C-m
tmux send-keys -t $SESSION "python train_mmd_aae.py --lambda_mmd 20.0 --epochs 50 --exp_name v2_m20_ep50" C-m

# 实验 3: 更小 recon + 大 MMD
SESSION="v2_r05_m20"
echo "启动: $SESSION (λ_r=0.5, λ_m=20, epochs=50)"
tmux new-session -d -s $SESSION
tmux send-keys -t $SESSION "$ACTIVATE_CMD" C-m
tmux send-keys -t $SESSION "cd /media/mldadmin/home/s125mdg34_03/state/src" C-m
tmux send-keys -t $SESSION "python train_mmd_aae.py --lambda_recon 0.5 --lambda_mmd 20.0 --epochs 50 --exp_name v2_r05_m20_ep50" C-m

echo ""
echo "========================================"
echo "✅ 3 个实验已启动!"
echo "========================================"
echo ""
echo "查看: tmux ls"
echo "进入: tmux attach -t v2_default"
echo ""
echo "完成后可视化:"
echo "  python visualize_mmd_aae.py --exp_name v2_m10_ep50"
echo "  python visualize_mmd_aae.py --exp_name v2_m20_ep50"
echo "  python visualize_mmd_aae.py --exp_name v2_r05_m20_ep50"
echo "========================================"
