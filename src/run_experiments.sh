#!/bin/bash
# run_experiments.sh - 在 tmux 中运行多个实验
#
# 使用方法:
#   cd ~/state/src
#   bash run_experiments.sh
#
# 查看实验:
#   tmux ls                    # 列出所有 session
#   tmux attach -t exp_m10_50  # 进入某个 session
#   Ctrl+b d                   # 退出 session (不关闭)
#   tmux kill-session -t exp_m10_50  # 关闭某个 session

echo "========================================"
echo "启动 MMD-AAE 实验 (tmux)"
echo "========================================"

# 进入工作目录
cd /media/mldadmin/home/s125mdg34_03/state/src

# 激活环境 (根据你的环境名调整)
ACTIVATE_CMD="conda activate state_env"

# ========================================
# 实验 1: λ_mmd=10, λ_a=0.1, epochs=200
# ========================================
SESSION_NAME="exp_r1_m10_a01_ep200"
echo "创建 tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "$ACTIVATE_CMD" C-m
tmux send-keys -t $SESSION_NAME "cd /media/mldadmin/home/s125mdg34_03/state/src" C-m
tmux send-keys -t $SESSION_NAME "python train_mmd_aae.py --lambda_recon 1 --lambda_mmd 10.0 --lambda_adv 0.1 --epochs 200 --exp_name m10_a01_ep200" C-m
echo "  -> 已启动: λ_recon=1, λ_mmd=10, λ_adv=0.1, epochs=200"

# ========================================
# 实验 2: λ_mmd=10, epochs=200
# ========================================
SESSION_NAME="exp_r1_m10_a05_ep200"
echo "创建 tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "$ACTIVATE_CMD" C-m
tmux send-keys -t $SESSION_NAME "cd /media/mldadmin/home/s125mdg34_03/state/src" C-m
tmux send-keys -t $SESSION_NAME "python train_mmd_aae.py --lambda_recon 1 --lambda_mmd 10.0  --lambda_adv 0.5 --epochs 200 --exp_name m10_a0.5_ep200" C-m
echo "  -> 已启动: λ_recon=1, λ_mmd=10, λ_adv=0.5, epochs=200"

# ========================================
# 实验 3: λ_mmd=5, epochs=200 (更温和的对齐)
# ========================================
SESSION_NAME="exp_r1_m5_a0.5_ep200"
echo "创建 tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "$ACTIVATE_CMD" C-m
tmux send-keys -t $SESSION_NAME "cd /media/mldadmin/home/s125mdg34_03/state/src" C-m
tmux send-keys -t $SESSION_NAME "python train_mmd_aae.py --lambda_recon 1 --lambda_mmd 5.0  --lambda_adv 0.5 --epochs 200 --exp_name m5_a0.5_ep200" C-m
echo "  -> 已启动: λ_recon=1, λ_mmd=5, λ_adv=0.5, epochs=200"

# ========================================
# 实验 4: λ_recon=0.5, λ_mmd=10 (降低重建权重)
# ========================================
SESSION_NAME="exp_r05_m10_a05"
echo "创建 tmux session: $SESSION_NAME"
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "$ACTIVATE_CMD" C-m
tmux send-keys -t $SESSION_NAME "cd /media/mldadmin/home/s125mdg34_03/state/src" C-m
tmux send-keys -t $SESSION_NAME "python train_mmd_aae.py --lambda_recon 0.5 --lambda_mmd 10.0 -- lambda_adv 0.5 --epochs 200 --exp_name r05_m10_a05_ep200" C-m
echo "  -> 已启动: λ_recon=0.5, λ_mmd=10, λ_adv=0.5, epochs=200"

echo ""
echo "========================================"
echo "✅ 所有实验已在后台启动!"
echo "========================================"
echo ""
echo "常用命令:"
echo "  tmux ls                      # 查看所有 session"
echo "  tmux attach -t exp_r1_m10_a05_ep200  # 进入查看某个实验"
echo "  Ctrl+b d                     # 退出 session (实验继续运行)"
echo "  tmux kill-session -t <name>  # 停止某个实验"
echo ""
echo "实验完成后查看可视化:"
echo "  python visualize_mmd_aae.py --exp_name r1_m10_a01_ep200"
echo "  python visualize_mmd_aae.py --exp_name r1_m10_a05_ep200"
echo "  python visualize_mmd_aae.py --exp_name r1_m5_a05_ep200"
echo "  python visualize_mmd_aae.py --exp_name r05_m10_a05_ep200"
echo "========================================"
