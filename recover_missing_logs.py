import os
import glob
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import subprocess
import re

# ==================== 1. 配置 (保持与原代码一致) ====================

DATA_ROOT = "competition_support_set" 
OUTPUT_ROOT = "competition_hepg2_2"
METRICS_FILE = f"{OUTPUT_ROOT}/final_score_log.csv"
PLOT_DIR = f"{OUTPUT_ROOT}/plots"
CHECKPOINT_DIR = f"{OUTPUT_ROOT}/run_3090/checkpoints"

# 验证集和训练集路径
VAL_DATA_PATH = f"{DATA_ROOT}/hepg2_val_with_controls.h5ad"
TRAIN_SUBSET_PATH = f"{DATA_ROOT}/train_subset_monitor_mixed.h5ad"

# Baseline
BASELINE_DES = 0.106
BASELINE_PDS = 0.514
BASELINE_MAE = 0.027

# ==================== 2. 复用你的核心函数 ====================

def calculate_competition_scores(pred_path, true_path):
    # ... (完全复用你的逻辑) ...
    try:
        adata_pred = sc.read_h5ad(pred_path)
        adata_true = sc.read_h5ad(true_path)
                
        if 'target_gene' in adata_true.obs:
            mask = adata_true.obs['target_gene'] != 'non-targeting'
            if mask.sum() == 0: return None
            adata_true = adata_true[mask]
            adata_pred = adata_pred[mask]
        
        common_genes = np.intersect1d(adata_pred.var_names, adata_true.var_names)
        if len(common_genes) == 0: return None
        
        X_pred = adata_pred[:, common_genes].X
        X_true = adata_true[:, common_genes].X
        
        if hasattr(X_pred, "toarray"): X_pred = X_pred.toarray()
        if hasattr(X_true, "toarray"): X_true = X_true.toarray()
        
        raw_mae = mean_absolute_error(X_true, X_pred)
        raw_des = r2_score(X_true.flatten(), X_pred.flatten())
        
        n_samples = min(500, X_pred.shape[0])
        idx = np.random.choice(X_pred.shape[0], n_samples, replace=False)
        corrs = []
        for i in idx:
            if np.std(X_pred[i]) == 0 or np.std(X_true[i]) == 0: corrs.append(0)
            else: corrs.append(pearsonr(X_pred[i], X_true[i])[0])
        raw_pds = np.nanmean(corrs)
        
        des_scaled = max(0, (raw_des - BASELINE_DES) / (1 - BASELINE_DES))
        pds_scaled = max(0, (raw_pds - BASELINE_PDS) / (1 - BASELINE_PDS))
        mae_scaled = max(0, (BASELINE_MAE - raw_mae) / BASELINE_MAE)
        score_s = (1/3) * (des_scaled + pds_scaled + mae_scaled) * 100
        
        return {
            "Raw_MAE": raw_mae, "Raw_DES": raw_des, "Raw_PDS": raw_pds,
            "Scaled_MAE": mae_scaled, "Scaled_DES": des_scaled, "Scaled_PDS": pds_scaled,
            "Score_S": score_s
        }
    except Exception as e:
        print(f"❌ 计算错误: {e}")
        return None

def run_inference(ckpt, input_adata):
    out_file = f"{OUTPUT_ROOT}/temp_pred_recover_{os.path.basename(ckpt)}.h5ad"
    # 注意：这里假设你在相同的环境下运行，可以直接调用 uv
    cmd = [
        "uv", "run", "state", "tx", "infer",
        f"--output={out_file}",
        f"--model-dir={OUTPUT_ROOT}/run_3090",
        f"--checkpoint={ckpt}",
        f"--adata={input_adata}",
        "--pert-col=target_gene"
    ]
    
    # 强制使用 GPU 1 (或者你可以改成 0，因为现在没有训练在跑了)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0" # <--- 恢复模式下可以直接用 0 号卡，因为没有训练进程抢占
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, env=env)
        return out_file
    except subprocess.CalledProcessError as e:
        print(f"❌ 推理失败 ({os.path.basename(ckpt)}): {e}")
        return None

def save_dashboard(history_df, step):
    if history_df.empty: return
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.switch_backend('Agg') 
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    steps = history_df['step']
    
    # Plot 1: Overall Score
    ax = axes[0, 0]
    if 'val_Score_S' in history_df.columns:
        ax.plot(steps, history_df['val_Score_S'], 'r-o', linewidth=2, label='Val Score (Test)')
        max_s = history_df['val_Score_S'].max()
        ax.axhline(y=max_s, color='red', linestyle=':', alpha=0.3)
        ax.text(steps.iloc[-1], max_s, f" Val Max: {max_s:.2f}", va='center', color='red')
    if 'train_Score_S' in history_df.columns:
        ax.plot(steps, history_df['train_Score_S'], 'b--x', linewidth=1.5, alpha=0.6, label='Train Score (Fit)')
    ax.set_title("Overall Score: Train vs Val")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Components
    ax = axes[0, 1]
    if 'val_Scaled_DES' in history_df.columns:
        ax.plot(steps, history_df['val_Scaled_DES'], label='Val DES', color='tab:blue')
        ax.plot(steps, history_df['val_Scaled_PDS'], label='Val PDS', color='tab:orange')
        ax.plot(steps, history_df['val_Scaled_MAE'], label='Val MAE', color='tab:green')
    ax.set_title("Val Score Composition")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Raw MAE
    ax = axes[1, 0]
    if 'val_Raw_MAE' in history_df.columns:
        ax.plot(steps, history_df['val_Raw_MAE'], 'r-o', label='Val MAE')
    if 'train_Raw_MAE' in history_df.columns:
        ax.plot(steps, history_df['train_Raw_MAE'], 'b--x', label='Train MAE')
    ax.axhline(y=BASELINE_MAE, color='black', linestyle='--', label='Baseline')
    ax.set_title("Raw MAE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Bio Signals
    ax = axes[1, 1]
    if 'val_Raw_PDS' in history_df.columns:
        ax.plot(steps, history_df['val_Raw_PDS'], color='tab:orange', label='Val PDS')
    if 'val_Raw_DES' in history_df.columns:
        ax.plot(steps, history_df['val_Raw_DES'], color='tab:blue', label='Val DES')
    ax.set_title("Biological Signals")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Training Monitor - Step {step}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/latest_score_card.png")
    plt.savefig(f"{PLOT_DIR}/step_{step:05d}.png")
    plt.close()

# ==================== 3. 恢复逻辑主程序 ====================

if __name__ == "__main__":
    print(f"🚑 启动恢复模式: {CHECKPOINT_DIR}")
    
    # 1. 读取现有的 Log
    if os.path.exists(METRICS_FILE):
        history_df = pd.read_csv(METRICS_FILE)
        processed_steps = set(history_df['step'].unique())
        print(f"✅ 已记录的 Steps: {sorted(list(processed_steps))}")
    else:
        print("⚠️ 未找到 Log 文件，将创建新的。")
        history_df = pd.DataFrame()
        processed_steps = set()

    # 2. 扫描所有 Checkpoint
    raw_ckpts = glob.glob(f"{CHECKPOINT_DIR}/*.ckpt")
    ckpts_to_process = []
    
    for c in raw_ckpts:
        filename = os.path.basename(c)
        # 过滤 val_loss 和 重复项
        if "val_loss" in filename or "step=step" in filename: continue
        
        m = re.search(r"step=(\d+)", filename)
        if m: 
            step_num = int(m.group(1))
            # 关键：只处理 Log 中没有的步数
            if step_num not in processed_steps:
                ckpts_to_process.append((step_num, c))
    
    ckpts_to_process.sort(key=lambda x: x[0])
    
    if not ckpts_to_process:
        print("🎉 没有发现缺失的日志，所有 checkpoint 都已处理完毕！")
        exit()

    print(f"🔥 发现 {len(ckpts_to_process)} 个缺失的 Checkpoints 需要处理: {[x[0] for x in ckpts_to_process]}")

    # 3. 循环补全
    for step, ckpt_path in ckpts_to_process:
        print(f"\n[{pd.Timestamp.now().strftime('%H:%M')}] ♻️  Recovering Step {step} ...")
        
        row = {'step': step, 'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
        
        # --- Val ---
        if os.path.exists(VAL_DATA_PATH):
            pred_val = run_inference(ckpt_path, VAL_DATA_PATH)
            if pred_val:
                val_m = calculate_competition_scores(pred_val, VAL_DATA_PATH)
                if val_m:
                    for k, v in val_m.items(): row[f"val_{k}"] = v
                    print(f"   -> Val Score: {val_m['Score_S']:.2f}")
                if os.path.exists(pred_val): os.remove(pred_val)
        
        # --- Train ---
        if os.path.exists(TRAIN_SUBSET_PATH):
            pred_train = run_inference(ckpt_path, TRAIN_SUBSET_PATH)
            if pred_train:
                train_m = calculate_competition_scores(pred_train, TRAIN_SUBSET_PATH)
                if train_m:
                    for k, v in train_m.items(): row[f"train_{k}"] = v
                if os.path.exists(pred_train): os.remove(pred_train)

        # 4. 追加写入 CSV (防止再次中断)
        df_row = pd.DataFrame([row])
        # 使用 concat 替代 append
        history_df = pd.concat([history_df, df_row], ignore_index=True)
        # 按 step 排序，保证 CSV 顺序正确
        history_df = history_df.sort_values(by='step') 
        history_df.to_csv(METRICS_FILE, index=False)
        print(f"   ✅ Log updated for step {step}")

        # 5. 更新图表
        save_dashboard(history_df, step)
    
    print("\n✅ 所有缺失数据已补全！")