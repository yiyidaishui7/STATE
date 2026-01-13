import os
import glob
import time
import shutil
import subprocess
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use('Agg') # 服务器无头模式
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

# ==================== 1. 基础配置 ====================

DATA_ROOT = "competition_support_set" 
OUTPUT_ROOT = "competition_40000_500"
METRICS_FILE = f"{OUTPUT_ROOT}/final_score_log.csv"
PLOT_DIR = f"{OUTPUT_ROOT}/plots"
TRAIN_LOG_FILE = f"{OUTPUT_ROOT}/training.log"

# ===  官方baseline值 (User Provided) ===
BASELINE_DES = 0.106
BASELINE_PDS = 0.514
BASELINE_MAE = 0.027

# ==================== 2. 环境准备 ====================

if not os.path.exists(DATA_ROOT):
    # 自动下载数据的逻辑可以保留，以防万一
    raise FileNotFoundError(f"❌ 未找到数据目录: {DATA_ROOT}")

# 强制生成配置文件
CONFIG_PATH = os.path.join(DATA_ROOT, "server_config_2.toml")
with open(CONFIG_PATH, "w") as f:
    f.write(f"""
# Dataset paths - maps dataset names to their directories
[datasets]
replogle_h1 = "competition_support_set/{{hepg2,rpe1,jurkat,k562}}.h5"

# Training specifications
# All cell types in a dataset automatically go into training (excluding zeroshot/fewshot overrides)
[training]
replogle_h1 = "train"

# Zeroshot specifications - entire cell types go to val or test
[zeroshot]
"replogle_h1.hepg2" = "val"


# Fewshot specifications - explicit perturbation lists
[fewshot]
""")

TRAIN_CMD = [
    "uv", "run", "state", "tx", "train",
    f"data.kwargs.toml_config_path={CONFIG_PATH}",
    "data.kwargs.num_workers=8",
    "data.kwargs.batch_col=batch_var",
    "data.kwargs.pert_col=target_gene",
    "data.kwargs.cell_type_key=cell_type",
    "data.kwargs.control_pert=non-targeting",
    f"data.kwargs.perturbation_features_file={DATA_ROOT}/ESM2_pert_features.pt",
    "training.max_steps=40000",
    "training.ckpt_every_n_steps=500",
    "training.val_freq=500",
    "model=state_sm",
    "+wandb.mode=offline",
    f"output_dir={OUTPUT_ROOT}",
    "name=run_3090"
]
CHECKPOINT_DIR = f"{OUTPUT_ROOT}/run_3090/checkpoints"
# VAL_DATA_PATH = f"{DATA_ROOT}/hepg2_val_pure.h5ad"
VAL_DATA_PATH = f"{DATA_ROOT}/hepg2_val_with_controls.h5ad"

TRAIN_SUBSET_PATH = f"{DATA_ROOT}/train_subset_monitor_mixed.h5ad"

# ==================== 3. 核心计算逻辑 ====================

# def ensure_train_subset():
#     if os.path.exists(TRAIN_SUBSET_PATH): return
#     print("⏳ 生成训练集采样...")
#     try:
#         src = os.path.join(DATA_ROOT, "competition_train.h5")
#         if not os.path.exists(src): src = os.path.join(DATA_ROOT, "k562_gwps.h5")
#         adata = sc.read_h5ad(src)
#         if adata.n_obs > 2000:
#             idx = np.random.choice(adata.n_obs, 2000, replace=False)
#             adata = adata[idx].copy()
#         adata.var_names_make_unique()
#         adata.write(TRAIN_SUBSET_PATH)
#     except Exception as e:
#         print(f"⚠️ 采样失败: {e}")


def calculate_competition_scores(pred_path, true_path):
    """
    计算 Raw Metrics 并转换为 Scaled Scores 和 Overall S
    """
    try:
        adata_pred = sc.read_h5ad(pred_path)
        adata_true = sc.read_h5ad(true_path)
                
        # ==========================================
        # 🛑 [关键修改] 算分前剔除 Control (non-targeting)
        # ==========================================
        # 我们只关心模型对"微扰"的预测能力，不关心它复现 Control 的能力
        if 'target_gene' in adata_true.obs:
            mask = adata_true.obs['target_gene'] != 'non-targeting'
            
            # 如果剔除后没剩啥了（防止报错）
            if mask.sum() == 0:
                return None
            
            adata_true = adata_true[mask]
            adata_pred = adata_pred[mask] # 预测结果也要对应过滤
        # ==========================================
        
        common_genes = np.intersect1d(adata_pred.var_names, adata_true.var_names)
        if len(common_genes) == 0: return None
        
        X_pred = adata_pred[:, common_genes].X
        X_true = adata_true[:, common_genes].X
        
        if hasattr(X_pred, "toarray"): X_pred = X_pred.toarray()
        if hasattr(X_true, "toarray"): X_true = X_true.toarray()
        
        # --- 1. 计算 Raw Metrics ---
        # MAE
        raw_mae = mean_absolute_error(X_true, X_pred)
        
        # DES Proxy (R2)
        # 注意：官方DES是基因集合重叠度，R2是其强相关代理指标
        raw_des = r2_score(X_true.flatten(), X_pred.flatten())
        
        # PDS Proxy (Pearson Correlation)
        # 注意：官方PDS是基于L1距离的排名，Correlation是其强相关代理指标
        n_samples = min(500, X_pred.shape[0])
        idx = np.random.choice(X_pred.shape[0], n_samples, replace=False)
        corrs = []
        for i in idx:
            if np.std(X_pred[i]) == 0 or np.std(X_true[i]) == 0: corrs.append(0)
            else: corrs.append(pearsonr(X_pred[i], X_true[i])[0])
        raw_pds = np.nanmean(corrs)
        
        # --- 2. 计算 Scaled Scores (官方公式) ---
        
        # DES Scaled: (Your Model DES – Baseline) / (1 – Baseline)
        des_scaled = (raw_des - BASELINE_DES) / (1 - BASELINE_DES)
        des_scaled = max(0, des_scaled) # Negative clipped to 0

        # PDS Scaled: (Your Model PDS – Baseline) / (1 – Baseline)
        pds_scaled = (raw_pds - BASELINE_PDS) / (1 - BASELINE_PDS)
        pds_scaled = max(0, pds_scaled)

        # MAE Scaled: (Baseline - Your Model MAE) / Baseline
        mae_scaled = (BASELINE_MAE - raw_mae) / BASELINE_MAE
        mae_scaled = max(0, mae_scaled)

        # --- 3. Overall Score ---
        # S = 1/3 (DES_scaled + PDS_scaled + MAE_scaled) * 100
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
    out_file = f"{OUTPUT_ROOT}/temp_pred_{os.path.basename(ckpt)}.h5ad"
    cmd = [
        "uv", "run", "state", "tx", "infer",
        f"--output={out_file}",
        f"--model-dir={OUTPUT_ROOT}/run_3090",
        f"--checkpoint={ckpt}",
        f"--adata={input_adata}",
        "--pert-col=target_gene"
    ]
    
    # ==========================================
    #  [关键修改] 指定推理使用 GPU 1
    # ==========================================
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "3"  # <--- 强制使用 1 号卡
    
    try:
        # 注意这里增加了 env=env 参数
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, env=env)
        return out_file
    except subprocess.CalledProcessError as e:
        print(f"❌ 推理失败: {e}")
        return None
        
def save_dashboard(history_df, step):
    """
    无 Emoji 版：修复服务器字体缺失警告
    """
    if history_df.empty: return
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # 确保使用无头模式
    plt.switch_backend('Agg') 
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    steps = history_df['step']
    
    # ==========================================
    # 图 1: Overall Score S
    # ==========================================
    ax = axes[0, 0]
    # Val (红实线)
    if 'val_Score_S' in history_df.columns:
        ax.plot(steps, history_df['val_Score_S'], 'r-o', linewidth=2, label='Val Score (Test)')
        max_s = history_df['val_Score_S'].max()
        ax.axhline(y=max_s, color='red', linestyle=':', alpha=0.3)
        ax.text(steps.iloc[-1], max_s, f" Val Max: {max_s:.2f}", va='center', color='red', fontweight='bold')
    
    # Train (蓝虚线)
    if 'train_Score_S' in history_df.columns:
        ax.plot(steps, history_df['train_Score_S'], 'b--x', linewidth=1.5, alpha=0.6, label='Train Score (Fit)')
        
    ax.set_title("Overall Score: Train vs Val") # 🏆 已删除
    ax.set_ylabel("Score (0-100)")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # ==========================================
    # 图 2: Val Score Components
    # ==========================================
    ax = axes[0, 1]
    if 'val_Scaled_DES' in history_df.columns:
        ax.plot(steps, history_df['val_Scaled_DES'], label='Val DES (R2)', color='tab:blue')
        ax.plot(steps, history_df['val_Scaled_PDS'], label='Val PDS (Corr)', color='tab:orange')
        ax.plot(steps, history_df['val_Scaled_MAE'], label='Val MAE (Error)', color='tab:green', linewidth=2)
    
    ax.set_title("Val Score Composition") # 📊 已删除
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ==========================================
    # 图 3: Raw MAE
    # ==========================================
    ax = axes[1, 0]
    if 'val_Raw_MAE' in history_df.columns:
        ax.plot(steps, history_df['val_Raw_MAE'], 'r-o', label='Val MAE')
    if 'train_Raw_MAE' in history_df.columns:
        ax.plot(steps, history_df['train_Raw_MAE'], 'b--x', alpha=0.6, label='Train MAE')
        
    ax.axhline(y=BASELINE_MAE, color='black', linestyle='--', linewidth=2, label=f'Baseline ({BASELINE_MAE})')
    
    ax.set_title("Raw MAE (Lower is Better)") # 📉 已删除
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ==========================================
    # 图 4: Biological Signals
    # ==========================================
    ax = axes[1, 1]
    
    # PDS (Correlation)
    if 'val_Raw_PDS' in history_df.columns:
        ax.plot(steps, history_df['val_Raw_PDS'], color='tab:orange', linestyle='-', label='Val PDS')
    if 'train_Raw_PDS' in history_df.columns:
        ax.plot(steps, history_df['train_Raw_PDS'], color='tab:orange', linestyle='--', alpha=0.5, label='Train PDS')
    ax.axhline(y=BASELINE_PDS, color='tab:orange', linestyle=':', label='Base PDS')

    # DES (R2)
    if 'val_Raw_DES' in history_df.columns:
        ax.plot(steps, history_df['val_Raw_DES'], color='tab:blue', linestyle='-', label='Val DES')
    if 'train_Raw_DES' in history_df.columns:
        ax.plot(steps, history_df['train_Raw_DES'], color='tab:blue', linestyle='--', alpha=0.5, label='Train DES')
    ax.axhline(y=BASELINE_DES, color='tab:blue', linestyle=':', label='Base DES')

    ax.set_title("Biological Signals: Train(dashed) vs Val(solid)") # 🧬 已删除
    ax.legend(ncol=2, fontsize='small')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Training Monitor - Step {step}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/latest_score_card.png")
    plt.savefig(f"{PLOT_DIR}/step_{step:05d}.png")
    plt.close()

# ==================== 4. 主程序 ====================
if __name__ == "__main__":
    print(f"🚀 [Score train-500/val-500] 启动训练 (Baseline: DES={BASELINE_DES}, PDS={BASELINE_PDS}, MAE={BASELINE_MAE})...")
    
    # === 1. 清理旧目录 (确保是从头开始) ===
    if os.path.exists(OUTPUT_ROOT):
        try:
            shutil.rmtree(OUTPUT_ROOT)
            print(f"🧹 已清理旧目录: {OUTPUT_ROOT}")
        except Exception as e:
            print(f"⚠️ 清理目录失败: {e}")
            
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # ⚠️ 注释掉此行，因为我们已经手动生成了更好的 mixed 训练监控集
    # ensure_train_subset() 
    
    log_file = open(TRAIN_LOG_FILE, "w")
    print(f"📝 训练日志将写入: {TRAIN_LOG_FILE}")
    
    # === 2. 启动训练进程 (强制使用 GPU 2) ===
    # 这样可以防止训练进程抢占 GPU 3 的显存，留给推理进程专用
    train_env = os.environ.copy()
    train_env["CUDA_VISIBLE_DEVICES"] = "2" 

    try:
        process = subprocess.Popen(
            TRAIN_CMD, 
            stdout=log_file, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1,
            env=train_env # <--- 注入环境变量
        )
    except FileNotFoundError:
        print("❌ 找不到命令 (如 uv/state)，请检查环境。")
        sys.exit(1)

    processed_ckpts = set()
    history_df = pd.DataFrame()

    # === 3. 启动缓冲检查 (防止刚启动就挂) ===
    print("⏳ 等待训练进程启动 (5秒)...")
    time.sleep(5)
    if process.poll() is not None:
        print(f"❌ 训练进程立即退出！Exit Code: {process.returncode}")
        log_file.close()
        # 打印日志尾部帮助排错
        try:
            with open(TRAIN_LOG_FILE, 'r') as f:
                print("".join(f.readlines()[-20:]))
        except: pass
        sys.exit(1)
    else:
        print("✅ 训练进程 (GPU 2) 运行中... 监控脚本 (GPU 3) 准备就绪。")

    try:
        while True:
            if process.poll() is not None:
                print("🏁 训练结束。")
                break
            
            # 1. 获取所有 ckpt 文件
            raw_ckpts = glob.glob(f"{CHECKPOINT_DIR}/*.ckpt")
            
            # === ✨ 寻找并打印最佳模型信息 (你的新增逻辑) ===
            best_model_info = "尚未产生"
            best_step_num = -1
            import re
            
            for c in raw_ckpts:
                base = os.path.basename(c)
                if "val_loss" in base:
                    try:
                        best_model_info = base
                        m_best = re.search(r"step=(\d+)", base)
                        if m_best: best_step_num = int(m_best.group(1))
                    except: pass
            
            if best_step_num != -1:
                # 稍微美化一下输出，避免刷屏太快
                pass 
                # print(f"🌟 [当前最佳] Step: {best_step_num}") 

            # === 2. 过滤与排序 ===
            ckpts = []
            for c in raw_ckpts:
                filename = os.path.basename(c)
                
                # 🛑 过滤：跳过 val_loss 副本 和 重复命名的文件
                if "val_loss" in filename or "step=step" in filename:
                    continue
                
                m = re.search(r"step=(\d+)", filename)
                if m: 
                    step_num = int(m.group(1))
                    ckpts.append((step_num, c))
            
            ckpts.sort(key=lambda x: x[0])
            
            # === 3. 遍历执行推理 ===
            new_data = False
            for step, ckpt_path in ckpts:
                if ckpt_path in processed_ckpts: continue
                
                # 在打印 Step 时顺便显示一下当前的最佳状态
                best_info_str = f" (🌟 Best: {best_step_num})" if best_step_num != -1 else ""
                print(f"[{time.strftime('%H:%M')}] 🔎 Step {step}{best_info_str} ...")
                
                row = {'step': step, 'timestamp': time.strftime('%Y-%m-%d %H:%M')}
                
                # Val Set (GPU 1)
                # 确保 calculate_competition_scores 处理 None 的情况
                pred_val = run_inference(ckpt_path, VAL_DATA_PATH)
                if pred_val:
                    val_m = calculate_competition_scores(pred_val, VAL_DATA_PATH)
                    if val_m:
                        for k, v in val_m.items(): row[f"val_{k}"] = v
                        print(f"   🏆 Score: {val_m['Score_S']:.2f} (MAE_s:{val_m['Scaled_MAE']:.2f})")
                    if os.path.exists(pred_val): os.remove(pred_val)

                # Train Set (GPU 1)
                if os.path.exists(TRAIN_SUBSET_PATH):
                    pred_train = run_inference(ckpt_path, TRAIN_SUBSET_PATH)
                    if pred_train:
                        train_m = calculate_competition_scores(pred_train, TRAIN_SUBSET_PATH)
                        if train_m:
                            for k, v in train_m.items(): row[f"train_{k}"] = v
                        if os.path.exists(pred_train): os.remove(pred_train)
                
                # 写入 CSV
                df_row = pd.DataFrame([row])
                history_df = pd.concat([history_df, df_row], ignore_index=True)
                history_df.to_csv(METRICS_FILE, index=False)
                
                processed_ckpts.add(ckpt_path)
                new_data = True
            
            if new_data:
                save_dashboard(history_df, history_df['step'].iloc[-1])
            
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n🛑 用户停止训练。")
        process.terminate()
    finally:
        log_file.close()