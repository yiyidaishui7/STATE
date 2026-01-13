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

# ==================== 0. 官方库导入 ====================
try:
    # 1. 导入计算类 (根据文档: MetricsEvaluator)
    from cell_eval import MetricsEvaluator
    # 2. 导入打分函数
    from cell_eval import score_agg_metrics
    print("✅ Successfully imported cell_eval (MetricsEvaluator & score_agg_metrics)")
except ImportError:
    print("❌ Critical: 'cell_eval' library not found. Please install it.")
    sys.exit(1)

# ==================== 1. 基础配置 ====================

DATA_ROOT = "competition_support_set" 
OUTPUT_ROOT = "competition_hepg2_official"
METRICS_FILE = f"{OUTPUT_ROOT}/final_score_log.csv"
PLOT_DIR = f"{OUTPUT_ROOT}/plots"
TRAIN_LOG_FILE = f"{OUTPUT_ROOT}/training.log"

# === 官方 Baseline 数据 (用于 step 3) ===
# 必须构造成 DataFrame 格式，以便 score_agg_metrics 读取
BASELINE_DF = pd.DataFrame([{
    'mae': 0.027,          
    'r2_mean': 0.106,      
    'pearson_mean': 0.514  
}])

# ==================== 2. 环境准备 ====================

if not os.path.exists(DATA_ROOT):
    raise FileNotFoundError(f"❌ 未找到数据目录: {DATA_ROOT}")

# 生成配置文件 (保持不变)
CONFIG_PATH = os.path.join(DATA_ROOT, "server_config_4.toml")
with open(CONFIG_PATH, "w") as f:
    f.write(f"""
[datasets]
replogle_h1 = "competition_support_set/{{rpe1,jurkat,k562,k562_gwps,hepg2}}.h5"
[training]
replogle_h1 = "train"
[zeroshot]
"replogle_h1.hepg2" = "val"
""")

# 训练命令 (保持不变)
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
    "training.ckpt_every_n_steps=1000",
    "training.val_freq=1000",
    "model=state_sm",
    "+wandb.mode=offline",
    f"output_dir={OUTPUT_ROOT}",
    "name=run_3090"
]
CHECKPOINT_DIR = f"{OUTPUT_ROOT}/run_3090/checkpoints"
VAL_DATA_PATH = f"{DATA_ROOT}/hepg2_val_with_controls.h5ad"

# ==================== 3. 核心计算流程 (严格遵循文档) ====================

def calculate_official_score(pred_path, true_path):
    """
    [Step 2 & 3] Eval -> Score
    """
    try:
        # --- 数据加载 ---
        adata_pred = sc.read_h5ad(pred_path)
        adata_true = sc.read_h5ad(true_path)
        
        # 简单对齐基因 (防止 MetricsEvaluator 报错)
        common = np.intersect1d(adata_pred.var_names, adata_true.var_names)
        adata_pred = adata_pred[:, common].copy()
        adata_true = adata_true[:, common].copy()

        # ==========================================
        # Step 2: Run Evaluation (计算 Raw Metrics)
        # ==========================================
        print("   Running cell-eval MetricsEvaluator...")
        
        evaluator = MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_true,
            pert_col="target_gene",      # 根据你的数据列名修改
            control_pert="non-targeting", # 根据你的 Control 修改
            num_threads=16               # 根据 CPU 核数调整
        )
        
        # compute() 返回 (per_pert_results, agg_results)
        # agg_results 通常是一个 DataFrame (一行) 或 字典
        _, agg_results = evaluator.compute()
        
        # 确保 agg_results 是 DataFrame (为了下一步兼容)
        if isinstance(agg_results, dict):
            user_df = pd.DataFrame([agg_results])
        else:
            user_df = agg_results # 假设已经是 DF

        # ==========================================
        # Step 3: Score (归一化)
        # ==========================================
        print("   Running cell-eval score_agg_metrics...")
        
        # 临时文件路径 (score_agg_metrics 有时倾向于读文件)
        temp_score_csv = f"{OUTPUT_ROOT}/temp_final_score.csv"
        
        # 调用官方打分函数
        # 它会自动做 (User - Base) / (1 - Base) 等计算
        score_df = score_agg_metrics(
            results_user=user_df,
            results_base=BASELINE_DF,
            output=temp_score_csv
        )
        
        # 如果函数返回 None (旧版本特性)，则从文件读取
        if score_df is None or (isinstance(score_df, pd.DataFrame) and score_df.empty):
            if os.path.exists(temp_score_csv):
                score_df = pd.read_csv(temp_score_csv)
            else:
                return None

        # --- 提取结果用于绘图 ---
        # 注意：不同版本生成的列名可能包含 'score', 'total_score'
        res = {}
        
        # 1. 最终得分
        if 'total_score' in score_df.columns:
            res['Score_S'] = score_df['total_score'].iloc[0] * 100
        elif 'score' in score_df.columns:
            res['Score_S'] = score_df['score'].iloc[0] * 100
        else:
            res['Score_S'] = 0.0
            
        # 2. 原始指标 (来自 Step 2 的结果)
        # cell-eval 可能会用 'mae', 'mean_mae' 等不同名字
        def get_val(df, keys):
            for k in keys: 
                if k in df.columns: return df[k].iloc[0]
            return 0.0
            
        res['Raw_MAE'] = get_val(user_df, ['mae', 'mean_absolute_error'])
        res['Raw_DES'] = get_val(user_df, ['r2_mean', 'r2'])
        res['Raw_PDS'] = get_val(user_df, ['pearson_mean', 'pearson'])
        
        # 3. Scaled 指标 (来自 Step 3 的结果)
        res['Scaled_MAE'] = get_val(score_df, ['mae_scaled', 'scaled_mae'])
        res['Scaled_DES'] = get_val(score_df, ['r2_mean_scaled', 'r2_scaled'])
        res['Scaled_PDS'] = get_val(score_df, ['pearson_mean_scaled', 'pearson_scaled'])

        return res

    except Exception as e:
        print(f"❌ Eval Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_inference(ckpt, input_adata):
    """
    [Step 1] Inference: ckpt -> h5ad
    """
    out_file = f"{OUTPUT_ROOT}/temp_pred_{os.path.basename(ckpt)}.h5ad"
    cmd = [
        "uv", "run", "state", "tx", "infer",
        f"--output={out_file}",
        f"--model-dir={OUTPUT_ROOT}/run_3090",
        f"--checkpoint={ckpt}",
        f"--adata={input_adata}",
        "--pert-col=target_gene"
    ]
    
    # 指定 GPU 1 推理
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1" 
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, env=env)
        return out_file
    except subprocess.CalledProcessError as e:
        print(f"❌ 推理失败: {e}")
        return None

# ==================== 4. 绘图与主循环 (保持逻辑不变) ====================

def save_dashboard(history_df, step):
    if history_df.empty: return
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.switch_backend('Agg') 
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    steps = history_df['step']
    
    # Plot 1: Overall Score
    ax = axes[0, 0]
    if 'val_Score_S' in history_df.columns:
        ax.plot(steps, history_df['val_Score_S'], 'r-o', lw=2, label='Val Score')
        ax.text(steps.iloc[-1], history_df['val_Score_S'].max(), f" Max: {history_df['val_Score_S'].max():.2f}", color='red')
    ax.set_title("Official Score S")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Scaled Components
    ax = axes[0, 1]
    if 'val_Scaled_DES' in history_df.columns:
        ax.plot(steps, history_df['val_Scaled_DES'], label='DES (Scaled)')
        ax.plot(steps, history_df['val_Scaled_PDS'], label='PDS (Scaled)')
        ax.plot(steps, history_df['val_Scaled_MAE'], label='MAE (Scaled)')
    ax.set_title("Scaled Components (Normalized)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Raw MAE
    ax = axes[1, 0]
    if 'val_Raw_MAE' in history_df.columns:
        ax.plot(steps, history_df['val_Raw_MAE'], 'r-o', label='Val MAE')
    ax.axhline(y=0.027, color='k', linestyle='--', label='Baseline')
    ax.set_title("Raw MAE")
    ax.legend()
    
    # Plot 4: Raw Bio Signals
    ax = axes[1, 1]
    if 'val_Raw_PDS' in history_df.columns:
        ax.plot(steps, history_df['val_Raw_PDS'], color='orange', label='PDS (Pearson)')
    if 'val_Raw_DES' in history_df.columns:
        ax.plot(steps, history_df['val_Raw_DES'], color='blue', label='DES (R2)')
    ax.set_title("Raw Biological Signals")
    ax.legend()

    plt.suptitle(f"Training Monitor - Step {step}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/step_{step:05d}.png")
    plt.savefig(f"{PLOT_DIR}/latest.png")
    plt.close()

if __name__ == "__main__":
    print("🚀 启动 Official cell-eval 监控...")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 启动训练
    log_f = open(TRAIN_LOG_FILE, "w")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    proc = subprocess.Popen(TRAIN_CMD, stdout=log_f, stderr=subprocess.STDOUT, env=env, text=True)
    
    print("⏳ 等待 5秒...")
    time.sleep(5)
    if proc.poll() is not None:
        print("❌ 训练进程挂了")
        sys.exit(1)
        
    processed = set()
    history = pd.DataFrame()
    
    try:
        while True:
            if proc.poll() is not None: break
            
            # 扫描 checkpoints
            ckpts = []
            import re
            for c in glob.glob(f"{CHECKPOINT_DIR}/*.ckpt"):
                if "val_loss" in c: continue
                m = re.search(r"step=(\d+)", c)
                if m: ckpts.append((int(m.group(1)), c))
            ckpts.sort(key=lambda x: x[0])
            
            new_data = False
            for step, path in ckpts:
                if path in processed: continue
                print(f"[{time.strftime('%H:%M')}] 🔎 Step {step} Eval...")
                
                row = {'step': step, 'timestamp': time.strftime('%H:%M')}
                
                # 1. Inference (ckpt -> pred.h5ad)
                pred_file = run_inference(path, VAL_DATA_PATH)
                
                if pred_file:
                    # 2 & 3. Eval & Score (pred.h5ad + true.h5ad -> Score)
                    scores = calculate_official_score(pred_file, VAL_DATA_PATH)
                    if scores:
                        for k, v in scores.items(): row[f"val_{k}"] = v
                        print(f"   🏆 Score: {scores['Score_S']:.2f}")
                    
                    os.remove(pred_file) # 清理
                
                history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
                history.to_csv(METRICS_FILE, index=False)
                processed.add(path)
                new_data = True
                
            if new_data:
                save_dashboard(history, history['step'].iloc[-1])
            
            time.sleep(60)
            
    except KeyboardInterrupt:
        proc.terminate()
        log_f.close()