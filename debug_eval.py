import os
import pandas as pd
import numpy as np
import scanpy as sc
import subprocess
import glob
from cell_eval import MetricsEvaluator, score_agg_metrics

# ================= 配置区域 (请确认路径) =================
OUTPUT_ROOT = "competition_hepg2_official"
VAL_DATA_PATH = "competition_support_set/hepg2_val_with_controls.h5ad"
DATA_ROOT = "competition_support_set"

# 尝试自动寻找一个 step=1000 的 ckpt，如果找不到请手动修改下面的 path
CKPT_SEARCH = f"{OUTPUT_ROOT}/run_3090/checkpoints/step=1000.ckpt"
found_ckpts = glob.glob(CKPT_SEARCH)
if found_ckpts:
    CHECKPOINT_PATH = found_ckpts[0]
else:
    # 如果找不到，请在这里填入绝对路径
    CHECKPOINT_PATH = "你的ckpt文件的绝对路径.ckpt"

print(f"🎯 选定的 Checkpoint: {CHECKPOINT_PATH}")

# 官方 Baseline (必须是这三列)
BASELINE_DF = pd.DataFrame([{
    'mae': 0.027,          
    'r2_mean': 0.106,      
    'pearson_mean': 0.514  
}])
# =======================================================

def step1_inference():
    print(f"\n🔎 [1/3] 开始推理...")
    out_file = "debug_pred.h5ad"
    
    # 如果已经推理过且文件存在，直接跳过以节省时间 (可选)
    # if os.path.exists(out_file):
    #     print("   -> 检测到 debug_pred.h5ad 已存在，跳过推理。")
    #     return out_file

    cmd = [
        "uv", "run", "state", "tx", "infer",
        f"--output={out_file}",
        f"--model-dir={OUTPUT_ROOT}/run_3090",
        f"--checkpoint={CHECKPOINT_PATH}",
        f"--adata={VAL_DATA_PATH}",
        "--pert-col=target_gene"
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    
    try:
        subprocess.run(cmd, check=True, env=env)
        print("✅ 推理完成，已生成 debug_pred.h5ad")
        return out_file
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return None

def step2_compute_metrics(pred_path):
    print(f"\n🔎 [2/3] 计算 Raw Metrics (MetricsEvaluator)...")
    adata_pred = sc.read_h5ad(pred_path)
    adata_true = sc.read_h5ad(VAL_DATA_PATH)
    
    # 基因对齐
    common = np.intersect1d(adata_pred.var_names, adata_true.var_names)
    adata_pred = adata_pred[:, common].copy()
    adata_true = adata_true[:, common].copy()
    
    print("   Running compute() (这可能需要几分钟)...")
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_true,
        pert_col="target_gene",
        control_pert="non-targeting",
        num_threads=16
    )
    
    _, agg_results = evaluator.compute()
    
    # 转换为 DataFrame
    if isinstance(agg_results, dict):
        raw_df = pd.DataFrame([agg_results])
    else:
        raw_df = agg_results

    return raw_df

def step3_clean_and_score(raw_df):
    print(f"\n🔎 [3/3] 数据清洗与打分 (Fix Shapes)...")

    # ==========================================
    # 🛑 [关键修复] Polars -> Pandas 转换
    # ==========================================
    # 报错是因为 raw_df 是 polars 对象，不支持 Pandas 的筛选语法
    try:
        if hasattr(raw_df, "to_pandas"):
            print("   -> 检测到 Polars DataFrame，正在转换为 Pandas...")
            raw_df = raw_df.to_pandas()
    except Exception as e:
        print(f"⚠️ 转换失败，尝试直接处理: {e}")

    # === 1. 数据清洗 (提取 mean 行) ===
    # 现在它已经是 Pandas 了，原来的代码就能正常工作了
    if 'statistic' in raw_df.columns:
        mean_row = raw_df[raw_df['statistic'] == 'mean']
        if mean_row.empty:
            print("⚠️ 警告: 找不到 statistic='mean' 的行，尝试使用第一行")
            mean_row = raw_df.iloc[[0]]
    else:
        mean_row = raw_df.iloc[[0]]
    
    # === 2. 列名映射 (Mapping) ===
    # 提取官方需要的 3 个指标
    
    # (A) MAE
    val_mae = float(mean_row['mae'].iloc[0])
    
    # (B) Pearson (PDS)
    # 优先找 'pearson_mean' (如果 cell-eval 版本更新了)，否则找 'pearson_delta'
    if 'pearson_mean' in mean_row.columns:
        val_pearson = float(mean_row['pearson_mean'].iloc[0])
    elif 'pearson_delta' in mean_row.columns:
        val_pearson = float(mean_row['pearson_delta'].iloc[0])
    else:
        print("⚠️ 警告: 找不到 Pearson 相关列，设为 0")
        val_pearson = 0.0

    # (C) R2 (DES)
    # 优先找 'r2_mean'，否则找 'r2'，都没有就设为 0 (防止报错)
    if 'r2_mean' in mean_row.columns:
        val_r2 = float(mean_row['r2_mean'].iloc[0])
    elif 'r2' in mean_row.columns:
        val_r2 = float(mean_row['r2'].iloc[0])
    else:
        print("⚠️ 警告: 结果中没有 R2 (DES)，暂时设为 0.0")
        val_r2 = 0.0
        
    # === 3. 构造干净的 User DataFrame ===
    user_clean_df = pd.DataFrame([{
        'mae': val_mae,
        'r2_mean': val_r2,
        'pearson_mean': val_pearson
    }])
    
    print("\n📊 [Cleaned Data for Scoring]")
    print(user_clean_df)
    print("-" * 30)

    # === 4. 调用官方打分 ===
    try:
        score_df = score_agg_metrics(
            results_user=user_clean_df,
            results_base=BASELINE_DF,
            output="debug_final_score.csv"
        )
        
        print("\n✅ [Success] 打分成功！结果如下：")
        if score_df is None:
            # 有些版本只写文件不返回，手动读取
            if os.path.exists("debug_final_score.csv"):
                score_df = pd.read_csv("debug_final_score.csv")
            else:
                print("❌ 未生成分数文件")
                return

        print(score_df)
        
        # 提取最终分
        final_s = 0
        if 'total_score' in score_df.columns:
            final_s = score_df['total_score'].iloc[0] * 100
        elif 'score' in score_df.columns:
            final_s = score_df['score'].iloc[0] * 100
            
        print(f"\n🏆 最终得分 (Score S): {final_s:.2f}")
        
    except Exception as e:
        print(f"❌ 打分依然失败: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    # 1. 推理
    pred_file = step1_inference()
    if pred_file:
        # 2. 计算原始指标
        raw_metrics = step2_compute_metrics(pred_file)
        # 3. 清洗并打分
        step3_clean_and_score(raw_metrics)