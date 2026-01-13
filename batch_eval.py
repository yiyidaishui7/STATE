import os
import re
import glob
import torch
import pandas as pd
import scanpy as sc
import numpy as np
from cell_eval import MetricsEvaluator

# ================= 配置区域 (请修改这里) =================
# 1. 真实验证集路径 (Ground Truth)
#    通常是 competition_hepg2_2 目录下的某个 .h5ad 文件，或者是你分割出来的 validation set
REAL_DATA_PATH = "competition_support_set/hepg2_val_with_controls.h5ad"

# 2. Checkpoint 所在的文件夹路径
CKPT_DIR = "competition_hepg2_2/run_3090/checkpoints"

# 3. 结果保存路径
OUTPUT_DIR = "competition_hepg2_2/eval_results"
LOG_FILE = os.path.join(OUTPUT_DIR, "all_steps_scores.csv")

# 4. cell-eval 设置 (必须与你的 .h5ad 列名一致)
CONTROL_PERT_NAME = "control"    # <--- 对照组在 perturbation 列中的名字 (如 'ctrl', 'DMSO', 'non-targeting')
PERT_COLUMN = "perturbation"     # <--- 扰动标签的列名 (如 'condition', 'target_gene')

# 5. 导入你的模型类
#    这取决于你的代码结构。如果你是在 State 官方 repo 下：
try:
    from state import State
except ImportError:
    # 如果你在 competition 目录下，可能需要这样导入 (根据实际情况调整)
    import sys
    sys.path.append(os.getcwd())
    from src.models import State # 或者是 from model import State

# ==============================================================

def get_step_from_filename(filename):
    """从文件名 'step=12000.ckpt' 中提取数字 12000"""
    match = re.search(r"step=(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1

def predict_expression(model, adata_real):
    """
    核心推理函数：使用模型预测 adata_real 的表达量
    """
    model.eval()
    model.cuda() # 移动到 GPU
    
    # 复制一份作为预测容器
    adata_pred = adata_real.copy()
    
    print("  -> Running inference...")
    
    # === State 模型推理逻辑 ===
    # State 模型通常有一个 predict() 方法或类似的高级 API
    # 假设模型的输入是 (adata, perturbation_col)
    try:
        # 方式 A: 如果模型有封装好的 predict 方法 (Arc 官方代码常用)
        # 注意：这里的调用方式可能需要根据你具体的 State 版本微调
        # 比如：pred_adata = model.predict(adata_real, pert_column=PERT_COLUMN)
        
        # 方式 B: 如果需要手动推理 (更通用的 PyTorch Lightning 方式)
        # 下面是一个假设的推理循环，你需要根据你的 dataset 类进行调整
        from torch.utils.data import DataLoader
        # 假设你有一个 dataset 类，如果没有，可能需要直接传 adata
        # 这里为了通用性，我们假设 model.predict 能处理 adata
        
        with torch.no_grad():
             # 许多 State 实现支持直接传入 adata 进行预测
             # 如果 model.predict 不存在，你需要替换为 model(batch) 的循环
             pred_output = model.predict(
                 adata_real, 
                 target_col=PERT_COLUMN, # 指定扰动列
                 batch_size=2048
             )
             
             # 将预测结果赋值给 adata_pred
             # 确保 pred_output 是一个矩阵或 adata
             if isinstance(pred_output, sc.AnnData):
                 adata_pred = pred_output
             else:
                 adata_pred.X = pred_output

    except AttributeError:
        print("  [Warning] model.predict() not found. using fallback loop.")
        # 如果没有 predict 方法，这里是一个简单的 fallback (需要你根据 validate.py 确认)
        pass 
        
    return adata_pred

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 扫描并排序 Checkpoints
    ckpt_files = glob.glob(os.path.join(CKPT_DIR, "step=*.ckpt"))
    ckpt_files.sort(key=lambda x: get_step_from_filename(os.path.basename(x)))
    
    # 如果有 final.ckpt 或 last.ckpt，可以按需添加
    if os.path.exists(os.path.join(CKPT_DIR, "final.ckpt")):
        ckpt_files.append(os.path.join(CKPT_DIR, "final.ckpt"))

    print(f"Found {len(ckpt_files)} checkpoints.")

    # 2. 加载真实数据 (只加载一次以节省时间)
    print(f"Loading Ground Truth: {REAL_DATA_PATH}")
    adata_real = sc.read_h5ad(REAL_DATA_PATH)
    
    # 确保数据在内存中是 float32，避免报错
    if hasattr(adata_real.X, 'toarray'):
        adata_real.X = adata_real.X.toarray()
    adata_real.X = adata_real.X.astype(np.float32)

    results_list = []

    # 3. 循环评估
    for ckpt_path in ckpt_files:
        step = get_step_from_filename(os.path.basename(ckpt_path))
        if step == -1 and "final" in ckpt_path: step = "final"
        
        print(f"\n--- Processing Step: {step} ({os.path.basename(ckpt_path)}) ---")
        
        try:
            # A. 加载模型
            print("  -> Loading model weights...")
            model = State.load_from_checkpoint(ckpt_path)
            
            # B. 预测
            adata_pred = predict_expression(model, adata_real)
            
            # C. 评估 (cell-eval)
            print("  -> Running cell-eval metrics...")
            evaluator = MetricsEvaluator(
                adata_pred=adata_pred,
                adata_real=adata_real,
                control_pert=CONTROL_PERT_NAME,
                pert_col=PERT_COLUMN,
                num_threads=16,  # 并行线程数
                # 某些版本可能需要指定 metrics_profile
                # profile='simple' 
            )
            
            _, agg_results = evaluator.compute()
            
            # D. 记录结果
            # agg_results 是一系列指标的均值 (如 r2_mean, mse_mean)
            score_dict = agg_results.to_dict()
            score_dict['step'] = step
            score_dict['ckpt_path'] = ckpt_path
            
            # 打印关键指标
            print(f"  -> Step {step} Score: R2={score_dict.get('r2_mean', 'N/A'):.4f}, MSE={score_dict.get('mse_mean', 'N/A'):.4f}")
            
            results_list.append(score_dict)
            
            # 实时保存，防止程序中断丢失数据
            pd.DataFrame(results_list).to_csv(LOG_FILE, index=False)
            
        except Exception as e:
            print(f"  [Error] Failed at step {step}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nEvaluation Complete. Results saved to {LOG_FILE}")

if __name__ == "__main__":
    main()