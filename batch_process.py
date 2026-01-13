import os  
import subprocess  
import glob  
import pandas as pd  
from pathlib import Path  
  
def process_checkpoints_batch():  
    """批量处理所有 checkpoint"""  
      
    # 配置路径  
    ckpt_dir = Path("~/state/competition/run_1230/checkpoints").expanduser()  
    # real_data = "competition_support_set/hepg2_val_with_controls.h5ad"  
    real_data = "competition_support_set/competition_val_template.h5ad"  
    output_dir = Path("batch_results")  
      
    # 创建输出目录  
    output_dir.mkdir(exist_ok=True)  
    (output_dir / "predictions").mkdir(exist_ok=True)  
    (output_dir / "evaluations").mkdir(exist_ok=True)  
    (output_dir / "scores").mkdir(exist_ok=True)  
      
    # 生成基线  
    if not Path("baseline_eval/agg_results.csv").exists():  
        print("生成基线...")  
        subprocess.run([  
            "cell-eval", "baseline",  
            "--adata", real_data,  
            "--output-path", "baseline/baseline.h5ad",  
            "--output-de-path", "baseline/baseline_de.csv"  
        ], check=True)  
          
        subprocess.run([  
            "cell-eval", "run",  
            "-ap", "baseline/baseline.h5ad",  
            "-ar", real_data,  
            "--profile", "full",  
            "-o", "baseline_eval"  
        ], check=True)  
      
    results = []  
      
    # 查找所有 step=*.ckpt 文件  
    ckpt_files = list(ckpt_dir.glob("step=*.ckpt"))  
      
    for ckpt_file in ckpt_files:  
        # 提取 step 数值  
        ckpt_name = ckpt_file.name  
        if "last.ckpt" in ckpt_name or "val_loss" in ckpt_name:  
            print(f"跳过: {ckpt_name}")  
            continue  
              
        # 提取数字  
        import re  
        match = re.search(r'step=(\d+)\.ckpt', ckpt_name)  
        if not match:  
            print(f"无法解析 step 数值: {ckpt_name}")  
            continue  
              
        step = match.group(1)  
        print(f"处理 step={step}.ckpt")  
          
        try:  
            # 1. 生成预测  
            pred_path = output_dir / "predictions" / f"pred_{step}.h5ad"  
            subprocess.run([  
                "uv", "run", "state", "tx", "infer",  
                "--output", str(pred_path),  
                "--model-dir", "competition/run_1230",  
                "--checkpoint", str(ckpt_file),  
                "--adata", real_data,  
                "--pert-col", "target_gene"  
            ], check=True)  
              
            # 2. 运行评估  
            eval_outdir = output_dir / "evaluations" / f"eval_{step}"  
            subprocess.run([  
                "cell-eval", "run",  
                "-ap", str(pred_path),  
                "-ar", real_data,  
                "--profile", "full",  
                "-o", str(eval_outdir)  
            ], check=True)  
              
            # 3. 计算评分  
            score_path = output_dir / "scores" / f"score_{step}.csv"  
            subprocess.run([  
                "cell-eval", "score",  
                "--user-input", str(eval_outdir / "agg_results.csv"),  
                "--base-input", "baseline_eval/agg_results.csv",  
                "--output", str(score_path)  
            ], check=True)  
              
            # 4. 记录结果  
            score_df = pd.read_csv(score_path)  
            avg_score = score_df[score_df['metric'] == 'avg_score']['from_baseline'].iloc[0]  
            results.append({  
                "step": int(step),  
                "avg_score": avg_score,  
                "ckpt_file": str(ckpt_file)  
            })  
              
        except Exception as e:  
            print(f"处理 step={step} 时出错: {e}")  
            results.append({"step": int(step), "avg_score": None, "error": str(e)})  
      
    # 保存汇总结果  
    summary_df = pd.DataFrame(results).sort_values("step")  
    summary_df.to_csv(output_dir / "final_summary.csv", index=False)  
      
    print("\n处理完成！结果汇总:")  
    print(summary_df)  
      
    return summary_df  
  
if __name__ == "__main__":  
    results = process_checkpoints_batch()