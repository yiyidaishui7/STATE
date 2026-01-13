import os  
import subprocess  
from pathlib import Path  
import pandas as pd  
from cell_eval import MetricsEvaluator, score_agg_metrics  

def generate_prediction_from_ckpt(ckpt_path, output_path):  
    """  
    使用checkpoint生成预测AnnData  
      
    Args:  
        ckpt_path: checkpoint文件路径  
        output_path: 输出AnnData路径  
    """  
    # 这里需要根据你的模型实现  
    # 示例：  
    import torch  
    import anndata as ad  
      
    # 加载模型  
    model = load_your_model(ckpt_path)  
      
    # 生成预测  
    predictions = model.predict(test_data)  
      
    # 创建AnnData  
    adata = ad.AnnData(  
        X=predictions.expression,  
        obs=predictions.metadata  
    )  
      
    # 保存  
    adata.write_h5ad(output_path)
    
def process_checkpoints(ckpt_dir, real_data, output_dir):  
    """批量处理所有checkpoint"""  
      
    # 创建目录  
    Path(f"{output_dir}/predictions").mkdir(parents=True, exist_ok=True)  
    Path(f"{output_dir}/evaluations").mkdir(parents=True, exist_ok=True)  
    Path(f"{output_dir}/scores").mkdir(parents=True, exist_ok=True)  
      
    # 生成基线（只需要一次）  
    if not os.path.exists("baseline_eval/agg_results.csv"):  
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
      
    # 处理每个checkpoint  
    for step in range(1000, 40001, 1000):  
        print(f"处理 step={step}.ckpt")  
          
        try:  
            # 1. 生成预测（你需要实现这个函数）  
            pred_path = f"{output_dir}/predictions/pred_{step}.h5ad"  
            generate_prediction_from_ckpt(  
                f"{ckpt_dir}/step={step}.ckpt",   
                pred_path  
            )  
              
            # 2. 评估模型  
            eval_outdir = f"{output_dir}/evaluations/eval_{step}"  
            evaluator = MetricsEvaluator(  
                adata_pred=pred_path,  
                adata_real=real_data,  
                control_pert="non-targeting",  
                pert_col="target",  
                num_threads=64,  
                outdir=eval_outdir  
            )  
            _, agg_results = evaluator.compute(profile="full")  
              
            # 3. 计算评分  
            score_result = score_agg_metrics(  
                results_user=agg_results,  
                results_base="baseline_eval/agg_results.csv",  
                output=f"{output_dir}/scores/score_{step}.csv"  
            )  
              
            # 记录结果  
            avg_score = score_result.filter(  
                pl.col("metric") == "avg_score"  
            )["from_baseline"][0]  
              
            results.append({  
                "step": step,  
                "avg_score": avg_score,  
                "score_file": f"{output_dir}/scores/score_{step}.csv"  
            })  
              
        except Exception as e:  
            print(f"处理 step={step} 时出错: {e}")  
            results.append({"step": step, "avg_score": None, "error": str(e)})  
      
    # 保存汇总结果  
    summary_df = pd.DataFrame(results)  
    summary_df.to_csv(f"{output_dir}/final_summary.csv", index=False)  
    return summary_df  
  
# 使用示例  
if __name__ == "__main__":  
    results = process_checkpoints(  
        ckpt_dir="path/to/checkpoints",  
        real_data="competition_support_set/hepg2_val_with_controls.h5ad",  
        output_dir="batch_results"  
    )  
    print("处理完成！结果汇总:")  
    print(results)