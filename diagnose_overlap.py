#!/usr/bin/env python3
"""
诊断脚本：分析为什么HepG2有部分数据在训练集
"""

import h5py
import sys
from pathlib import Path

def analyze_perturbations(data_dir):
    """分析各个cell type的perturbations"""
    
    print("=" * 60)
    print("Perturbation重叠分析")
    print("=" * 60)
    print()
    
    cell_types = ['k562', 'jurkat', 'rpe1', 'hepg2']
    perturbations = {}
    
    # 读取每个cell type的perturbations
    for ct in cell_types:
        filepath = Path(data_dir) / f"{ct}.h5"
        if not filepath.exists():
            print(f"⚠️  文件不存在: {filepath}")
            continue
        
        with h5py.File(filepath, 'r') as f:
            # 尝试不同的可能路径
            pert_paths = [
                'obs/target_gene/categories',
                'obs/perturbation/categories',
                'obs/gene/categories',
            ]
            
            for path in pert_paths:
                if path in f:
                    perts = f[path][:]
                    # 转换为字符串
                    if hasattr(perts[0], 'decode'):
                        perts = [p.decode() for p in perts]
                    else:
                        perts = [str(p) for p in perts]
                    
                    perturbations[ct] = set(perts)
                    print(f"{ct}: {len(perts)} 个perturbations")
                    print(f"  前5个: {list(perts)[:5]}")
                    break
            else:
                print(f"❌ {ct}: 未找到perturbation信息")
    
    if len(perturbations) < 4:
        print("\n❌ 无法读取所有cell types的perturbation信息")
        return
    
    # 分析重叠
    print("\n" + "=" * 60)
    print("重叠分析")
    print("=" * 60)
    
    # 训练集的perturbations
    train_perts = set()
    for ct in ['k562', 'jurkat', 'rpe1']:
        if ct in perturbations:
            train_perts.update(perturbations[ct])
    
    print(f"\n训练集（K562+Jurkat+RPE1）总共: {len(train_perts)} 个perturbations")
    
    # HepG2的perturbations
    if 'hepg2' in perturbations:
        hepg2_perts = perturbations['hepg2']
        print(f"HepG2总共: {len(hepg2_perts)} 个perturbations")
        
        # 计算重叠
        overlap = train_perts & hepg2_perts
        unique = hepg2_perts - train_perts
        
        print(f"\n重叠的perturbations: {len(overlap)} 个")
        print(f"  这些perturbations在训练集和HepG2中都出现")
        if len(overlap) > 0:
            print(f"  示例: {list(overlap)[:10]}")
        
        print(f"\nHepG2独有的perturbations: {len(unique)} 个")
        print(f"  这些只在HepG2中出现（真正的zero-shot）")
        if len(unique) > 0:
            print(f"  示例: {list(unique)[:10]}")
        
        print("\n" + "=" * 60)
        print("结论")
        print("=" * 60)
        print()
        
        overlap_ratio = len(overlap) / len(hepg2_perts) * 100
        unique_ratio = len(unique) / len(hepg2_perts) * 100
        
        print(f"HepG2中：")
        print(f"  - {len(overlap)} 个 ({overlap_ratio:.1f}%) perturbations与训练集重叠")
        print(f"  - {len(unique)} 个 ({unique_ratio:.1f}%) perturbations是独有的")
        print()
        
        print("这可能解释了为什么会有4976个样本在训练集：")
        print(f"  - 如果这4976个样本对应重叠的perturbations")
        print(f"  - 那么9386个样本对应独有的perturbations（真正zero-shot）")
        print()
        
        if overlap_ratio > 30:
            print("⚠️  重叠率较高！建议：")
            print("  1. 接受这个配置，9386个样本仍然是zero-shot")
            print("  2. 在论文中说明这一点")
            print("  3. 或者预处理数据，移除重叠的perturbations")
        else:
            print("✓ 重叠率可接受，可以继续使用这个配置")

if __name__ == "__main__":
    data_dir = "~/state/competition_support_set/"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    # 展开~
    data_dir = Path(data_dir).expanduser()
    
    analyze_perturbations(data_dir)