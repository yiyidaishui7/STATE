#!/usr/bin/env python3
"""
检查Virtual Cell Challenge的.h5数据文件
用于验证数据结构和生成baseline配置建议
"""

import h5py
import os
import sys
from pathlib import Path

def check_h5_files(data_dir):
    """
    检查目录中的所有.h5文件
    """
    print("="*60)
    print("Virtual Cell Challenge 数据检查工具")
    print("="*60)
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"❌ 错误：目录不存在 {data_dir}")
        return False
    
    print(f"\n✅ 数据目录: {data_dir}")
    
    # 查找所有.h5文件
    h5_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
    
    if not h5_files:
        print("❌ 错误：没有找到.h5文件")
        return False
    
    print(f"\n找到 {len(h5_files)} 个.h5文件：")
    for f in h5_files:
        print(f"   - {f}")
    
    # 检查是否有starter.toml
    starter_path = os.path.join(data_dir, "starter.toml")
    if os.path.exists(starter_path):
        print("\n✅ 找到 starter.toml（官方配置示例）")
        print("   建议先查看: cat", starter_path)
    else:
        print("\n⚠️  未找到 starter.toml")
    
    # 识别cell type文件
    print("\n" + "="*60)
    print("Cell Type文件识别")
    print("="*60)
    
    cell_type_files = []
    other_files = []
    
    for filename in h5_files:
        # 常见的cell type名称
        cell_types = ['k562', 'jurkat', 'rpe1', 'hepg2', 'h1', 'a549', 'mcf7']
        
        name = filename.lower().replace('.h5', '')
        is_cell_type = any(ct in name for ct in cell_types)
        
        if is_cell_type:
            cell_type_files.append(filename)
        else:
            other_files.append(filename)
    
    print(f"\nCell type数据文件 ({len(cell_type_files)}个):")
    for f in cell_type_files:
        print(f"   ✓ {f}")
    
    if other_files:
        print(f"\n其他.h5文件 ({len(other_files)}个):")
        for f in other_files:
            print(f"   - {f}")
    
    # 详细检查每个cell type文件
    print("\n" + "="*60)
    print("数据文件详细信息")
    print("="*60)
    
    total_samples = {}
    
    for filename in cell_type_files:
        filepath = os.path.join(data_dir, filename)
        cell_type = filename.replace('.h5', '')
        
        print(f"\n{filename}:")
        try:
            with h5py.File(filepath, 'r') as f:
                print(f"   数据集keys: {list(f.keys())}")
                
                # 尝试获取样本数
                for key in f.keys():
                    dataset = f[key]
                    if hasattr(dataset, 'shape'):
                        shape = dataset.shape
                        print(f"   {key}: shape={shape}, dtype={dataset.dtype}")
                        
                        # 估计样本数（通常是第一个维度）
                        if len(shape) > 0:
                            total_samples[cell_type] = shape[0]
                
        except Exception as e:
            print(f"   ❌ 读取失败: {e}")
    
    # 生成配置建议
    print("\n" + "="*60)
    print("Baseline配置建议")
    print("="*60)
    
    if len(cell_type_files) < 3:
        print("\n⚠️  警告：找到的cell type少于3个，可能不足以进行domain generalization")
    
    print("\n推荐的数据划分：")
    
    # 按样本数排序
    if total_samples:
        sorted_cts = sorted(total_samples.items(), key=lambda x: x[1], reverse=True)
        
        print("\n1. 训练集（Source Domains）- 前3个:")
        for i, (ct, count) in enumerate(sorted_cts[:3]):
            print(f"   {i+1}. {ct}: {count:,} 样本")
        
        if len(sorted_cts) > 3:
            print("\n2. 验证集 - 第4个:")
            ct, count = sorted_cts[3]
            print(f"   - {ct}: {count:,} 样本")
        
        if len(sorted_cts) > 4:
            print("\n3. 测试集 - 第5个:")
            ct, count = sorted_cts[4]
            print(f"   - {ct}: {count:,} 样本")
    
    # 生成配置文件内容
    print("\n" + "="*60)
    print("建议的配置文件内容")
    print("="*60)
    
    print('\n[datasets]')
    print(f'competition = "{data_dir}"')
    
    print('\n[training]')
    print('competition = "train"')
    
    print('\n[zeroshot]')
    if total_samples:
        sorted_cts = sorted(total_samples.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_cts) > 3:
            ct, _ = sorted_cts[3]
            print(f'"competition.{ct}" = "val"')
        if len(sorted_cts) > 4:
            ct, _ = sorted_cts[4]
            print(f'"competition.{ct}" = "test"')
    
    print('\n[fewshot]')
    print('# 留空')
    
    print("\n" + "="*60)
    print("下一步操作")
    print("="*60)
    
    print("\n1. 查看官方配置（如果存在）：")
    print(f"   cat {starter_path}")
    
    print("\n2. 创建你的baseline配置：")
    print("   mkdir -p /state/configs")
    print(f"   cp {starter_path} /state/configs/baseline_config.toml")
    print("   nano /state/configs/baseline_config.toml")
    
    print("\n3. 根据上面的建议修改 [zeroshot] 部分")
    
    print("\n4. 运行训练命令（参考 starter.toml 中的参数）")
    
    print("\n" + "="*60)
    print("检查完成！")
    print("="*60)
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_h5_files.py <data_directory>")
        print("\n示例:")
        print("  python check_h5_files.py /state/competition_support_set/")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    success = check_h5_files(data_dir)
    
    if not success:
        sys.exit(1)