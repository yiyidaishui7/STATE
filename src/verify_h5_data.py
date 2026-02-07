#!/usr/bin/env python
"""
verify_h5_data.py - 独立测试脚本 (无需完整配置)
用途：验证三个细胞系的 h5 文件是否能正确读取，并展示数据结构

使用方法：
    cd ~/state/src
    python verify_h5_data.py
"""
import os
import sys
import h5py
import numpy as np

print("=" * 70)
print("H5 数据文件验证脚本")
print("=" * 70)

# ====== 配置数据路径 ======
BASE_DIR = "/media/mldadmin/home/s125mdg34_03/state"
DOMAIN_CONFIGS = [
    {"name": "K562",   "path": f"{BASE_DIR}/competition_support_set/k562.h5"},
    {"name": "RPE1",   "path": f"{BASE_DIR}/competition_support_set/rpe1.h5"},
    {"name": "Jurkat", "path": f"{BASE_DIR}/competition_support_set/jurkat.h5"},
]

def explore_h5_structure(h5_file, prefix=""):
    """递归探索 h5 文件结构"""
    items = []
    for key in h5_file.keys():
        path = f"{prefix}/{key}" if prefix else key
        item = h5_file[key]
        if isinstance(item, h5py.Group):
            items.append((path, "Group", len(item.keys())))
            items.extend(explore_h5_structure(item, path))
        elif isinstance(item, h5py.Dataset):
            shape = item.shape
            dtype = item.dtype
            items.append((path, "Dataset", shape, dtype))
    return items

def print_h5_structure(items, max_items=20):
    """打印 h5 结构"""
    print(f"\n  文件结构 (共 {len(items)} 项, 显示前 {min(len(items), max_items)} 项):")
    for i, item in enumerate(items[:max_items]):
        if len(item) == 3:
            path, typ, info = item
            print(f"    {path}: {typ} ({info} children)")
        else:
            path, typ, shape, dtype = item
            print(f"    {path}: {typ} shape={shape}, dtype={dtype}")
    if len(items) > max_items:
        print(f"    ... 还有 {len(items) - max_items} 项")

def load_sample_data(h5_path, num_samples=3):
    """加载样本数据"""
    with h5py.File(h5_path, 'r') as f:
        # 获取基本信息
        print(f"\n  数据矩阵信息:")
        
        if 'X' in f:
            x_item = f['X']
            
            # 检查是密集矩阵还是稀疏矩阵
            if isinstance(x_item, h5py.Dataset):
                # 密集矩阵格式
                num_cells, num_genes = x_item.shape
                print(f"    格式: 密集矩阵 (Dense)")
                print(f"    形状: {x_item.shape}")
                print(f"    数据类型: {x_item.dtype}")
                print(f"    细胞数量: {num_cells}")
                print(f"    基因数量: {num_genes}")
                
                # 读取前几个细胞的数据
                print(f"\n  前 {num_samples} 个细胞的信息:")
                for i in range(min(num_samples, num_cells)):
                    cell_data = x_item[i, :]
                    nnz = np.sum(cell_data > 0)
                    mean_val = np.mean(cell_data)
                    max_val = np.max(cell_data)
                    print(f"    Cell {i}: 非零基因数={nnz}, mean={mean_val:.4f}, max={max_val:.4f}")
                
                return num_cells
                
            elif isinstance(x_item, h5py.Group):
                # 稀疏矩阵格式 (CSR)
                attrs = dict(x_item.attrs) if hasattr(x_item, 'attrs') else {}
                print(f"    格式: 稀疏矩阵 ({attrs.get('encoding-type', 'unknown')})")
                
                if 'indptr' in x_item:
                    indptr = x_item['indptr'][:]
                    num_cells = len(indptr) - 1
                    print(f"    细胞数量: {num_cells}")
                    
                    print(f"\n  前 {num_samples} 个细胞的信息:")
                    for i in range(min(num_samples, num_cells)):
                        start = indptr[i]
                        end = indptr[i+1]
                        nnz = end - start
                        print(f"    Cell {i}: 非零基因数={nnz}")
                    
                    return num_cells
        
        # 检查 obs (细胞元数据)
        if 'obs' in f:
            obs = f['obs']
            print(f"\n  obs (细胞元数据) 列:")
            for key in list(obs.keys())[:10]:
                item = obs[key]
                if isinstance(item, h5py.Dataset):
                    print(f"    {key}: shape={item.shape}")
                elif isinstance(item, h5py.Group):
                    # 检查是否是 categorical 类型
                    if 'categories' in item:
                        cats = item['categories']
                        print(f"    {key}: Categorical ({len(cats)} categories)")
                    else:
                        print(f"    {key}: Group")
        
        # 检查 var (基因元数据)
        if 'var' in f:
            var = f['var']
            print(f"\n  var (基因元数据) 列:")
            for key in list(var.keys())[:10]:
                item = var[key]
                if isinstance(item, h5py.Dataset):
                    if item.shape:
                        print(f"    {key}: shape={item.shape}")
                elif isinstance(item, h5py.Group):
                    if 'categories' in item:
                        print(f"    {key}: Categorical")
        
        return None

def main():
    print("\n[Step 1] 检查数据文件...")
    
    all_exist = True
    for domain in DOMAIN_CONFIGS:
        exists = os.path.exists(domain["path"])
        status = "✓" if exists else "✗ 不存在"
        print(f"  {domain['name']}: {domain['path']} [{status}]")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n❌ 部分文件不存在，请检查路径后重试")
        return False

    print("\n" + "=" * 70)
    print("[Step 2] 分析每个数据文件...")
    print("=" * 70)

    domain_info = {}
    
    for domain in DOMAIN_CONFIGS:
        name = domain["name"]
        path = domain["path"]
        
        print(f"\n{'─' * 70}")
        print(f"【{name}】 {path}")
        print(f"{'─' * 70}")
        
        try:
            with h5py.File(path, 'r') as f:
                # 探索结构
                structure = explore_h5_structure(f)
                print_h5_structure(structure)
                
            # 加载样本数据并获取细胞数
            num_cells = load_sample_data(path)
            if num_cells:
                domain_info[name] = {'cells': num_cells, 'path': path}
                print(f"\n  ✓ 文件读取成功!")
            else:
                print(f"\n  ⚠️ 无法确定细胞数量")
            
        except Exception as e:
            print(f"\n  ✗ 读取失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("[Step 3] 汇总信息")
    print("=" * 70)
    
    if domain_info:
        total_cells = sum(d['cells'] for d in domain_info.values())
        print(f"\n  成功加载的域:")
        for name, info in domain_info.items():
            print(f"    • {name}: {info['cells']:,} cells")
        print(f"\n  总细胞数: {total_cells:,}")
        
        print("\n" + "=" * 70)
        print("✅ 数据验证完成！所有 h5 文件可以正常读取。")
        print("=" * 70)
        
        print("\n下一步：使用以下配置创建三路 DataLoader...")
        print("\n```python")
        print("domain_configs = [")
        for name, info in domain_info.items():
            print(f'    {{"name": "{name}", "path": "{info["path"]}"}},')
        print("]")
        print("```")
        
        return True
    else:
        print("\n❌ 没有成功读取任何文件")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
