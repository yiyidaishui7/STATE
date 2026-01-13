import os
import time
from pathlib import Path
from datetime import datetime

# ================= 配置区域 =================
# 你想要扫描的根目录 (默认当前目录，你可以修改为绝对路径，如 '/home/user/projects')
SEARCH_DIR = "." 

# 你关心的文件后缀
EXTENSIONS = {'.ckpt', '.pth', '.pt', '.bin', '.h5'}

# 是否只显示大文件 (例如大于 100MB)
MIN_SIZE_MB = 0 
# ===========================================

def get_file_size_mb(file_path):
    """获取文件大小 (MB)"""
    return os.path.getsize(file_path) / (1024 * 1024)

def format_time(timestamp):
    """格式化时间戳"""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def scan_checkpoints(root_dir):
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"错误: 路径 '{root_dir}' 不存在。")
        return

    print(f"正在扫描 '{root_path.resolve()}' 下的 checkpoint 文件...\n")

    found_files = []

    # 递归遍历目录
    for p in root_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTENSIONS:
            size_mb = get_file_size_mb(p)
            if size_mb >= MIN_SIZE_MB:
                found_files.append({
                    "path": str(p),
                    "filename": p.name,
                    "size_mb": size_mb,
                    "mtime": p.stat().st_mtime,
                    "parent": p.parent.name
                })

    if not found_files:
        print("未找到任何 checkpoint 文件。")
        return

    # 按修改时间排序 (最近的在最后)
    found_files.sort(key=lambda x: x['mtime'])

    # === 打印报告 ===
    print(f"{'文件名':<40} | {'所在文件夹':<20} | {'大小 (MB)':<10} | {'最后修改时间':<20}")
    print("-" * 100)

    total_size = 0
    for f in found_files:
        total_size += f['size_mb']
        # 截断过长的文件名以保持对齐
        display_name = (f['filename'][:37] + '..') if len(f['filename']) > 37 else f['filename']
        display_folder = (f['parent'][:17] + '..') if len(f['parent']) > 17 else f['parent']
        
        print(f"{display_name:<40} | {display_folder:<20} | {f['size_mb']:<10.2f} | {format_time(f['mtime']):<20}")

    print("-" * 100)
    print(f"\n总结:")
    print(f"共找到文件: {len(found_files)} 个")
    print(f"总占用空间: {total_size / 1024:.2f} GB")

if __name__ == "__main__":
    scan_checkpoints(SEARCH_DIR)