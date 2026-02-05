import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler
from collections import defaultdict
from typing import Optional, List, Dict, Iterator

class DomainBalancedSampler(Sampler):
    """
    DomainBalancedSampler (结构化域平衡采样器)
    
    专为 MMD + Center Loss 设计。
    每个 Batch 的结构如下：
    [ RPE1_Control (N/k) | HepG2_Control (N/k) | ... | Mixed_Perturbations (Batch - N_ctrl) ]
    
    其中：
    - 确保每个 Batch 包含所有域的 Control 样本（用于 MMD 对齐）。
    - Perturbations 随机混合，决定 Epoch 的长度。
    - Controls 无限循环采样。
    """

    def __init__(
        self, 
        metadata: pd.DataFrame, 
        batch_size: int, 
        n_controls: int, 
        cell_line_col: str = 'cell_type', 
        pert_col: str = 'perturbation', 
        control_label: str = 'control',
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """
        Args:
            metadata (pd.DataFrame): 包含所有样本元数据的 DataFrame，索引须与 Dataset 对应。
            batch_size (int): 批次大小。
            n_controls (int): 每个 Batch 中保留给 Control 样本的固定槽位数。
            cell_line_col (str): 细胞系列名。
            pert_col (str): 扰动列名。
            control_label (str): 对照组的标签文本。
            shuffle (bool): 是否打乱扰动样本。
            seed (int): 随机种子。
        """
        if n_controls >= batch_size:
            raise ValueError(f"n_controls ({n_controls}) must be less than batch_size ({batch_size}).")
        
        self.metadata = metadata
        self.batch_size = batch_size
        self.n_controls = n_controls
        self.n_perturbed = batch_size - n_controls
        self.shuffle = shuffle
        self.seed = seed
        
        # 1. 解析数据结构
        # 找出所有 Cell Lines
        self.cell_lines = sorted(metadata[cell_line_col].unique().tolist())
        self.n_domains = len(self.cell_lines)
        
        if self.n_domains == 0:
            raise ValueError("No cell lines found in metadata.")
            
        # 2. 构建索引池
        self.ctrl_indices_map = defaultdict(list)
        self.pert_indices = []
        
        # 矢量化分离索引 (比循环快)
        is_ctrl_mask = (metadata[pert_col] == control_label)
        
        # 获取 Perturbed 全局索引
        self.pert_indices = np.where(~is_ctrl_mask)[0].tolist()
        
        # 获取各 Cell Line 的 Control 全局索引
        for cl in self.cell_lines:
            cl_mask = (metadata[cell_line_col] == cl)
            # 既是该细胞系 又是 Control
            idxs = np.where(cl_mask & is_ctrl_mask)[0].tolist()
            if len(idxs) == 0:
                print(f"[Warning] Cell line '{cl}' has NO control samples! MMD loss will fail for this domain.")
            self.ctrl_indices_map[cl] = idxs

        # 3. 计算 Epoch 长度
        # 以“遍历完一次所有扰动样本”为一个 Epoch
        self.n_batches = math.ceil(len(self.pert_indices) / self.n_perturbed)
        
        # 4. 计算每个 Batch 分给每个域的 Control 数量
        # 尽量均匀分配，多余的分配给前几个域
        base_count = self.n_controls // self.n_domains
        remainder = self.n_controls % self.n_domains
        self.ctrl_counts_per_domain = [
            base_count + (1 if i < remainder else 0) 
            for i in range(self.n_domains)
        ]
        
        print(f"[Sampler] Initialized for {self.n_domains} domains: {self.cell_lines}")
        print(f"[Sampler] Epoch strategy: {len(self.pert_indices)} perturbations -> {self.n_batches} batches.")
        print(f"[Sampler] Batch structure: {self.n_controls} Controls ({self.ctrl_counts_per_domain}) + {self.n_perturbed} Perturbations")

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
            np.random.seed(self.seed) # 用于 numpy shuffle
            
        # 1. 准备 Perturbed 队列
        pert_indices = np.array(self.pert_indices)
        if self.shuffle:
            # 使用 torch generator 保证复现性，或者直接用 numpy
            # 这里用 numpy shuffle 比较方便操作 array
            np.random.shuffle(pert_indices)
            
        # 2. 准备 Controls 无限迭代器
        # 为每个域创建一个独立的打乱迭代器
        ctrl_iterators = {}
        for cl in self.cell_lines:
            indices = np.array(self.ctrl_indices_map[cl])
            if len(indices) > 0:
                ctrl_iterators[cl] = self._infinite_shuffled_iterator(indices)
            else:
                # 容错处理：如果某域无Control，返回空
                ctrl_iterators[cl] = (x for x in [])

        # 3. 生成 Batches
        pert_ptr = 0
        for _ in range(self.n_batches):
            batch_indices = []
            
            # --- Step A: 填充 Controls (Anchor) ---
            # 严格按照固定顺序填充，方便 Loss 计算时快速切片
            # 例如：前10个是RPE1_Ctrl，接下10个是HepG2_Ctrl...
            for i, cl in enumerate(self.cell_lines):
                count = self.ctrl_counts_per_domain[i]
                for _ in range(count):
                    try:
                        batch_indices.append(next(ctrl_iterators[cl]))
                    except StopIteration:
                        # 理论上不会发生，除非该域完全没有样本
                        pass
            
            # --- Step B: 填充 Perturbations ---
            take_n = min(self.n_perturbed, len(pert_indices) - pert_ptr)
            batch_indices.extend(pert_indices[pert_ptr : pert_ptr + take_n])
            pert_ptr += take_n
            
            # --- Step C: 补齐 (Drop Last逻辑) ---
            # 如果最后一个batch因为扰动样本不足导致长度不够，PyTorch DataLoader默认行为通常是允许的
            # 但如果你需要固定 Batch Size，可以从头补采样。
            # 这里我们允许最后一个 batch 略小（仅 Perturbed 部分变少，Control 部分依然完整）
            
            # 重要：虽然我们构造了结构化的Batch，但为了 BatchNorm 等层的稳定性，
            # 最好在 Batch 内部再次打乱？
            # 答：对于 MMD Loss 实现，保持结构化顺序（Controls在前，Perturbed在后）会让 Loss 计算代码非常简单（直接切片）。
            # 因此这里 *不* 打乱 Batch 内部顺序。
            # 如果模型对顺序敏感，请在模型 forward 前自行 shuffle。
            
            yield batch_indices

    def __len__(self):
        return self.n_batches

    def _infinite_shuffled_iterator(self, indices):
        """辅助生成器：无限次打乱并产生索引"""
        while True:
            np.random.shuffle(indices)
            for idx in indices:
                yield idx