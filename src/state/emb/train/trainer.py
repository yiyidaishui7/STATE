import os
import torch
import lightning as L
import copy

from torch import nn
from torch.utils.data import DataLoader
from datetime import timedelta

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from ..nn.model import StateEmbeddingModel
from ..data import H5adSentenceDataset, VCIDatasetSentenceCollator
from ..train.callbacks import (
    LogLR,
    ProfilerCallback,
    ResumeCallback,
    EMACallback,
    PerfProfilerCallback,
    CumulativeFLOPSCallback,
)
from ..utils import get_latest_checkpoint, get_embedding_cfg, get_dataset_cfg


# ============================================================================
# MMD-AAE 三路并行 DataLoader
# ============================================================================

class ParallelZipLoader:
    """
    并行混合加载器：用于 MMD-AAE 多域训练。
    同时从三个 DataLoader 中各取一个 Batch，
    合并为单个 batch 并附加域标签 (batch[9])。
    """
    def __init__(self, loaders, domain_names=None):
        self.loaders = loaders
        self.domain_names = domain_names or [f"domain_{i}" for i in range(len(loaders))]
        self._datasets = [loader.dataset for loader in loaders]
    
    def __iter__(self):
        for batches in zip(*self.loaders):
            yield self._merge_batches(batches)
    
    @staticmethod
    def _merge_batches(batches):
        """
        Merge N domain batches into a single batch, appending domain labels as batch[9].
        Each batch is a tuple of tensors (or None). We concatenate along dim=0.
        """
        num_fields = len(batches[0])
        merged = []
        domain_labels_list = []
        
        for field_idx in range(num_fields):
            field_items = [b[field_idx] for b in batches]
            if field_items[0] is None:
                merged.append(None)
            elif isinstance(field_items[0], torch.Tensor):
                merged.append(torch.cat(field_items, dim=0))
                if field_idx == 0:  # count batch sizes from the first tensor
                    for domain_id, item in enumerate(field_items):
                        domain_labels_list.append(
                            torch.full((item.size(0),), domain_id, dtype=torch.long)
                        )
            else:
                # For non-tensor fields, just concatenate as lists
                merged.append(field_items[0])
        
        # Append domain labels as batch[9]
        # Pad merged to ensure it has at least 10 elements
        while len(merged) < 9:
            merged.append(None)
        
        if domain_labels_list:
            domain_labels = torch.cat(domain_labels_list, dim=0)
        else:
            domain_labels = None
        merged.append(domain_labels)
        
        return tuple(merged)
    
    def __len__(self):
        return min(len(l) for l in self.loaders)

    @property
    def batch_size(self):
        return sum(l.batch_size for l in self.loaders)
    
    @property
    def num_workers(self):
        return self.loaders[0].num_workers
    
    @property
    def datasets(self):
        """返回所有 datasets，用于后续更新 cfg"""
        return self._datasets


def get_embeddings(cfg):
    """Load ESM2 embeddings and special tokens"""
    all_pe = torch.load(get_embedding_cfg(cfg).all_embeddings, weights_only=False)
    if isinstance(all_pe, dict):
        all_pe = torch.vstack(list(all_pe.values()))
    all_pe = all_pe.cuda()
    return all_pe


def create_domain_dataloaders(cfg, DatasetClass, collator, batch_size=32, num_workers=2):
    """
    创建三路并行 DataLoader
    
    Args:
        cfg: 配置对象
        DatasetClass: Dataset 类
        collator: Collator 函数
        batch_size: 每个域的 batch_size
        num_workers: DataLoader workers 数量
    
    Returns:
        ParallelZipLoader 实例
    """
    
    # 三个细胞系的数据路径 (服务器路径)
    BASE_DIR = "/media/mldadmin/home/s125mdg34_03/state"
    domain_configs = [
        {"name": "K562",   "path": f"{BASE_DIR}/competition_support_set/k562.h5"},
        {"name": "RPE1",   "path": f"{BASE_DIR}/competition_support_set/rpe1.h5"},
        {"name": "Jurkat", "path": f"{BASE_DIR}/competition_support_set/jurkat.h5"},
    ]
    
    parallel_loaders = []
    domain_names = []
    
    print(f"\n{'='*60}")
    print(f"[MMD-AAE] 开始初始化 3 路并行 DataLoaders...")
    print(f"{'='*60}")
    
    for domain in domain_configs:
        name = domain["name"]
        path = domain["path"]
        
        print(f"\n正在加载 {name}...")
        print(f"  路径: {path}")
        
        # 检查文件是否存在
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在: {path}")
        
        # 克隆配置，避免相互污染
        dom_cfg = copy.deepcopy(cfg)
        
        # 修改数据路径
        # 根据配置结构，设置正确的路径属性
        ds_cfg = get_dataset_cfg(dom_cfg)
        # 设置 train 路径为当前域的数据文件
        ds_cfg.train = path
        
        try:
            # 创建 Dataset
            dom_dataset = DatasetClass(dom_cfg)
            print(f"  ✓ Dataset 创建成功, 样本数: {len(dom_dataset)}")
            
            # 创建 DataLoader
            loader = DataLoader(
                dom_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collator,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False,
                pin_memory=True,
                drop_last=True  # 必须开启，防止尾部数据对不齐
            )
            parallel_loaders.append(loader)
            domain_names.append(name)
            print(f"  ✓ DataLoader 创建成功")
            
        except Exception as e:
            print(f"  ✗ 创建失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # 封装成并行加载器
    train_dataloader = ParallelZipLoader(parallel_loaders, domain_names)
    
    print(f"\n{'='*60}")
    print(f"[MMD-AAE] 并行加载器构建完成!")
    print(f"  - 成功加载的域: {domain_names}")
    print(f"  - 每个域 Batch Size: {batch_size}")
    print(f"  - 总 Batch Size: {train_dataloader.batch_size}")
    print(f"  - 每轮迭代次数: {len(train_dataloader)}")
    print(f"{'='*60}\n")
    
    return train_dataloader


def verify_parallel_dataloader(train_dataloader):
    """验证并行 DataLoader 输出格式 (merged batch with domain labels)"""
    print("\n" + "="*60)
    print("Verifying merged DataLoader output format...")
    print("="*60)
    
    try:
        # 取一个 batch
        test_batch = next(iter(train_dataloader))
        
        print(f"\nMerged batch info:")
        print(f"  - Type: {type(test_batch)}")
        print(f"  - Fields: {len(test_batch)}")
        
        for j, item in enumerate(test_batch):
            if item is None:
                print(f"  [{j}]: None")
            elif isinstance(item, torch.Tensor):
                print(f"  [{j}]: Tensor shape={item.shape}, dtype={item.dtype}")
            else:
                print(f"  [{j}]: {type(item)}")
        
        # Verify domain labels at batch[9]
        if len(test_batch) > 9 and test_batch[9] is not None:
            domain_labels = test_batch[9]
            print(f"\n  Domain labels (batch[9]):")
            print(f"    shape={domain_labels.shape}")
            print(f"    unique values={domain_labels.unique().tolist()}")
            for d in domain_labels.unique():
                cnt = (domain_labels == d).sum().item()
                print(f"    domain {d.item()}: {cnt} samples")
        else:
            print("\n  WARNING: No domain labels found at batch[9]")
        
        print("\n" + "="*60)
        print("Verification passed!")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\nVerification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main(cfg):
    """
    MMD-AAE 训练主函数
    """
    print(f"\n{'#'*60}")
    print(f"# MMD-AAE Training")
    print(f"# Embedding: {cfg.embeddings.current}")
    print(f"# Dataset: {cfg.dataset.current}")
    print(f"{'#'*60}\n")
    
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["NCCL_LAUNCH_TIMEOUT"] = str(cfg.experiment.ddp_timeout)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
    TOTAL_N_CELL = cfg.dataset.num_cells
    EPOCH_LENGTH = int(TOTAL_N_CELL // cfg.model.batch_size // 24)
    warmup_steps = EPOCH_LENGTH * 6

    # 创建 Collators
    train_collator = VCIDatasetSentenceCollator(cfg, is_train=True)
    val_collator = VCIDatasetSentenceCollator(cfg, is_train=False)

    generator = torch.Generator()
    generator.manual_seed(cfg.dataset.seed)

    if get_dataset_cfg(cfg).ds_type == "h5ad":
        DatasetClass = H5adSentenceDataset
    else:
        raise ValueError(f"Unknown dataset type: {get_dataset_cfg(cfg).ds_type}")
    
    # ====================================================================
    # 创建三路并行 DataLoader (MMD-AAE)
    # ====================================================================
    train_dataloader = create_domain_dataloaders(
        cfg, 
        DatasetClass, 
        train_collator,
        batch_size=32,   # 每个域 32，总共 96
        num_workers=2
    )
    
    # 验证 DataLoader
    verify_success = verify_parallel_dataloader(train_dataloader)
    if not verify_success:
        print("DataLoader 验证失败，退出训练")
        return
    
    # ====================================================================
    # 以下为测试模式 - 验证成功后可注释掉 exit() 继续训练
    # ====================================================================
    print("\n" + "="*60)
    print("DataLoader test passed!")
    print("="*60 + "\n")
    # return  # <-- Removed: proceed to full training
    
    # ====================================================================
    # 创建 Validation DataLoader (使用原始配置)
    # ====================================================================
    val_dataset = DatasetClass(cfg, test=True)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        collate_fn=val_collator,
        num_workers=cfg.dataset.num_val_workers,
        persistent_workers=True,
        generator=generator,
    )

    # ====================================================================
    # 创建模型
    # ====================================================================
    model = StateEmbeddingModel(
        token_dim=get_embedding_cfg(cfg).size,
        d_model=cfg.model.emsize,
        nhead=cfg.model.nhead,
        d_hid=cfg.model.d_hid,
        nlayers=cfg.model.nlayers,
        output_dim=cfg.model.output_dim,
        dropout=cfg.model.dropout,
        warmup_steps=warmup_steps,
        compiled=False,
        max_lr=cfg.optimizer.max_lr,
        emb_size=get_embedding_cfg(cfg).size,
        collater=val_collator,
        cfg=cfg,
    )
    
    model.update_config(cfg)
    
    # 更新所有 datasets 和 collators 的 cfg
    for ds in train_dataloader.datasets:
        ds.cfg = cfg
    val_dataset.cfg = cfg
    train_collator.cfg = cfg
    val_collator.cfg = cfg
    model.collater = val_collator
    
    model = model.cuda()
    all_pe = get_embeddings(cfg)
    all_pe.requires_grad = False
    model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
    model = model.train()

    # ====================================================================
    # 设置 Callbacks 和 Logger
    # ====================================================================
    run_name, chk = get_latest_checkpoint(cfg)
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=cfg.experiment.checkpoint.every_n_train_steps,
        dirpath=os.path.join(cfg.experiment.checkpoint.path, cfg.experiment.name),
        filename=f"{run_name}" + "-{epoch}-{step}",
        save_last=True,
        save_top_k=cfg.experiment.checkpoint.save_top_k,
        monitor=cfg.experiment.checkpoint.monitor,
    )

    if cfg.wandb.enable:
        try:
            import wandb
            exp_logger = WandbLogger(project=cfg.wandb.project, name=cfg.experiment.name)
            exp_logger.watch(model, log_freq=1000)
        except ImportError:
            print("Warning: wandb is not installed. Skipping wandb logging.")
            exp_logger = None
        except Exception as e:
            print(f"Warning: Failed to initialize wandb logger: {e}")
            exp_logger = None
    else:
        exp_logger = None

    callbacks = [checkpoint_callback, LogLR(100), ResumeCallback(cfg), PerfProfilerCallback()]

    if getattr(cfg.model, "ema", False):
        ema_decay = getattr(cfg.model, "ema_decay", 0.999)
        callbacks.append(EMACallback(decay=ema_decay))

    callbacks.append(CumulativeFLOPSCallback(use_backward=cfg.experiment.cumulative_flops_use_backward))

    max_steps = -1
    if cfg.experiment.profile.enable_profiler:
        callbacks.append(ProfilerCallback(cfg=cfg))
        max_steps = cfg.experiment.profile.max_steps

    # ====================================================================
    # 创建 Trainer 并开始训练
    # ====================================================================
    val_interval = int(cfg.experiment.val_check_interval * cfg.experiment.num_gpus_per_node * cfg.experiment.num_nodes)
    trainer = L.Trainer(
        max_epochs=cfg.experiment.num_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        devices=cfg.experiment.num_gpus_per_node,
        num_nodes=cfg.experiment.num_nodes,
        gradient_clip_val=cfg.optimizer.max_grad_norm,
        accumulate_grad_batches=cfg.optimizer.gradient_accumulation_steps,
        precision="bf16-mixed",
        strategy=DDPStrategy(
            process_group_backend="nccl",
            find_unused_parameters=False,
            timeout=timedelta(seconds=cfg.experiment.get("ddp_timeout", 3600)),
        ),
        val_check_interval=val_interval,
        logger=exp_logger,
        fast_dev_run=False,
        limit_val_batches=cfg.experiment.limit_val_batches,
    )

    if chk:
        print(f"******** Loading checkpoint {run_name} {chk}...")
    else:
        print(f"******** Initialized fresh {run_name}...")

    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader, 
        ckpt_path=chk
    )

    trainer.save_checkpoint(os.path.join(cfg.experiment.checkpoint.path, f"{run_name}_final.pt"))
    print("\n✅ Training completed!")
