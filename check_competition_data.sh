#!/bin/bash

#############################################################
# 全面检查competition_support_set中的所有数据
#############################################################

cd ~/state

echo "========================================"
echo "检查Competition数据集"
echo "========================================"
echo ""

DATA_DIR="competition_support_set"

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 目录不存在: $DATA_DIR"
    echo ""
    echo "请确认数据目录位置"
    exit 1
fi

echo "数据目录: $DATA_DIR"
echo ""

# ============================================
# 1. 列出所有文件
# ============================================
echo "========================================"
echo "1. 所有文件"
echo "========================================"
echo ""

ls -lh "$DATA_DIR" | grep -E "\.h5|\.h5ad"

echo ""

# ============================================
# 2. 统计文件数量
# ============================================
echo "========================================"
echo "2. 文件统计"
echo "========================================"
echo ""

H5_COUNT=$(ls "$DATA_DIR"/*.h5 2>/dev/null | wc -l)
H5AD_COUNT=$(ls "$DATA_DIR"/*.h5ad 2>/dev/null | wc -l)

echo "  .h5 文件: $H5_COUNT 个"
echo "  .h5ad 文件: $H5AD_COUNT 个"

echo ""

# ============================================
# 3. 详细检查每个cell type
# ============================================
echo "========================================"
echo "3. Cell Type 数据详情"
echo "========================================"
echo ""

CELL_TYPES=("k562" "jurkat" "rpe1" "hepg2" "h1")

for ct in "${CELL_TYPES[@]}"; do
    echo "检查 $ct:"
    
    # 检查.h5
    if [ -f "$DATA_DIR/${ct}.h5" ]; then
        SIZE=$(ls -lh "$DATA_DIR/${ct}.h5" | awk '{print $5}')
        echo "  ✓ ${ct}.h5 存在 (大小: $SIZE)"
        
        # 尝试读取样本数
        python3 << PYEOF 2>/dev/null
import h5py
import anndata as ad
try:
    adata = ad.read_h5ad("$DATA_DIR/${ct}.h5")
    print(f"    样本数: {adata.n_obs:,}")
    print(f"    基因数: {adata.n_vars:,}")
except Exception as e:
    print(f"    无法读取详细信息")
PYEOF
        
    # 检查.h5ad
    elif [ -f "$DATA_DIR/${ct}.h5ad" ]; then
        SIZE=$(ls -lh "$DATA_DIR/${ct}.h5ad" | awk '{print $5}')
        echo "  ✓ ${ct}.h5ad 存在 (大小: $SIZE)"
        
        python3 << PYEOF 2>/dev/null
import anndata as ad
try:
    adata = ad.read_h5ad("$DATA_DIR/${ct}.h5ad")
    print(f"    样本数: {adata.n_obs:,}")
    print(f"    基因数: {adata.n_vars:,}")
except Exception as e:
    print(f"    无法读取详细信息")
PYEOF
        
    else
        echo "  ❌ ${ct} 数据不存在"
    fi
    
    echo ""
done

# ============================================
# 4. H1特别说明
# ============================================
echo "========================================"
echo "4. H1数据集说明"
echo "========================================"
echo ""

if [ -f "$DATA_DIR/h1.h5" ] || [ -f "$DATA_DIR/h1.h5ad" ]; then
    echo "✓✓✓ 发现H1数据集！"
    echo ""
    echo "H1 (H1-hESC) 是人类胚胎干细胞系"
    echo ""
    echo "推荐使用方式："
    echo "  - 训练集: K562 + Jurkat + RPE1"
    echo "  - 验证集: HepG2"
    echo "  - 测试集: H1  ← 作为额外的held-out评估"
    echo ""
    echo "配置示例："
    echo "----------------------------------------"
    cat << 'EOF'
[datasets]
train_data = "competition_support_set/{k562,jurkat,rpe1}.h5"
val_data = "competition_support_set/hepg2.h5"
test_data = "competition_support_set/h1.h5"

[training]
train_data = "train"
val_data = "val"
test_data = "test"
EOF
    echo "----------------------------------------"
    
else
    echo "❌ 未找到H1数据集"
    echo ""
    echo "没有H1的替代方案："
    echo ""
    echo "方案A：使用RPE1作为测试集"
    echo "  - 训练: K562 + Jurkat"
    echo "  - 验证: HepG2"
    echo "  - 测试: RPE1"
    echo ""
    echo "方案B：分割HepG2"
    echo "  - 将HepG2分成val和test"
    echo "  - 需要预处理"
    echo ""
    echo "方案C：只用验证集（推荐）"
    echo "  - 训练: K562 + Jurkat + RPE1"
    echo "  - 验证: HepG2（已经是held-out）"
    echo "  - 不需要额外测试集"
fi

echo ""

# ============================================
# 5. 其他文件
# ============================================
echo "========================================"
echo "5. 其他重要文件"
echo "========================================"
echo ""

# 检查ESM2特征
if [ -f "$DATA_DIR/ESM2_pert_features.pt" ]; then
    SIZE=$(ls -lh "$DATA_DIR/ESM2_pert_features.pt" | awk '{print $5}')
    echo "  ✓ ESM2_pert_features.pt (大小: $SIZE)"
else
    echo "  ❌ ESM2_pert_features.pt 不存在"
fi

echo ""

# ============================================
# 6. 总结
# ============================================
echo "========================================"
echo "6. 总结和建议"
echo "========================================"
echo ""

# 统计可用的cell types
AVAILABLE=()
for ct in "${CELL_TYPES[@]}"; do
    if [ -f "$DATA_DIR/${ct}.h5" ] || [ -f "$DATA_DIR/${ct}.h5ad" ]; then
        AVAILABLE+=("$ct")
    fi
done

echo "可用的cell types: ${AVAILABLE[*]}"
echo ""

# 根据可用数据给出建议
if [[ " ${AVAILABLE[*]} " =~ " h1 " ]]; then
    echo "✓ 推荐配置（包含H1）："
    echo "  训练: K562 + Jurkat + RPE1"
    echo "  验证: HepG2"
    echo "  测试: H1"
elif [ ${#AVAILABLE[@]} -ge 4 ]; then
    echo "✓ 推荐配置（使用4个cell types）："
    echo "  训练: K562 + Jurkat"
    echo "  验证: HepG2"
    echo "  测试: RPE1"
elif [ ${#AVAILABLE[@]} -ge 3 ]; then
    echo "✓ 推荐配置（使用3个cell types）："
    echo "  训练: K562 + Jurkat"
    echo "  验证: RPE1（或HepG2）"
    echo "  测试: 不需要"
else
    echo "⚠️  数据不足，至少需要3个cell types"
fi

echo ""
echo "========================================"