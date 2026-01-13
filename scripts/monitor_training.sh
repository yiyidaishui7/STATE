#!/bin/bash

#############################################################
# 实时监控训练进度并生成图表
# 每隔N秒更新一次图表
#############################################################

# 默认参数
INTERVAL=300  # 5分钟更新一次
LOG_DIR=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查参数
if [ -z "$LOG_DIR" ]; then
    echo "用法: $0 --log-dir <日志目录> [--interval <秒数>]"
    echo "示例: $0 --log-dir results/baseline_20251110_123456 --interval 300"
    exit 1
fi

# 检查目录
if [ ! -d "$LOG_DIR" ]; then
    echo "❌ 目录不存在: $LOG_DIR"
    exit 1
fi

echo "=========================================="
echo "实时监控训练进度"
echo "=========================================="
echo "日志目录: $LOG_DIR"
echo "更新间隔: ${INTERVAL}秒"
echo "按 Ctrl+C 停止监控"
echo "=========================================="
echo ""

COUNTER=1

while true; do
    CURRENT_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$CURRENT_TIME] 更新 #$COUNTER"
    
    # 检查训练是否还在进行
    if [ -f "$LOG_DIR/training.log" ]; then
        # 显示最新的几行日志
        echo "最新日志:"
        tail -5 "$LOG_DIR/training.log" | sed 's/^/  /'
        echo ""
        
        # 生成图表
        echo "生成图表..."
        python scripts/plot_training.py --log-dir "$LOG_DIR" --smooth-window 10 2>&1 | grep "✓" | sed 's/^/  /'
        echo ""
        
        # 显示简要统计
        if [ -f "$LOG_DIR/plots/training_summary.txt" ]; then
            echo "训练进度:"
            grep -E "(总训练步数|最终|最小)" "$LOG_DIR/plots/training_summary.txt" | sed 's/^/  /'
        fi
    else
        echo "  等待训练开始..."
    fi
    
    echo ""
    echo "下次更新: $((INTERVAL))秒后"
    echo "=========================================="
    echo ""
    
    COUNTER=$((COUNTER + 1))
    sleep $INTERVAL
done