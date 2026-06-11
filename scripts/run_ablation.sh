#!/bin/bash
# 消融实验自动运行脚本
set -e

PYTHON=/opt/anaconda3/envs/MonoGS/bin/python
CDIR=/home/easyai/MonoGS
cd "$CDIR"

clean_gpu() {
    # 清理残留 GPU 进程
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 2>/dev/null | grep -v "^1550$"); do
        kill "$pid" 2>/dev/null || true
    done
    sleep 2
}

run_and_rename() {
    local config=$1
    local new_name=$2
    local log_file="/tmp/ablation_${new_name}.log"

    echo "[$(date '+%H:%M')] Starting: $config → $new_name"

    # 清理显存
    clean_gpu

    # 运行
    WANDB_MODE=offline $PYTHON slam.py --config "$config" --eval 2>&1 | tee "$log_file"
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        # 找到最新生成的结果目录
        latest_dir=$(grep "saving results in" "$log_file" | tail -1 | awk '{print $NF}')
        if [ -n "$latest_dir" ] && [ -d "$latest_dir" ]; then
            # 重命名
            parent=$(dirname "$latest_dir")
            mv "$latest_dir" "$parent/$new_name"
            echo "[$(date '+%H:%M')] ✅ $new_name complete → $parent/$new_name"
        else
            echo "[$(date '+%H:%M')] ⚠️ $new_name ran but no dir found"
        fi
    else
        echo "[$(date '+%H:%M')] ❌ $new_name failed (exit $exit_code)"
        clean_gpu
    fi
}

echo "=== 消融实验开始 $(date) ==="

# === TUM (Monocular, ~30 min each) ===
run_and_rename "configs/mono/tum/fr3_office_imp1.yaml" "tum_imp1"
run_and_rename "configs/mono/tum/fr3_office_imp2.yaml" "tum_imp2"
run_and_rename "configs/mono/tum/fr3_office_imp3.yaml" "tum_imp3"
run_and_rename "configs/mono/tum/fr3_office_imp4.yaml" "tum_imp4"

# === Replica (RGB-D, ~75 min each) ===
run_and_rename "configs/rgbd/replica/room0_imp1.yaml" "rep_imp1"
run_and_rename "configs/rgbd/replica/room0_imp2.yaml" "rep_imp2"
run_and_rename "configs/rgbd/replica/room0_imp3.yaml" "rep_imp3"
run_and_rename "configs/rgbd/replica/room0_imp4.yaml" "rep_imp4"

echo "=== 全部完成 $(date) ==="
