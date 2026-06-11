#!/bin/bash
# 清理残留 GPU 进程
# 每次跑 slam.py 之前执行一次：bash scripts/clean_gpu.sh

echo "检查 GPU 残留进程..."
pids=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i 0 2>/dev/null | grep -v "^1550" | awk -F', ' '{print $1}')

if [ -z "$pids" ]; then
    echo "GPU 干净，无需清理"
    exit 0
fi

echo "发现残留进程:"
for pid in $pids; do
    cmd=$(ps -p $pid -o cmd= 2>/dev/null | head -c 80)
    mem=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader -i 0 2>/dev/null | head -$pid | tail -1)
    echo "  PID $pid ($cmd)"
done

echo ""
echo "杀掉这些进程? (y/n)"
read -r answer
if [ "$answer" = "y" ]; then
    kill $pids 2>/dev/null
    sleep 2
    echo "已清理"
fi
