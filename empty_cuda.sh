#!/bin/bash
# 这是一个用于杀死所有 NVIDIA GPU 计算进程的脚本

# 获取所有计算进程的 PID
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)

# 如果没有找到任何进程，则退出
if [ -z "$PIDS" ]; then
  echo "没有找到正在运行的 NVIDIA GPU 计算进程。"
  exit 0
fi

echo "将要终止以下 NVIDIA GPU 进程: $PIDS"

# 循环并杀死每个进程
for PID in $PIDS; do
  echo "正在终止 PID: $PID"
  kill $PID
done

echo "完成。请再次运行 nvidia-smi 检查状态。"
