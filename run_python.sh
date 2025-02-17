#!/bin/bash

# 检查是否传递了两个参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <python_file> <output_file>"
    exit 1
fi

# 获取参数
python_file=$1
output_file=$2

# 在后台运行Python文件并重定向输出
nohup python3 "$python_file" > "$output_file" 2>&1 &

# 获取后台运行的进程ID
pid=$!

# 输出进程ID信息
echo "Python script '$python_file' is running in the background with PID $pid. Output is redirected to '$output_file'."
