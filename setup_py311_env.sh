#!/bin/zsh
# 自动创建Python 3.11虚拟环境并安装依赖
# 需本机已安装python3.11（可用 `brew install python@3.11` 安装）

cd "$(dirname "$0")"

# 1. 创建3.11虚拟环境
python3.11 -m venv .venv311
source .venv311/bin/activate

# 2. 升级pip
pip install --upgrade pip

# 3. 安装依赖
pip install pandas numpy pillow librosa soundfile moviepy

# 4. 提示

echo "\n已创建并激活Python 3.11虚拟环境(.venv311)，依赖已装好。"
echo "后续运行："
echo "  source .venv311/bin/activate"
echo "  python OceanWaveFinal.py"
