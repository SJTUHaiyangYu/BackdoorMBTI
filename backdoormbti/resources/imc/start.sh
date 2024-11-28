#!/bin/bash

# Python 脚本路径
python_script="./example.py"

# YAML 文件路径
yaml_file="imc.yaml"

# 读取 YAML 文件并解析参数
args=$(python - <<END
import yaml
process = ""
with open('$yaml_file', 'r') as f:
    args = yaml.safe_load(f)
 
# 构建参数字符串
    for k, v in args.items():
        process = process + " --" + k + " " + str(v)

print(process)
END
)

# 构建最终的命令
command="python $python_script $args"

# 打印最终的命令以便调试
echo "Running command: $command"

# 执行命令
eval $command

