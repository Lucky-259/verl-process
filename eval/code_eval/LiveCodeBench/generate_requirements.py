import toml

# 读取 uv.lock 文件
lock_file = "code_eval/LiveCodeBench/uv.lock"
with open(lock_file, "r", encoding="utf-8") as f:
    data = toml.load(f)

requirements = []

# 遍历 package 列表
for pkg in data.get("package", []):
    name = pkg["name"]
    version = pkg["version"]
    requirements.append(f"{name}=={version}")

# 写入 requirements.txt
with open("code_eval/LiveCodeBench/requirements.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(requirements))

print("requirements.txt 已生成")