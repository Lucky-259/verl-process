### 配置环境
conda create -n cky_process python=3.10

首先进入verl-process文件夹
```bash
cd verl-process
```
然后，执行命令安装所需的包
```bash
pip install -e .
```
最后，安装flash-attn(如果无法下载，可以去官网(https://github.com/Dao-AILab/flash-attention/releases)下载对应Python版本的.whl文件，本项目用的是flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)
```bash
pip install flash-attn --no-build-isolation
```
可以参考requirement_process.txt里面的环境配置。

### 下载模型
将${your_DeepSeek-R1-Distill-Qwen-1.5B_path}替换为您的DeepSeek-R1-Distill-Qwen-1.5B要下载到的本地路径
```bash
huggingface-cli download DeepSeek-R1-Distill-Qwen-1.5B \
  --local-dir ${your_DeepSeek-R1-Distill-Qwen-1.5B_path} \
  --local-dir-use-symlinks False
```

### 启动vllm服务
在服务器B的命令行输入启动vllm服务命令，将${your_DeepSeek-V3.2}替换为您下载的deepseek-ai/DeepSeek-V3.2模型路径，部署服务的IP地址需要获取（利用hostname -I命令即可输出本机IPV4地址），端口号自行设置（我们以8000为例部署服务），设置完请到服务器A中的verl-process/recipe/multi_prm/reward_function.py中把 VLLM_API_BASE = "http://localhost:8000/" 这条语句中的localhost改为服务器B的IPV4地址；8000改为修改后的端口号

多卡，修改显卡数量与-dp后面的数字对应：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve ${your_Qwen3-32B} --max-model-len 16384 --host 0.0.0.0 --port 8000 -dp 8 -tp 8
```

### 训练
在verl-process目录下运行脚本，将${your_DeepSeek-R1-Distill-Qwen-1.5B_path}换成您下载的DeepSeek-R1-Distill-Qwen-1.5B模型路径
```bash
bash scripts/rl_1.5B.sh --model ${your_DeepSeek-R1-Distill-Qwen-1.5B_path}
```
