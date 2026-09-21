# 快速开始

[文档首页](README.md)

需要 Python 3.11+、[uv](https://docs.astral.sh/uv/getting-started/installation/) 和一个本地因果语言模型。模型应提供支持 system/user 消息的聊天模板和普通 `torch.nn.Linear` 输出头。

在仓库根目录安装：

```sh
uv sync --locked --extra cpu --extra transformers
```

uv 在项目内创建 `.venv`，按 `uv.lock` 安装依赖。NVIDIA CUDA 13.0 环境改用 `uv sync --locked --extra cu130 --extra transformers`；运行 Qwen 图文示例时再将 `transformers` 换成 `qwen`。CPU 与 cu130 不能同时选择，其他设备使用基座支持的 PyTorch 构建。

下方代码保存为 Python 文件后，用 `uv run --no-sync python 文件名.py` 运行。`--no-sync` 保留刚才选择的 PyTorch 后端；切换设备时应先更新环境。

以下代码在 CPU 上加载本地模型，不下载权重：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from s1kit import SystemOne

path = "./my-model"  # 改为本地模型目录
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    path, local_files_only=True, dtype=torch.float32,
)
engine = SystemOne.from_transformers(model, tokenizer)
answer = engine.choice(
    state={"message": "PDF 导出失败，但 CSV 可用。"},
    instructions="选择主要意图。",
    options={"bug": "报告功能故障", "feature": "请求新增功能"},
)
print(answer["choice"])
print(answer["probabilities"])
```

返回候选键和完整概率分布，实际结果取决于基座。也可运行：

```sh
uv run --no-sync python examples/text.py ./my-model
uv run --no-sync python examples/text.py ./my-model --customize
```

CUDA 使用 `--device cuda`，显存需求取决于模型和输入长度。量化由调用方负责，输出头须保持普通稠密线性层。

下一步阅读[输入与提示词](inputs.md)。没有聊天模板或需要多模态处理时，参阅 [API 的预处理契约](api.md#预处理)及可运行示例。

## 如何读取结果

Choice 的 `choice` 是概率最高的候选键，`probabilities` 包含全部候选；概率总和为 1，仅描述所给候选之间的相对权重。增加或改变候选后，分布也可能改变。

Score 返回从 0 开始的等级索引期望，不是取整后的等级。Noul 返回真值概率，是否将其转换为布尔值以及采用何种阈值，由业务决定。confidence 是分布统计量，不是正确率。

## 常见接入问题

| 情况 | 处理方式 |
|---|---|
| tokenizer 没有聊天模板，或模板不支持 system 消息 | 按基座调用方式提供 `preprocess`，参考 [GPT-2 示例](../examples/gpt2.py) |
| 找不到独立主干 | 用 `backbone=` 显式传入主干，或使用 `SystemOne(...)` 绑定 |
| 标签不是单 token，或答案边界变化 | 检查模板与分词结果，配置合适的 [labels](api.md#标签与上下文限制) |
| 输入超过 `max_tokens` | 在基座支持范围内提高预算，或缩短证据；不要截断候选 |
| 输出头类型不支持 | 保留原生稠密 `nn.Linear` 输出头；量化与自定义输出头不能直接绑定 |
| 希望传图片或音频 | 绑定基座原生 processor；默认 `from_transformers` 预处理仅支持文本 |

模型加载失败、内存不足等问题应先确认基座能在相同设备和精度下独立运行，再接入 S1Kit。
