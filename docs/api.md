# API

[文档首页](README.md) · [快速开始](quickstart.md) · [输入与提示词](inputs.md) · [功能对照](compatibility.md)

## 创建引擎

```python
engine = SystemOne.from_transformers(
    model,
    tokenizer,
    max_tokens=4096,
    template_kwargs={"enable_thinking": False},
    system_prompt="根据材料比较全部候选，只回答候选标签。",
)
```

默认使用 `model.base_model` 和 `model.get_output_embeddings()`，预处理只接收文本。tokenizer 必须有可处理 system/user 消息的聊天模板。`enable_thinking=False` 是否生效取决于模板。

需要特殊输入或位置处理时传入 `preprocess=`，必要时用 `backbone=` 覆盖主干。自定义预处理不能同时使用 `template_kwargs`，模板配置应写在函数内部。

对于非 Transformers 模型或不同的输出结构，可以直接绑定：

```python
engine = SystemOne(
    preprocess=prepare,
    backbone=backbone,
    head=output_head,
    tokenizer=tokenizer,
    hidden=lambda output: output.last_hidden_state,
    max_tokens=4096,
)
```

主干和输出层会设为 eval 模式，调用在 `torch.inference_mode()` 中执行。调用方负责模型加载、设备和生命周期。

两种构造方式均接受 `system_prompt`（字符串，省略时使用内置提示词）与 `labels`（可选标签列表或元组）。自定义提示词完整替换默认值。模型实例应串行使用。

## 单题

```python
engine.choice(state, instructions, {"accept": "接受", "review": "人工复核"})
engine.score(state, instructions, ["低", "中", "高"])
engine.noul(state, instructions, criteria={"false": "不成立", "true": "成立"})
```

三个方法都支持 `temperature=1.0`、`system_prompt=None` 和传给预处理的媒体参数，例如 `images=[image]`。`None` 沿用引擎默认提示词，字符串仅覆盖本次。直接返回答案字典：

- Choice：`choice`、`probabilities`、`confidence`。支持 2–255 项，同概率取第一项；描述可为字符串、对象、数组或 null。
- Score：`score`、`legend`、`probabilities`、`confidence`。支持 2–10 个非空字符串、对象或数组等级。score 为等级索引期望 `Σ(i × p_i)`，索引从 0 开始；结构化等级在 legend 中为 JSON 字符串。
- Noul：`{"noul": p_true}`，不自动转换为布尔值。

温度必须是正有限值。概率只在指定候选内归一化；改变温度不改变最大项，但会改变分布及 Score 的期望。

## 多题和完整输出

```python
result = engine.decide({
    "state": {"message": "PDF 导出失败，但 CSV 可用。"},
    "questions": {
        "intent": {
            "type": "choice",
            "instructions": "选择主要意图。",
            "criteria": {"bug": "功能故障", "feature": "新增功能"},
        },
        "workaround": {
            "type": "noul",
            "instructions": "是否有可用的替代方式？",
        },
    },
})
record = result["questions"]["intent"]
```

每题包含 `answer`、`option_logits`、`input_tokens` 和 `input_ids_sha256`。token 指纹不含图像像素，不能用作多模态缓存键。顶层包含输出契约版本、提示版本、温度和概率状态。

顶层 `usage.input_tokens` 为各题实际前向长度合计，重复 state 重复计数，不包含标签预处理探测；`usage.output_tokens` 为 0，因为没有生成 token。这不是托管 API 的计费口径。`system_prompt_sha256` 指纹记录配置或覆盖后的系统提示词，默认扩展标签的措辞调整不计入此指纹。

`decide(request, *, temperature=1.0, system_prompt=None, **media)` 支持同样的按次覆盖。请求字典只接受 `state` 和 `questions`；system_prompt 是调用关键字，不是请求字段。

state 接受有限 JSON 值；问题和候选保留插入顺序。未知字段、无效候选及 NaN/Infinity 会被拒绝。每题只看到共享材料和自身的全部候选，逐题独立执行。

## 预处理

```python
def prepare(messages, *, answer="", **media):
    # 1. 按模型要求构造消息、应用聊天模板。
    # 2. 将 answer 追加到渲染后的答案位置，再分词。
    # 3. 返回主干接受的参数字典，处理设备、媒体和位置。
    ...
```

- `input_ids` 必须为 `[1, sequence]` 的整数张量，不截断、不 padding。若有 attention_mask，须同形状且全为 1。
- `hidden(output)` 必须返回 `[1, sequence, head.in_features]`；输出头为普通稠密 `nn.Linear`。
- 预处理应返回新的字典和张量，不修改此前返回的输入，不携带上一题状态。
- 图片等数据作为 `decide()` 或单题方法的额外关键字参数传入，格式由预处理函数约定。音频未提供集成示例。

每题会额外预处理各候选标签，验证实际 token 序列满足 `encode(prompt + label) == encode(prompt) + [label_token]`。只有不含候选答案的原输入进入主干。多模态 token 展开也参与检查，因此图片预处理可能重复执行。

具体实现见 [文本预处理](../examples/gpt2.py)和[图文预处理](../examples/qwen.py)。

## 标签与上下文限制

不超过 26 项默认使用 A–Z；27–255 项使用数字字符串 0–254。标签必须独立编码为单个 token，并在实际输入末尾保持边界稳定。失败会在该题前向前报错。

用 `labels=[...]` 为基座提供其他标签，须为足够多的不重复非空字符串，每题取前 N 项。内部候选 JSON 的 `letter` 字段承载标签，即使标签为数字。自定义系统提示词应要求回答该标签。

255 项是请求上限，不保证每个 tokenizer 都支持。全部候选必须可见；不能通过分批或截断绕过一次前向契约。`max_tokens` 为单题输入预算，不会扩大基座上下文容量。S1Kit 不自动发现模态编码器，也不提供编码器缓存。

例如 Qwen3.5 的 tokenizer 会拆分多位数字。需要大量候选时，可以从 tokenizer 的词表选择标签后显式绑定：

```python
import re

labels = sorted(
    (text for text in tokenizer.get_vocab()
     if re.fullmatch(r"[A-Za-z]{1,8}", text)
     and len(tokenizer.encode(text, add_special_tokens=False)) == 1
     and tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) == text),
    key=lambda text: (len(text), text),
)[:255]
engine = SystemOne.from_transformers(
    model, tokenizer, labels=labels, max_tokens=8192,
)
```

这只是标签选取示例；仍由每次调用检查真实模板边界，输入预算也须适合基座。词语标签可能引入语义偏差，扩大候选集后应重新评估任务准确率。
