# 使用文档

S1Kit 将已加载的因果语言模型用于选择、评分和真假判断。每题一次主干前向，从候选 logits 计算结果，不训练模型或生成回答文本。

| 你要做什么 | 阅读 |
|---|---|
| 加载模型并完成第一次判断 | [快速开始](quickstart.md) |
| 配置业务数据、系统提示词、结构化条件 | [输入与提示词](inputs.md) |
| 查询参数、返回值、接入自己的预处理 | [API 参考](api.md) |
| 对照 TypeSafe 的决策功能 | [功能对照](compatibility.md) |
| 运行文本和图文脚本 | [Examples](../examples/README.md) |
| 查看图文输入和复杂多图案例 | [图文判断案例](../examples/vision/README.md) |
| 查看准确率、失败案例与复现方法 | [Benchmarks](../benchmarks/README.md) |

调用流程：`state + instructions + criteria + system_prompt → 预处理 → 主干前向 → 候选 logits → 概率与决策`。图片等媒体由预处理传给模型，相应能力来自基座。

## 仓库布局

| 目录 | 内容 |
|---|---|
| `src/s1kit/` | 请求校验、提示构造、模型调用与概率读出 |
| `docs/` | 教程与 API 参考 |
| `examples/` | 接入脚本、请求样例和图文素材 |
| `benchmarks/` | 评测工具、配置和已发布结果 |
| `tests/` | 离线测试和共享夹具 |

运行输出写入被忽略的 `runs/`，本地缓存使用 `.cache/`；模型权重和临时构建文件不提交。发布结果集中保存在 `benchmarks/results/`。
