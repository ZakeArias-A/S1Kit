# 参与开发

依赖由 `pyproject.toml` 和提交到仓库的 `uv.lock` 管理。开发工具放在 `dependency-groups.dev`；修改依赖后运行 `uv lock` 并提交锁文件。构建发行包使用 `uv build`。

安装与测试：

```sh
uv sync --locked --extra cpu
uv run --no-sync python -m pytest -q
```

普通测试使用随机小模型，不下载权重。涉及真实模型的性能或准确率变化，按 [benchmarks](benchmarks/README.md) 的方法验证，并记录模型 revision、设备、精度和输入规模。

核心代码位于 `src/s1kit/`；模型专用接入放在 `examples/`。修改公开参数或返回值时同步更新文档和相应示例，修复行为问题时添加能复现问题的测试。

提交前运行 `git diff --check`。不要提交凭证、模型权重、运行日志或本地环境；评测原始输出放在 `runs/`，经核对的公开结果放在 `benchmarks/results/`。

报告问题时附上最小复现、完整异常及 Python、PyTorch、Transformers 版本。提交变更时说明问题、修改后的行为和验证方法。
