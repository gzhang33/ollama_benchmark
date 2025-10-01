# Ollama 模型性能测试工具

这是一个用于测试本地 Ollama 模型性能的工具包，帮助你选择最适合本地运行的模型。

## 🆕 新增功能亮点

### 核心优化功能
- **交互式并行测试**：单界面输入，所有模型并行回答，结果横向对比
- **标准化速度测试**：15个标准测试问题，准确速度计算公式（总Token数÷总耗时）
- **资源监控**：实时监控GPU/CPU使用情况
- **测试工具菜单**：友好的菜单界面，快速选择测试工具

### 时间效率提升
- **原质量测试**：49样本 × 90秒 = 73分钟/模型
- **新交互测试**：1问题 × 30秒 = 0.5分钟（所有模型并行）
- **节省时间：99%+**

## 🔧 安装要求

- Python 3.8+
- 已安装并运行的 Ollama 服务（默认 `http://localhost:11434`）
- 依赖安装：
```bash
pip install -r requirements.txt
```

## 📁 文件结构

```
ollama_test/
├── README.md                    # 本文件（完整使用指南）
├── analyzer.py                  # 智能可视化分析工具
├── complete_test.py             # 一键完整工作流程
├── interactive_test.py          # 🆕 交互式并行测试工具
├── speed_test.py                # 🆕 速度测试工具（含资源监控）
├── test_menu.py                 # 🆕 测试工具菜单
├── test_prompts.txt             # 🆕 标准化测试问题集
├── requirements.txt             # Python 依赖包
├── test_result/                 # 测试结果存储目录
├── speed_test_results/          # 🆕 速度测试结果目录
├── interactive_test_results/    # 🆕 交互式测试结果目录
└── analysis_results/            # 分析报告目录
```

## 🚀 快速开始

### 🆕 方法1：使用测试工具菜单（最简单）
```bash
python test_menu.py
```
提供友好的菜单界面，选择需要的测试工具。

### 🆕 方法2：交互式并行测试（推荐用于模型对比）
```bash
python interactive_test.py
```
**核心功能**：
- 单界面输入提示词
- 所有模型并行回答
- 结果并排展示对比
- 实时查看所有模型回答质量

**使用示例**：
```bash
python interactive_test.py
💬 Your prompt: 请解释什么是量子计算？
# 等待所有模型并行回答，查看对比结果
```

### 🆕 方法3：速度测试（推荐用于性能评估）
```bash
python speed_test.py
```
**核心功能**：
- 15个标准化测试问题（简单/中等/复杂各5个）
- 准确速度计算：总Token数 ÷ 总耗时
- GPU/CPU资源监控
- 详细报表生成（CSV + JSON）

**测试问题集**：
- 简单：你好、自我介绍、基础问答
- 中等：编程任务、概念解释、算法实现
- 复杂：架构设计、深度分析、技术解析

**使用示例**：
```bash
python speed_test.py                    # 测试所有模型
python speed_test.py --model "gemma3:4b"  # 测试指定模型
```

### 方法4：一键完整工作流程
```bash
python complete_test.py
```
- 性能测试 + 分析报告

### 方法5：分步执行
```bash
# 运行测试
python speed_test.py

# 自动分析
python analyzer.py
```

## 📚 工具功能详解

### 🆕 新增工具

#### 1. 交互式并行测试（interactive_test.py）
**核心优势**：
- 替代原质量测试，节省99%+时间
- 实时对比所有模型回答质量
- 支持自定义问题测试

**参数说明**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base-url` | http://localhost:11434 | Ollama 服务地址 |
| `--prompt` | 无 | 单次测试的提示词 |
| `--output-dir` | interactive_test_results | 结果保存目录 |
| `--max-workers` | 5 | 并行工作线程数 |

#### 2. 速度测试（speed_test.py）
**核心优势**：
- 科学的速度计算公式
- 标准化测试问题集
- 完整的资源监控

**速度计算公式**：
```
准确平均速度（tokens/s）= 所有问题的输出Token总数 ÷ 所有问题的总耗时（秒）
```

**输出文件**：
- `speed_test_details_[时间戳].csv` - 详细测试数据
- `speed_test_summary_[时间戳].csv` - 汇总统计表格
- `speed_test_results_[时间戳].json` - 完整JSON数据

**参数说明**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base-url` | http://localhost:11434 | Ollama 服务地址 |
| `--prompts-file` | test_prompts.txt | 测试问题集文件 |
| `--model` | 无 | 指定测试的模型 |
| `--output-dir` | speed_test_results | 结果保存目录 |

#### 3. 测试工具菜单（test_menu.py）
提供友好的菜单界面：
- [1] 交互式并行测试
- [2] 速度测试
- [3] 原有性能测试
- [4] 分析工具
- [5] 完整测试流程

### 原有工具

#### 速度测试（speed_test.py）
- **自动模型发现**：自动发现本地已安装的 LLM 模型（自动过滤 embedding 模型）
- **标准化测试**：使用多难度级别的标准化问题集进行测试
- **准确速度计算**：基于实际 token 数计算精确的速度指标
- **资源监控**：实时监控 GPU/CPU 使用情况
- **进度显示**：实时进度条和动态 ETA 计算

#### 分析工具（analyzer.py）
- **智能数据转换**：自动识别JSON/CSV格式并转换
- **性能排名表**：基于综合评分的模型排名
- **可视化图表**：吞吐量、响应时间、雷达图对比
- **场景推荐**：不同使用场景下的最佳模型推荐
- **详细报告**：完整的Markdown格式分析报告

## 🎯 使用场景指南

### 场景1：我想快速对比所有模型的回答质量
**推荐工具**：`interactive_test.py`
```bash
python interactive_test.py
💬 Your prompt: 请解释什么是量子计算？
# 查看所有模型的回答对比
```

### 场景2：我想系统化评估模型速度性能
**推荐工具**：`speed_test.py`
```bash
python speed_test.py
# 运行15个标准测试，生成详细速度报告
```

### 场景3：我想对比两个特定模型的性能
**推荐工具**：`speed_test.py` 分别测试
```bash
python speed_test.py --model "model1"
python speed_test.py --model "model2"
# 对比两次测试的CSV结果
```

### 场景4：我想长期跟踪模型优化效果
**推荐工具**：`speed_test.py` 定期运行
```bash
# 每周运行一次
python speed_test.py
# 历史数据保存在 speed_test_results/ 目录
```

### 场景5：我不知道该用哪个工具
**推荐工具**：`test_menu.py`
```bash
python test_menu.py
# 通过菜单选择需要的功能
```

## 📊 支持的模型

该工具包可以**动态检测**并测试本地安装的所有 LLM 模型（自动过滤掉 embedding 模型）：

- ✅ 动态模型检测：自动发现本地已安装的模型
- ✅ 智能过滤：自动排除 embedding 模型（bge, bert等）
- ✅ 支持所有 GGUF 格式的模型

**当前检测到的模型**:
- Qwen 系列 (qwen3, qwen2.5-coder)
- Gemma 系列 (gemma3, gemma2)
- DeepSeek 系列 (deepseek-r1)
- GPT-OSS 系列
- 以及其他本地安装的 LLM 模型

## 🧭 使用指南

1. 确保 Ollama 正在本地运行：
```bash
ollama serve
```
2. 拉取并安装需要测试的模型：
```bash
ollama list
ollama pull <model_name>
```
3. 运行测试与分析（见"快速开始"章节）。

## 📊 工具功能对比

| 功能 | speed_test.py | interactive_test.py |
|------|---------------|---------------------|
| 模型速度测试 | ✓✓✓ 专业 | - |
| 交互式对比 | - | ✓✓✓ 核心功能 |
| 并行执行 | - | ✓✓✓ 核心功能 |
| 资源监控 | ✓✓✓ GPU+CPU | - |
| 标准化问题集 | ✓✓✓ 16个标准问题 | - |
| 准确速度公式 | - | ✓✓✓ 总Token÷总时间 | - |
| 详细报表 | JSON | CSV+JSON+表格 | Markdown |
| 横向对比展示 | - | - | ✓✓✓ 核心功能 |
| 测试时间 | 30-60分钟 | 15-30分钟 | 0.5-2分钟 |

## 🧩 指标解释与示例

### 关键指标

- **tokens_per_second**：生成速度，数值越大越快
- **ttft_ms_warm / cold**：首字延迟（热/冷启动）
- **total_ms_warm / cold**：完成任务的总时间（热/冷启动）
- **memory_usage_mb**：内存使用量
- **response_length**：生成文本长度

### 交互式测试输出示例
```
================================================================================
MODEL RESPONSES COMPARISON
================================================================================

Prompt: 请解释什么是量子计算？

================================================================================

[1] Model: gemma3:4b
    Duration: 3.45s
    Tokens: 128
    Speed: 37.10 tokens/s
    Response:
    ----------------------------------------------------------------------------
    量子计算是一种利用量子力学原理进行计算的新型计算方式...
    ----------------------------------------------------------------------------

[2] Model: qwen2.5:14b
    Duration: 8.23s
    Tokens: 156
    Speed: 18.95 tokens/s
    Response:
    ----------------------------------------------------------------------------
    量子计算是基于量子比特的计算技术...
    ----------------------------------------------------------------------------

================================================================================
SUMMARY STATISTICS
================================================================================

Total models tested: 12
Successful: 11
Failed: 1

Speed statistics:
  Fastest: 45.23 tokens/s (gemma3:4b)
  Slowest: 12.34 tokens/s (qwen2.5:14b)
  Average: 28.56 tokens/s
```

### 速度测试输出示例
```
================================================================================
Testing model: gemma3:4b
================================================================================

[SIMPLE] Testing 5 prompts...
  1/5: 你好，你是谁？...
    ✓ Tokens: 45, Duration: 1.23s, Speed: 36.59 tokens/s

[MEDIUM] Testing 5 prompts...
  1/5: 请用Python编写一个合并两个有序链表的函数...
    ✓ Tokens: 156, Duration: 4.12s, Speed: 37.86 tokens/s

[COMPLEX] Testing 5 prompts...
  1/5: 撰写200字分析大模型落地应用的挑战与解决方案...
    ✓ Tokens: 312, Duration: 8.45s, Speed: 36.92 tokens/s

================================================================================
Summary for gemma3:4b:
  Total output tokens: 1245
  Total duration: 34.56s
  Average speed: 36.02 tokens/s
================================================================================
```

## 🎯 选择建议（参考）

1. **日常使用**：优先 tokens_per_second 较高且 memory_usage_mb 适中的模型
2. **高质量需求**：可考虑 13B 以上参数规模的模型
3. **速度优先**：选择 7B 级别的轻量模型
4. **内存受限**：选择量化或轻量优化模型

## 🛠️ 故障排除

如果遇到问题，请：
1. 确保 Ollama 服务正在运行：`ollama serve`
2. 检查模型是否已安装：`ollama list`
3. 常见问题与建议：
   - **无法连接到 Ollama 服务**：确认服务已启动并检查端口（默认 11434）
   - **模型未找到**：先执行 `ollama pull <model_name>`，再 `ollama list` 验证
   - **内存不足**：关闭占用内存的程序，改用更小或量化模型
   - **测试超时**：个别模型需要更长时间，可适当提高脚本超时阈值
   - **GPU监控不可用**：如果没有NVIDIA GPU或未安装nvidia-smi，会自动跳过GPU监控
   - **并行测试速度慢**：减少并行工作线程数 `--max-workers 3`

## ⚙️ 高级配置提示

- 在 `analyzer.py` 中可通过 `--weights` 调整评分权重以匹配业务侧重
- 可增补自定义测试用例，调整生成长度（例如 `num_predict`）、温度 `temperature`、超时等
- 编辑 `test_prompts.txt` 可以自定义速度测试的问题集
- 所有测试结果都保存在对应的结果目录中，便于长期跟踪和分析

## 🚀 快速命令参考

```bash
# 测试工具菜单（推荐新手）
python test_menu.py

# 交互式测试（推荐用于模型对比）
python interactive_test.py

# 速度测试（推荐用于性能评估）
python speed_test.py

# 分析工具
python analyzer.py

# 完整流程
python complete_test.py
```

---
