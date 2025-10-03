# Ollama Test Project Modularization Rules

## Project Overview

This project has been refactored into a modular architecture, extracting common code from `speed_test.py` and `app.py` into independent modules to improve code reusability and maintainability. Please refer to this document for future updates.

## Module Structure

### Core Modules

#### 1. `ollama_client.py` - Ollama Client Module
**Function**: Provides basic Ollama API interaction functionality

**Main Classes and Functions**:
- `OllamaClient`: 主要的客户端类
  - `get_available_models()`: 获取可用模型列表
  - `query_model()`: 查询单个模型
- 便捷函数（向后兼容）:
  - `get_available_models(base_url)`: 独立函数版本
  - `query_model(base_url, model_name, prompt, timeout)`: 独立函数版本

**依赖**: `requests`

#### 2. `ollama_utils.py` - 工具模块
**功能**: 提供数据处理、文件操作等实用功能

**主要函数**:
- `get_test_prompts()`: 返回内置测试提示词
- `safe_float(value)`: 安全转换为 float
- `export_results_to_csv()`: 导出结果到 CSV
- `generate_summary_table()`: 生成摘要表格
- `save_json_results()`: 保存 JSON 结果
- `create_timestamp()`: 创建时间戳
- `ensure_output_dir()`: 确保输出目录存在

**依赖**: `pandas`

#### 3. `resource_monitor.py` - 资源监控模块
**功能**: 提供 GPU 和 CPU 使用率监控

**主要类和函数**:
- `ResourceMonitor`: 资源监控器类
  - `get_snapshot()`: 获取资源快照
  - `get_gpu_util()`: 获取 GPU 使用率
  - `get_cpu_util()`: 获取 CPU 使用率
- 独立函数:
  - `get_gpu_usage()`: 获取 GPU 使用情况
  - `get_cpu_usage()`: 获取 CPU 使用率
  - `extract_gpu_util()`: 提取 GPU 使用率

**依赖**: `psutil`, `nvidia-smi` (可选)

### 应用模块

#### 4. `speed_test.py` - 速度测试工具
**功能**: 批量测试多个模型的性能

**重构内容**:
- 移除了重复的工具函数
- 使用 `OllamaClient` 进行模型查询
- 使用 `ollama_utils` 进行数据处理
- 使用 `ResourceMonitor` 进行资源监控

**主要函数**:
- `test_model_with_prompt()`: 测试单个提示词
- `run_speed_test()`: 运行速度测试
- `main()`: 主函数

#### 5. `app.py` - 交互式测试工具
**功能**: 提供 GUI 界面进行模型对比

**重构内容**:
- 移除了重复的模型查询函数
- 使用 `OllamaClient` 进行模型查询
- 使用 `ollama_utils` 进行时间戳生成

**主要类**:
- `ModelComparisonGUI`: 主 GUI 应用类
- 各种 UI 组件类

## 代码复用规则

### 1. 导入规则
```python
# 标准库
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 第三方库
import requests  # 仅在需要时导入

# 项目模块
from ollama_client import OllamaClient
from ollama_utils import get_test_prompts, create_timestamp
from resource_monitor import ResourceMonitor
```

### 2. 客户端使用规则
```python
# 创建客户端实例
client = OllamaClient(base_url="http://localhost:11434", timeout=120)

# 获取模型列表
models = client.get_available_models()

# 查询模型
result = client.query_model(model_name, prompt, temperature=0.7, num_predict=512)
```

### 3. 资源监控使用规则
```python
# 创建监控器
monitor = ResourceMonitor(enable_gpu=True, enable_cpu=True)

# 获取快照
snapshot = monitor.get_snapshot()

# 获取特定指标
gpu_util = monitor.get_gpu_util()
cpu_util = monitor.get_cpu_util()
```

### 4. 工具函数使用规则
```python
# 获取测试提示词
prompts = get_test_prompts()

# 创建时间戳
timestamp = create_timestamp()

# 确保输出目录
ensure_output_dir(Path("output"))

# 导出结果
export_results_to_csv(results, output_file)
generate_summary_table(results, summary_file)
save_json_results(results, json_file, timestamp, total_models)
```

## 依赖管理

### 必需依赖
```txt
requests>=2.25.0
pandas>=1.3.0
```

### 可选依赖
```txt
psutil>=5.8.0  # 用于 CPU 监控
tqdm>=4.62.0   # 用于进度条显示
```

### 系统依赖
- `nvidia-smi`: 用于 GPU 监控（可选）

## 向后兼容性

为了保持向后兼容，模块提供了独立函数版本：
- `get_available_models(base_url)` 
- `query_model(base_url, model_name, prompt, timeout)`

这些函数内部使用 `OllamaClient` 类，确保现有代码无需修改即可使用。

## 扩展指南

### 添加新的工具函数
1. 在 `ollama_utils.py` 中添加函数
2. 确保函数有适当的类型提示和文档字符串
3. 在需要的地方导入使用

### 添加新的监控功能
1. 在 `resource_monitor.py` 中添加功能
2. 更新 `ResourceMonitor` 类或添加新的独立函数
3. 确保错误处理完善

### 添加新的客户端功能
1. 在 `ollama_client.py` 中扩展 `OllamaClient` 类
2. 提供向后兼容的独立函数版本
3. 更新文档和类型提示

## 测试和维护

### 单元测试
每个模块都应该有对应的测试文件：
- `test_ollama_client.py`
- `test_ollama_utils.py` 
- `test_resource_monitor.py`

### 集成测试
- 确保模块间的集成正常工作
- 测试向后兼容性
- 验证性能没有回归

### 维护注意事项
1. 保持模块间的松耦合
2. 避免循环依赖
3. 及时更新文档
4. 保持向后兼容性
5. 遵循单一职责原则

## 版本历史

- **v1.0**: 初始模块化重构
  - 提取共同代码到独立模块
  - 保持向后兼容性
  - 建立模块化架构

## 贡献指南

1. 遵循现有的代码风格
2. 添加适当的类型提示
3. 更新相关文档
4. 确保向后兼容性
5. 添加单元测试

## 联系方式

如有问题或建议，请通过项目仓库提交 Issue 或 Pull Request。
