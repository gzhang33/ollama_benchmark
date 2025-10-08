# HTTP Configuration Guide
# HTTP配置指南

## Environment Variables
## 环境变量

The unified HTTP client supports the following environment variables for configuration:

统一HTTP客户端支持以下环境变量进行配置：

### Connection Pool Settings
### 连接池设置

```bash
# Maximum number of keep-alive connections
# 最大保持连接数
HTTP_MAX_KEEPALIVE=20

# Maximum number of total connections  
# 最大总连接数
HTTP_MAX_CONNECTIONS=100

# Keep-alive connection expiry time in seconds
# 保持连接过期时间（秒）
HTTP_KEEPALIVE_EXPIRY=30.0
```

### Timeout Settings
### 超时设置

```bash
# HTTP request timeout in seconds
# HTTP请求超时时间（秒）
HTTP_TIMEOUT=60
```

### Protocol Settings
### 协议设置

```bash
# Enable HTTP/2 support
# 启用HTTP/2支持
HTTP_ENABLE_HTTP2=true
```

## Usage Examples
## 使用示例

### Basic Usage
### 基本使用

```python
from src.unified_ollama_client import UnifiedOllamaClient

# Create client with default configuration
# 使用默认配置创建客户端
client = UnifiedOllamaClient()

# Get available models
# 获取可用模型
models = client.get_available_models()

# Query a model
# 查询模型
result = client.query_model("llama2", "Hello, world!")

# Close client
# 关闭客户端
client.close_sync()
```

### Async Usage
### 异步使用

```python
import asyncio
from src.unified_ollama_client import UnifiedOllamaClient

async def main():
    client = UnifiedOllamaClient()
    
    try:
        # Get models asynchronously
        # 异步获取模型
        models = await client.get_available_models_async()
        
        # Query model with streaming
        # 流式查询模型
        result = await client.query_model_stream(
            "llama2", 
            "Hello, world!",
            on_chunk_callback=lambda chunk: print(chunk.content)
        )
        
    finally:
        # Close async client
        # 关闭异步客户端
        await client.close_async()

# Run async function
# 运行异步函数
asyncio.run(main())
```

### Custom Configuration
### 自定义配置

```python
from src.unified_ollama_client import UnifiedOllamaClient, HTTPConfig

# Create custom configuration
# 创建自定义配置
config = HTTPConfig()
config.max_keepalive_connections = 50
config.max_connections = 200
config.timeout = 120

# Create client with custom config
# 使用自定义配置创建客户端
client = UnifiedOllamaClient("http://localhost:11434", config)
```

## Performance Optimization
## 性能优化

### Connection Pool Tuning
### 连接池调优

For high-concurrency scenarios, increase the connection pool settings:

对于高并发场景，增加连接池设置：

```bash
HTTP_MAX_KEEPALIVE=50
HTTP_MAX_CONNECTIONS=200
HTTP_KEEPALIVE_EXPIRY=60.0
```

### Timeout Tuning
### 超时调优

For long-running queries, increase the timeout:

对于长时间运行的查询，增加超时时间：

```bash
HTTP_TIMEOUT=300
```

## Migration from requests
## 从requests迁移

The unified client maintains backward compatibility with the old requests-based client:

统一客户端保持与旧requests客户端的向后兼容性：

```python
# Old code (still works)
# 旧代码（仍然有效）
from src.ollama_client import OllamaClient
client = OllamaClient()
models = client.get_available_models()

# New code (recommended)
# 新代码（推荐）
from src.unified_ollama_client import UnifiedOllamaClient
client = UnifiedOllamaClient()
models = client.get_available_models()
```

## Benefits of the Unified Client
## 统一客户端的优势

1. **Single Dependency**: Only requires httpx, no requests
   **单一依赖**：只需要httpx，不需要requests

2. **Better Performance**: HTTP/2 support, connection pooling
   **更好的性能**：HTTP/2支持，连接池

3. **Configurable**: Environment variable configuration
   **可配置**：环境变量配置

4. **Unified API**: Same interface for sync and async operations
   **统一API**：同步和异步操作使用相同接口

5. **Memory Efficient**: Better resource management
   **内存高效**：更好的资源管理
