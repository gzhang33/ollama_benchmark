#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Ollama Client Module
统一Ollama客户端 - 使用httpx替代requests，支持同步和异步操作
"""

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Callable
import httpx
from dataclasses import dataclass


@dataclass
class StreamChunk:
    """流式响应数据块"""
    model: str
    content: str
    done: bool = False
    tokens: int = 0
    duration: float = 0.0


class HTTPConfig:
    """HTTP配置管理类"""
    
    def __init__(self):
        self.max_keepalive_connections = int(os.getenv('HTTP_MAX_KEEPALIVE', 20))
        self.max_connections = int(os.getenv('HTTP_MAX_CONNECTIONS', 100))
        self.keepalive_expiry = float(os.getenv('HTTP_KEEPALIVE_EXPIRY', 30.0))
        self.timeout = int(os.getenv('HTTP_TIMEOUT', 60))
        self.enable_http2 = os.getenv('HTTP_ENABLE_HTTP2', 'true').lower() == 'true'
    
    def get_limits(self) -> httpx.Limits:
        """获取httpx连接限制配置"""
        return httpx.Limits(
            max_keepalive_connections=self.max_keepalive_connections,
            max_connections=self.max_connections,
            keepalive_expiry=self.keepalive_expiry
        )
    
    def get_timeout(self) -> httpx.Timeout:
        """获取httpx超时配置"""
        return httpx.Timeout(self.timeout)


class UnifiedOllamaClient:
    """统一Ollama客户端，支持同步和异步操作"""
    
    def __init__(self, base_url: str = "http://localhost:11434", config: Optional[HTTPConfig] = None):
        self.base_url = base_url.rstrip('/')
        self.config = config or HTTPConfig()
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
    
    def _ensure_sync_client(self):
        """确保同步客户端已初始化"""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                timeout=self.config.get_timeout(),
                limits=self.config.get_limits(),
                http2=self.config.enable_http2
            )
    
    async def _ensure_async_client(self):
        """确保异步客户端已初始化"""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self.config.get_timeout(),
                limits=self.config.get_limits(),
                http2=self.config.enable_http2
            )
    
    def close_sync(self):
        """关闭同步客户端"""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
    
    async def close_async(self):
        """关闭异步客户端"""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
    
    def get_available_models(self) -> List[str]:
        """获取可用模型列表（同步）"""
        self._ensure_sync_client()
        
        try:
            response = self._sync_client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
            
            models: List[str] = []
            for item in payload.get("models", []):
                name = item["name"]
                # 过滤掉嵌入模型
                if not any(word in name.lower() for word in ("embedding", "bge", "bert")):
                    models.append(name)
            
            return sorted(models)
        except Exception as exc:
            print(f"Error getting models: {exc}")
            return []
    
    async def get_available_models_async(self) -> List[str]:
        """获取可用模型列表（异步）"""
        await self._ensure_async_client()
        
        try:
            response = await self._async_client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
            
            models: List[str] = []
            for item in payload.get("models", []):
                name = item["name"]
                # 过滤掉嵌入模型
                if not any(word in name.lower() for word in ("embedding", "bge", "bert")):
                    models.append(name)
            
            return sorted(models)
        except Exception as exc:
            print(f"Error getting models: {exc}")
            return []
    
    def query_model(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        num_predict: int = 512
    ) -> Dict[str, Any]:
        """查询单个模型（同步）"""
        self._ensure_sync_client()
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict
            }
        }
        
        start_time = time.perf_counter()
        
        try:
            response = self._sync_client.post(url, json=payload)
            response.raise_for_status()
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            data = response.json()
            response_text = data.get("response", "")
            
            # Token计数
            eval_count = data.get("eval_count", 0)
            if eval_count == 0:
                eval_count = len(response_text.split())
            
            tokens_per_second = eval_count / duration if duration > 0 else 0
            
            return {
                "model": model_name,
                "success": True,
                "response": response_text,
                "duration": duration,
                "tokens": eval_count,
                "tokens_per_second": tokens_per_second
            }
            
        except httpx.TimeoutException:
            return {
                "model": model_name,
                "success": False,
                "error": "Timeout",
                "duration": self.config.timeout
            }
        except Exception as e:
            return {
                "model": model_name,
                "success": False,
                "error": str(e),
                "duration": 0
            }
    
    async def query_model_stream(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        num_predict: int = 512,
        on_chunk_callback: Optional[Callable[[StreamChunk], Any]] = None
    ) -> Dict[str, Any]:
        """流式查询模型（异步）"""
        await self._ensure_async_client()
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
            }
        }
        
        start_time = time.perf_counter()
        response_text = ""
        total_tokens = 0
        
        try:
            async with self._async_client.stream('POST', url, json=payload) as response:
                response.raise_for_status()
                
                try:
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                response_content = data.get('response', '')
                                
                                # 处理响应内容
                                if response_content:
                                    response_text += response_content
                                    
                                    # 使用Ollama返回的eval_count，如果没有则估算
                                    current_tokens = 0
                                    if 'eval_count' in data:
                                        current_tokens = data['eval_count']
                                    else:
                                        current_tokens = len(response_text) // 4
                                    
                                    # 创建数据块并调用回调
                                    if on_chunk_callback:
                                        chunk = StreamChunk(
                                            model=model_name,
                                            content=response_content,
                                            done=False,
                                            tokens=current_tokens
                                        )
                                        await on_chunk_callback(chunk)
                                
                                # 检查是否完成
                                if data.get('done', False):
                                    final_tokens = len(response_text) // 4
                                    if 'eval_count' in data:
                                        final_tokens = data['eval_count']
                                    
                                    total_tokens = final_tokens
                                    
                                    if on_chunk_callback:
                                        final_chunk = StreamChunk(
                                            model=model_name,
                                            content="",
                                            done=True,
                                            tokens=final_tokens
                                        )
                                        await on_chunk_callback(final_chunk)
                                    break
                                
                            except json.JSONDecodeError:
                                continue
                            except Exception as e:
                                print(f"Error processing stream chunk: {e}")
                                continue
                except Exception as e:
                    print(f"Error in streaming loop: {e}")
                finally:
                    try:
                        await response.aclose()
                    except Exception:
                        pass
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            tokens_per_second = total_tokens / duration if duration > 0 else 0
            
            return {
                "model": model_name,
                "success": True,
                "response": response_text,
                "duration": duration,
                "tokens": total_tokens,
                "tokens_per_second": tokens_per_second
            }
            
        except Exception as e:
            return {
                "model": model_name,
                "success": False,
                "error": str(e),
                "duration": 0,
                "tokens": 0,
                "tokens_per_second": 0
            }
    
    async def query_model_async(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        num_predict: int = 512
    ) -> Dict[str, Any]:
        """异步查询模型（非流式）"""
        result = {
            "model": model_name,
            "success": False,
            "response": "",
            "duration": 0,
            "tokens": 0,
            "tokens_per_second": 0
        }
        
        async def collect_chunks(chunk: StreamChunk):
            if not chunk.done:
                result["response"] += chunk.content
            else:
                result["tokens"] = chunk.tokens
        
        return await self.query_model_stream(
            model_name, prompt, temperature, num_predict, collect_chunks
        )


class ModelTaskQueue:
    """模型任务队列管理器"""
    
    def __init__(self, client: UnifiedOllamaClient):
        self.client = client
        self.task_semaphore = asyncio.Semaphore(1)  # 限制同时运行的模型数量
    
    async def execute_sequential_tasks(
        self,
        models: List[str],
        prompt: str,
        num_predict: int = 512,
        on_model_start: Optional[Callable[[str, int], Any]] = None,
        on_model_chunk: Optional[Callable[[StreamChunk], Any]] = None,
        on_model_complete: Optional[Callable[[str, Dict[str, Any], int], Any]] = None
    ) -> List[Dict[str, Any]]:
        """顺序执行模型任务（异步调度）"""
        results = []
        
        for i, model in enumerate(models):
            async with self.task_semaphore:
                # 通知模型开始
                if on_model_start:
                    await on_model_start(model, i)
                
                # 执行模型查询
                result = await self.client.query_model_stream(
                    model,
                    prompt,
                    num_predict=num_predict,
                    on_chunk_callback=on_model_chunk
                )
                
                results.append(result)
                
                # 通知模型完成
                if on_model_complete:
                    await on_model_complete(model, result, i)
        
        return results


# 向后兼容的函数
def get_available_models(base_url: str = "http://localhost:11434") -> List[str]:
    """获取可用模型列表（向后兼容函数）"""
    client = UnifiedOllamaClient(base_url)
    try:
        return client.get_available_models()
    finally:
        client.close_sync()


def query_model(
    base_url: str,
    model_name: str,
    prompt: str,
    timeout: int = 120
) -> Dict[str, Any]:
    """查询单个模型（向后兼容函数）"""
    config = HTTPConfig()
    config.timeout = timeout
    client = UnifiedOllamaClient(base_url, config)
    try:
        return client.query_model(model_name, prompt)
    finally:
        client.close_sync()


# GPU环境变量配置
def setup_gpu_environment():
    """配置GPU加速环境变量"""
    import os
    
    # 基础GPU配置
    os.environ.setdefault('OLLAMA_ENABLE_CUDA', '1')
    os.environ.setdefault('OLLAMA_USE_MLOCK', '1')
    
    # 设置最大GPU内存为15GB，让Ollama自动决定GPU层数
    os.environ['OLLAMA_MAX_GPU_MEMORY'] = '15GB'
    
    print(f"GPU Environment configured:")
    print(f"  OLLAMA_ENABLE_CUDA: {os.environ.get('OLLAMA_ENABLE_CUDA')}")
    print(f"  OLLAMA_MAX_GPU_MEMORY: {os.environ.get('OLLAMA_MAX_GPU_MEMORY')}")
    print(f"  OLLAMA_USE_MLOCK: {os.environ.get('OLLAMA_USE_MLOCK')}")
    print(f"  OLLAMA_GPU_LAYERS: Using Ollama default (not set)")


# 在模块导入时自动配置GPU环境
setup_gpu_environment()
