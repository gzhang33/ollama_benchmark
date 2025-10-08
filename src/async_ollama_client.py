#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async Ollama Client with Connection Pool and Streaming Support
异步Ollama客户端 - 支持连接池和流式响应
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, AsyncGenerator
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


class AsyncOllamaClient:
    """异步Ollama客户端，支持连接池和流式响应"""
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 60):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        # 连接池配置
        self.client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._ensure_client()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    async def _ensure_client(self):
        """确保客户端已初始化"""
        if self.client is None:
            # 配置连接池参数
            limits = httpx.Limits(
                max_keepalive_connections=20,  # 保持20个长连接
                max_connections=100,           # 最大100个连接
                keepalive_expiry=30.0          # 连接保持30秒
            )
            
            # 启用HTTP/2支持
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=limits,
                http2=True  # 启用HTTP/2支持
            )
    
    async def close(self):
        """关闭客户端连接"""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    async def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        await self._ensure_client()
        
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
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
    
    async def query_model_stream(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        num_predict: int = 512,
        on_chunk_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        流式查询模型
        
        Args:
            model_name: 模型名称
            prompt: 提示文本
            temperature: 温度参数
            num_predict: 最大预测token数
            on_chunk_callback: 数据块回调函数
            
        Returns:
            包含完整响应的字典
        """
        await self._ensure_client()
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,  # 启用流式响应
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
            async with self.client.stream('POST', url, json=payload) as response:
                response.raise_for_status()
                
                # Set a shorter timeout for streaming to reduce buffering
                response._timeout = httpx.Timeout(5.0, read=30.0)

                try:
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                response_content = data.get('response', '')
                                # 避免编码问题，只显示ASCII字符
                                safe_content = response_content[:50].encode('ascii', 'ignore').decode('ascii')
                                print(f"[DEBUG] Received streaming data: '{safe_content}...' (length: {len(response_content)})")

                                # 处理响应内容
                                if response_content:  # 只有当有内容时才处理
                                    response_text += response_content

                                    # 使用Ollama返回的eval_count，如果没有则估算
                                    current_tokens = 0
                                    if 'eval_count' in data:
                                        current_tokens = data['eval_count']
                                    else:
                                        # 估算token数（按字符数/4估算）
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
                                    # 使用最终估算的token数和实际token数
                                    final_tokens = len(response_text) // 4
                                    # 如果有eval_count，使用实际值，否则使用估算值
                                    if 'eval_count' in data:
                                        final_tokens = data['eval_count']

                                    total_tokens = final_tokens  # 更新total_tokens用于最终计算

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
                    # Ensure the async generator is properly closed
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
    
    async def query_model(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        num_predict: int = 512
    ) -> Dict[str, Any]:
        """
        非流式查询模型（兼容性方法）
        """
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
    
    def __init__(self, client: AsyncOllamaClient):
        self.client = client
        self.task_semaphore = asyncio.Semaphore(1)  # 限制同时运行的模型数量
    
    async def execute_sequential_tasks(
        self,
        models: List[str],
        prompt: str,
        num_predict: int = 512,
        on_model_start: Optional[callable] = None,
        on_model_chunk: Optional[callable] = None,
        on_model_complete: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        顺序执行模型任务（异步调度）
        
        Args:
            models: 模型列表
            prompt: 提示文本
            on_model_start: 模型开始回调
            on_model_chunk: 数据块回调
            on_model_complete: 模型完成回调
            
        Returns:
            所有模型的执行结果
        """
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
