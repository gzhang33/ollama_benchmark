#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct Ollama Tester Module
使用非HTTP方法直接调用ollama，避免API开销
"""

import json
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional


class DirectOllamaTester:
    """直接调用ollama命令的测试器"""
    
    def __init__(self, ollama_path: str = "ollama", timeout: int = 120, enable_caching: bool = True):
        """
        初始化优化的直接ollama测试器

        Args:
            ollama_path: ollama命令路径
            timeout: 超时时间
            enable_caching: 是否启用模型缓存
        """
        self.ollama_path = ollama_path
        self.timeout = timeout
        self.enable_caching = enable_caching
        self._model_cache = {}  # 模型缓存：存储已验证的模型
        self._warmup_models = set()  # 预热模型集合

    def warmup_model(self, model_name: str) -> bool:
        """
        预热模型，提前验证模型可用性

        Args:
            model_name: 模型名称

        Returns:
            是否预热成功
        """
        if model_name in self._warmup_models:
            return True

        try:
            # 使用简单的验证命令检查模型是否存在
            cmd = [self.ollama_path, "show", model_name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                self._warmup_models.add(model_name)
                print(f"[DEBUG] Model {model_name} warmed up successfully")
                return True
            else:
                print(f"[DEBUG] Model {model_name} not available: {result.stderr}")
                return False
        except Exception as e:
            print(f"[DEBUG] Model warmup failed for {model_name}: {e}")
            return False

    def test_output_generation_time(
        self,
        model_name: str,
        prompt: str,
        num_predict: int = 400
    ) -> Dict[str, Any]:
        """
        使用subprocess直接调用ollama命令测试输出生成时间
        
        Args:
            model_name: 模型名称
            prompt: 输入提示
            num_predict: 最大输出token数
            
        Returns:
            包含输出生成时间等详细信息的字典
        """
        # 构建ollama命令
        cmd = [
            self.ollama_path, "run",
            model_name,
            prompt,
            "--verbose"
        ]
        
        # 记录开始时间
        start_time = time.perf_counter_ns()

        try:
            # 预热模型（如果启用了缓存）
            if self.enable_caching:
                self.warmup_model(model_name)

            # 执行命令并捕获输出，设置编码避免Windows编码问题
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=self.timeout,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'LANG': 'C.UTF-8'},
                # 添加进程优化参数
                stdin=subprocess.DEVNULL,
                preexec_fn=None  # 在Unix上可以设置进程组
            )
            
            end_time = time.perf_counter_ns()
            total_duration = (end_time - start_time) / 1e9
            
            if result.returncode != 0:
                return {
                    "model": model_name,
                    "success": False,
                    "error": f"Command failed: {result.stderr}",
                    "eval_count": 0,
                    "eval_duration": 0,
                    "eval_rate": 0
                }
            
            # 解析输出 - 时间统计信息在stderr中
            output = result.stderr  # 修正：时间统计在stderr中
            metrics = self._parse_verbose_output(output)
            
            # 提取响应内容（从stdout中获取）
            response_content = result.stdout.strip()
            
            return {
                "model": model_name,
                "success": True,
                "response": response_content,
                "response_length": len(response_content),
                
                # 主要指标
                "eval_count": metrics.get("eval_count", 0),
                "eval_duration": metrics.get("eval_duration", 0),
                "eval_rate": metrics.get("eval_rate", 0),
                
                # 详细时间信息
                "prompt_eval_count": metrics.get("prompt_eval_count", 0),
                "prompt_eval_duration": metrics.get("prompt_eval_duration", 0),
                "load_duration": metrics.get("load_duration", 0),
                "total_duration": metrics.get("total_duration", total_duration),
                
                # 兼容性字段
                "tokens": metrics.get("eval_count", 0),
                "duration": metrics.get("eval_duration", 0),
                "tokens_per_second": metrics.get("eval_rate", 0),
                
                # 原始输出用于调试
                "raw_output": output
            }
            
        except subprocess.TimeoutExpired:
            return {
                "model": model_name,
                "success": False,
                "error": "Timeout",
                "eval_count": 0,
                "eval_duration": 0,
                "eval_rate": 0
            }
        except Exception as e:
            return {
                "model": model_name,
                "success": False,
                "error": str(e),
                "eval_count": 0,
                "eval_duration": 0,
                "eval_rate": 0
            }
    
    def _parse_verbose_output(self, output: str) -> Dict[str, float]:
        """解析ollama --verbose的输出"""
        metrics = {}

        # 清理ANSI控制字符（更彻底的清理）
        clean_output = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', output)
        clean_output = re.sub(r'\x1b\[[0-9;]*[mGKH]', '', clean_output)
        clean_output = re.sub(r'\x1b\][0-9;]*', '', clean_output)
        clean_output = re.sub(r'\x1b[0-9;]*', '', clean_output)
        # 清理残留的控制字符序列
        clean_output = re.sub(r'\[\?[0-9;]*[hl]', '', clean_output)
        clean_output = re.sub(r'\[[0-9;]*[mGKH]', '', clean_output)
        # 清理其他可能的控制字符
        clean_output = re.sub(r'[^\x20-\x7E\n\r\t]', '', clean_output)

        # 使用正则表达式直接从整个输出中提取，注意匹配最末尾的值
        patterns = {
            "total_duration": r"total duration:\s+([\d.]+)s",
            "load_duration": r"load duration:\s+([\d.]+)ms",
            "prompt_eval_count": r"prompt eval count:\s+(\d+)\s+token\(s\)",
            "prompt_eval_duration": r"prompt eval duration:\s+([\d.]+)ms",
            "eval_count": r"^eval count:\s+(\d+)\s+token\(s\)",  # 匹配行开头的eval count（不是prompt eval）
            "eval_duration": r"^eval duration:\s+([\d.]+)ms",     # 匹配行开头的eval duration（不是prompt eval）
            "eval_rate": r"^eval rate:\s+([\d.]+)\s+tokens/s"   # 匹配行开头的eval rate（不是prompt eval）
        }

        # 记录哪些模式匹配到了毫秒格式
        ms_patterns = {"load_duration", "prompt_eval_duration", "eval_duration"}

        for key, pattern in patterns.items():
            match = re.search(pattern, clean_output, re.MULTILINE)
            if match:
                value = match.group(1)
                if key in ms_patterns:
                    # 将毫秒转换为秒
                    metrics[key] = float(value) / 1000.0
                elif "duration" in key and "s" in pattern:
                    # 已经是秒
                    metrics[key] = float(value)
                else:
                    # 其他数值直接转换
                    metrics[key] = float(value) if '.' in value else int(value)

        # 如果没有找到eval_rate，手动计算
        if "eval_rate" not in metrics and "eval_count" in metrics and "eval_duration" in metrics:
            if metrics["eval_duration"] > 0:
                metrics["eval_rate"] = metrics["eval_count"] / metrics["eval_duration"]

        return metrics
    
    def _extract_response_content(self, output: str) -> str:
        """提取响应内容，去掉时间统计部分"""
        # 找到最后一个时间统计行的位置
        lines = output.split('\n')
        
        # 查找包含时间统计的行
        stats_start_idx = -1
        for i, line in enumerate(lines):
            if "total duration:" in line.lower():
                stats_start_idx = i
                break
        
        # 如果找到统计信息，只取之前的内容
        if stats_start_idx > 0:
            content_lines = lines[:stats_start_idx]
        else:
            content_lines = lines
        
        # 清理输出，去掉控制字符
        content = '\n'.join(content_lines)
        # 移除ANSI控制字符（更全面的清理）
        content = re.sub(r'\x1b\[[0-9;]*[mGKH]', '', content)
        content = re.sub(r'\x1b\][0-9;]*', '', content)
        content = re.sub(r'\x1b[0-9;]*', '', content)
        
        return content.strip()

    def test_output_generation_time_optimized(
        self,
        model_name: str,
        prompt: str,
        num_predict: int = 400,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        优化的输出生成时间测试，针对GUI应用的高并发场景

        Args:
            model_name: 模型名称
            prompt: 测试提示
            num_predict: 最大输出token数
            use_cache: 是否使用模型缓存

        Returns:
            测试结果字典
        """
        # 检查缓存
        cache_key = f"{model_name}_{hash(prompt) % 10000}"
        if use_cache and cache_key in self._model_cache:
            cached_result = self._model_cache[cache_key]
            print(f"[DEBUG] Using cached result for {model_name}")
            return cached_result

        # 预热模型
        if use_cache:
            self.warmup_model(model_name)

        # 使用优化的命令参数（移除不支持的参数）
        cmd = [
            self.ollama_path, "run", model_name,
            prompt,
            "--verbose"
        ]

        start_time = time.perf_counter_ns()

        try:
            # 使用优化的子进程参数，处理编码问题
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.timeout,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'LANG': 'C.UTF-8'},
                stdin=subprocess.DEVNULL,
                # 使用encoding=None来获取原始字节，避免编码错误
            )

            # 手动处理字节解码，忽略无法解码的字符
            if result.stderr:
                try:
                    result.stderr = result.stderr.decode('utf-8', errors='ignore')
                except:
                    result.stderr = ""
            if result.stdout:
                try:
                    result.stdout = result.stdout.decode('utf-8', errors='ignore')
                except:
                    result.stdout = ""

            end_time = time.perf_counter_ns()
            total_duration = (end_time - start_time) / 1e9

            if result.returncode != 0:
                return {
                    "model": model_name,
                    "success": False,
                    "error": f"Command failed: {result.stderr}",
                    "eval_count": 0,
                    "eval_duration": 0,
                    "eval_rate": 0
                }

            # 解析输出
            output = result.stderr
            metrics = self._parse_verbose_output(output)
            response_content = result.stdout.strip()

            result_dict = {
                "model": model_name,
                "success": True,
                "response": response_content,
                "response_length": len(response_content),
                "eval_count": metrics.get("eval_count", 0),
                "eval_duration": metrics.get("eval_duration", 0),
                "eval_rate": metrics.get("eval_rate", 0),
                "prompt_eval_count": metrics.get("prompt_eval_count", 0),
                "prompt_eval_duration": metrics.get("prompt_eval_duration", 0),
                "load_duration": metrics.get("load_duration", 0),
                "total_duration": metrics.get("total_duration", total_duration),
                "tokens": metrics.get("eval_count", 0),
                "duration": metrics.get("eval_duration", 0),
                "tokens_per_second": metrics.get("eval_rate", 0),
                "raw_output": output
            }

            # 缓存结果
            if use_cache:
                self._model_cache[cache_key] = result_dict

            return result_dict

        except subprocess.TimeoutExpired:
            return {
                "model": model_name,
                "success": False,
                "error": "Timeout",
                "eval_count": 0,
                "eval_duration": 0,
                "eval_rate": 0
            }
        except Exception as e:
            return {
                "model": model_name,
                "success": False,
                "error": str(e),
                "eval_count": 0,
                "eval_duration": 0,
                "eval_rate": 0
            }

    def batch_test_output_generation(
        self,
        model_name: str,
        prompts: List[str],
        num_predict: int = 400
    ) -> Dict[str, Any]:
        """
        批量测试输出生成时间
        
        Args:
            model_name: 模型名称
            prompts: 测试提示列表
            num_predict: 最大输出token数
            
        Returns:
            批量测试结果
        """
        results: List[Dict[str, Any]] = []
        total_eval_tokens = 0
        total_eval_duration = 0.0
        successful_tests = 0
        
        for prompt in prompts:
            result = self.test_output_generation_time(model_name, prompt, num_predict)
            
            if result["success"]:
                successful_tests += 1
                total_eval_tokens += result["eval_count"]
                total_eval_duration += result["eval_duration"]
                
                results.append({
                    "model": model_name,
                    "prompt": prompt,
                    "eval_count": result["eval_count"],
                    "eval_duration": result["eval_duration"],
                    "eval_rate": result["eval_rate"],
                    "response_preview": result["response"][:200]
                })
        
        # 计算平均输出生成速度
        avg_eval_rate = total_eval_tokens / total_eval_duration if total_eval_duration > 0 else 0
        success_rate = successful_tests / len(prompts) if prompts else 0
        
        return {
            "model": model_name,
            "total_eval_tokens": total_eval_tokens,
            "total_eval_duration": total_eval_duration,
            "average_eval_rate": avg_eval_rate,
            "success_rate": success_rate,
            "test_count": len(results),
            "details": results,
            
            # 兼容性字段
            "total_tokens": total_eval_tokens,
            "total_duration": total_eval_duration,
            "average_speed": avg_eval_rate
        }


class SocketOllamaTester:
    """使用socket直接与ollama通信的测试器"""
    
    def __init__(self, host: str = "localhost", port: int = 11434):
        """
        初始化socket测试器
        
        Args:
            host: ollama服务主机
            port: ollama服务端口
        """
        self.host = host
        self.port = port
    
    def test_output_generation_time(
        self,
        model_name: str,
        prompt: str,
        num_predict: int = 400
    ) -> Dict[str, Any]:
        """
        使用socket直接与ollama通信
        
        Args:
            model_name: 模型名称
            prompt: 输入提示
            num_predict: 最大输出token数
            
        Returns:
            测试结果
        """
        import socket
        import json
        
        # 构建请求
        request_data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": num_predict
            }
        }
        
        start_time = time.perf_counter_ns()
        
        try:
            # 创建socket连接
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.host, self.port))
                sock.settimeout(self.timeout)
                
                # 发送HTTP请求
                http_request = (
                    f"POST /api/generate HTTP/1.1\r\n"
                    f"Host: {self.host}:{self.port}\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(json.dumps(request_data))}\r\n"
                    f"\r\n"
                    f"{json.dumps(request_data)}"
                )
                
                sock.send(http_request.encode())
                
                # 接收响应
                response = b""
                while True:
                    try:
                        data = sock.recv(4096)
                        if not data:
                            break
                        response += data
                        
                        # 检查是否接收完整
                        if b"\r\n\r\n" in response:
                            headers, body = response.split(b"\r\n\r\n", 1)
                            if b"Content-Length:" in headers:
                                # 解析Content-Length
                                for line in headers.split(b"\r\n"):
                                    if line.startswith(b"Content-Length:"):
                                        content_length = int(line.split(b":", 1)[1].strip())
                                        if len(body) >= content_length:
                                            break
                            else:
                                break
                    except socket.timeout:
                        break
                
                end_time = time.perf_counter_ns()
                total_duration = (end_time - start_time) / 1e9
                
                # 解析响应
                if b"HTTP/1.1 200" in response:
                    # 提取JSON响应
                    json_start = response.find(b"{")
                    if json_start != -1:
                        json_data = json.loads(response[json_start:].decode())
                        
                        # 提取指标
                        eval_count = json_data.get("eval_count", 0)
                        eval_duration_ns = json_data.get("eval_duration", 0)
                        eval_duration = eval_duration_ns / 1e9 if eval_duration_ns > 0 else 0
                        eval_rate = eval_count / eval_duration if eval_duration > 0 else 0
                        
                        return {
                            "model": model_name,
                            "success": True,
                            "response": json_data.get("response", ""),
                            "eval_count": eval_count,
                            "eval_duration": eval_duration,
                            "eval_rate": eval_rate,
                            "prompt_eval_count": json_data.get("prompt_eval_count", 0),
                            "prompt_eval_duration": json_data.get("prompt_eval_duration", 0) / 1e9,
                            "load_duration": json_data.get("load_duration", 0) / 1e9,
                            "total_duration": total_duration,
                            
                            # 兼容性字段
                            "tokens": eval_count,
                            "duration": eval_duration,
                            "tokens_per_second": eval_rate
                        }
                
                return {
                    "model": model_name,
                    "success": False,
                    "error": "Invalid response",
                    "eval_count": 0,
                    "eval_duration": 0,
                    "eval_rate": 0
                }
                
        except Exception as e:
            return {
                "model": model_name,
                "success": False,
                "error": str(e),
                "eval_count": 0,
                "eval_duration": 0,
                "eval_rate": 0
            }
