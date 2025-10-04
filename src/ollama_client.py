#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Client Module
Shared Ollama client module providing basic model querying and listing functionality
"""

import time
from typing import Any, Dict, List, Optional

import requests


class OllamaClient:
    """Ollama client class, encapsulates model querying and listing functionality"""
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        """
        Initialize Ollama client
        
        Args:
            base_url: Base URL of Ollama service
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
    
    def get_available_models(self) -> List[str]:
        """
        Get available LLM model list
        
        Returns:
            List of model names, filtered to exclude embedding models
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            payload = response.json()
            
            models: List[str] = []
            for item in payload.get("models", []):
                name = item["name"]
                # Filter out embedding models
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
        """
        Query a single model
        
        Args:
            model_name: Model name
            prompt: Prompt text
            temperature: Temperature parameter
            num_predict: Maximum prediction tokens
            
        Returns:
            Dictionary containing query results
        """
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
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            data = response.json()
            response_text = data.get("response", "")
            
            # Token counting
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
            
        except requests.Timeout:
            return {
                "model": model_name,
                "success": False,
                "error": "Timeout",
                "duration": self.timeout
            }
        except Exception as e:
            return {
                "model": model_name,
                "success": False,
                "error": str(e),
                "duration": 0
            }


# Convenience functions for backward compatibility
def get_available_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Get available model list (backward compatibility function)"""
    client = OllamaClient(base_url)
    return client.get_available_models()


def query_model(
    base_url: str,
    model_name: str,
    prompt: str,
    timeout: int = 120
) -> Dict[str, Any]:
    """Query a single model (backward compatibility function)"""
    client = OllamaClient(base_url, timeout)
    return client.query_model(model_name, prompt)

