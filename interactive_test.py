#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Model Comparison Tool with GUI
可视化交互式模型对比工具 - 单界面输入，所有模型并行回答
"""

import concurrent.futures
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog


def get_available_models(base_url: str) -> List[str]:
    """Get available LLM models from Ollama."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        models = []
        for model_info in data.get("models", []):
            model_name = model_info["name"]
            if not any(keyword in model_name.lower() 
                      for keyword in ["embedding", "bge", "bert"]):
                models.append(model_name)
        
        return sorted(models)
    except Exception as e:
        print(f"Error getting models: {e}")
        return []


def query_model(
    base_url: str,
    model_name: str,
    prompt: str,
    timeout: int = 120
) -> Dict[str, Any]:
    """Query a single model and return response with timing."""
    url = f"{base_url}/api/generate"
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 512
        }
    }
    
    start_time = time.perf_counter()
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        data = response.json()
        response_text = data.get("response", "")
        
        # Token count
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
            "duration": timeout
        }
    except Exception as e:
        return {
            "model": model_name,
            "success": False,
            "error": str(e),
            "duration": 0
        }


def save_results_to_file(
    results: List[Dict[str, Any]],
    prompt: str,
    output_dir: Path,
    filename: str = None
) -> None:
    """Save results to a markdown file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interactive_test_{timestamp}.md"
    
    output_file = output_dir / filename
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Interactive Model Comparison\n\n")
        f.write(f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Prompt**: {prompt}\n\n")
        f.write(f"---\n\n")
        
        successful_results = [r for r in results if r["success"]]
        
        for idx, result in enumerate(successful_results, 1):
            f.write(f"## [{idx}] {result['model']}\n\n")
            f.write(f"- **Duration**: {result['duration']:.2f}s\n")
            f.write(f"- **Tokens**: {result['tokens']}\n")
            f.write(f"- **Speed**: {result['tokens_per_second']:.2f} tokens/s\n\n")
            f.write(f"**Response**:\n\n")
            f.write(f"```\n{result['response']}\n```\n\n")
            f.write(f"---\n\n")
        
        # Add summary
        if successful_results:
            speeds = [r['tokens_per_second'] for r in successful_results]
            durations = [r['duration'] for r in successful_results]
            
            f.write(f"## Summary Statistics\n\n")
            f.write(f"- **Total models**: {len(results)}\n")
            f.write(f"- **Successful**: {len(successful_results)}\n")
            f.write(f"- **Failed**: {len([r for r in results if not r['success']])}\n\n")
            f.write(f"### Speed Statistics\n\n")
            f.write(f"- Fastest: {max(speeds):.2f} tokens/s ({successful_results[speeds.index(max(speeds))]['model']})\n")
            f.write(f"- Slowest: {min(speeds):.2f} tokens/s ({successful_results[speeds.index(min(speeds))]['model']})\n")
            f.write(f"- Average: {sum(speeds)/len(speeds):.2f} tokens/s\n\n")


class ModelComparisonGUI:
    """GUI界面类"""
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.output_dir = Path("interactive_test_results")
        self.output_dir.mkdir(exist_ok=True)
        self.models = []
        self.current_results = []
        
        self.setup_gui()
        self.load_models()
    
    def setup_gui(self):
        """设置GUI界面"""
        self.root = tk.Tk()
        self.root.title("模型对比工具 - Interactive Model Comparison")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#f8f9fa')
        
        # 设置现代化样式
        self.setup_styles()
        
        # 创建主容器
        self.main_container = tk.Frame(self.root, bg='#f8f9fa')
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 创建顶部标题栏
        self.create_header()
        
        # 创建主内容区域
        self.create_main_content()
        
        # 创建底部状态栏
        self.create_status_bar()
    
    def setup_styles(self):
        """设置现代化样式"""
        style = ttk.Style()
        
        # 配置主题
        style.theme_use('clam')
        
        # 自定义样式
        style.configure('Title.TLabel', 
                       background='#f8f9fa',
                       foreground='#2c3e50',
                       font=('Arial', 18, 'bold'))
        
        style.configure('Card.TFrame',
                       background='white',
                       relief='flat',
                       borderwidth=1)
        
        style.configure('Primary.TButton',
                       background='#3498db',
                       foreground='white',
                       font=('Arial', 10, 'bold'),
                       padding=(20, 10))
        
        style.configure('Secondary.TButton',
                       background='#95a5a6',
                       foreground='white',
                       font=('Arial', 10),
                       padding=(15, 8))
        
        style.configure('Success.TButton',
                       background='#27ae60',
                       foreground='white',
                       font=('Arial', 10),
                       padding=(15, 8))
        
        style.map('Primary.TButton',
                 background=[('active', '#2980b9')])
        
        style.map('Secondary.TButton',
                 background=[('active', '#7f8c8d')])
        
        style.map('Success.TButton',
                 background=[('active', '#229954')])
    
    def create_header(self):
        """创建顶部标题栏"""
        header_frame = tk.Frame(self.main_container, bg='white', height=80)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # 标题
        title_label = ttk.Label(header_frame, text="🤖 模型对比工具", 
                               style='Title.TLabel')
        title_label.pack(expand=True)
        
        # 副标题
        subtitle_label = ttk.Label(header_frame, 
                                  text="Interactive Model Comparison",
                                  background='white',
                                  foreground='#7f8c8d',
                                  font=('Arial', 12))
        subtitle_label.pack()
    
    def create_main_content(self):
        """创建主内容区域"""
        content_frame = tk.Frame(self.main_container, bg='#f8f9fa')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧面板 - 输入和控制
        self.create_left_panel(content_frame)
        
        # 右侧面板 - 结果显示
        self.create_right_panel(content_frame)
    
    def create_left_panel(self, parent):
        """创建左侧面板"""
        left_panel = tk.Frame(parent, bg='#f8f9fa', width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left_panel.pack_propagate(False)
        
        # 输入区域卡片
        input_card = ttk.Frame(left_panel, style='Card.TFrame', padding=20)
        input_card.pack(fill=tk.X, pady=(0, 20))
        
        # 输入区域标题
        input_title = ttk.Label(input_card, text="📝 输入提示词", 
                               background='white',
                               foreground='#2c3e50',
                               font=('Arial', 12, 'bold'))
        input_title.pack(anchor=tk.W, pady=(0, 10))
        
        # 输入文本框
        self.prompt_text = scrolledtext.ScrolledText(input_card, height=4, wrap=tk.WORD,
                                                   font=('Arial', 11),
                                                   bg='#f8f9fa',
                                                   relief='flat',
                                                   borderwidth=1)
        self.prompt_text.pack(fill=tk.X, pady=(0, 15))
        
        # 按钮区域
        button_frame = tk.Frame(input_card, bg='white')
        button_frame.pack(fill=tk.X)
        
        self.query_button = ttk.Button(button_frame, text="🚀 开始查询", 
                                      command=self.start_query, 
                                      style='Primary.TButton')
        self.query_button.pack(fill=tk.X, pady=(0, 10))
        
        # 次要按钮行
        secondary_frame = tk.Frame(button_frame, bg='white')
        secondary_frame.pack(fill=tk.X)
        
        self.clear_button = ttk.Button(secondary_frame, text="🗑️ 清空", 
                                      command=self.clear_prompt,
                                      style='Secondary.TButton')
        self.clear_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.save_button = ttk.Button(secondary_frame, text="💾 保存", 
                                     command=self.save_results, 
                                     state=tk.DISABLED,
                                     style='Success.TButton')
        self.save_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # 进度区域
        self.create_progress_section(left_panel)
        
        # 模型选择区域
        self.create_model_selection(left_panel)
    
    def create_progress_section(self, parent):
        """创建进度区域"""
        progress_card = ttk.Frame(parent, style='Card.TFrame', padding=20)
        progress_card.pack(fill=tk.X, pady=(0, 20))
        
        # 进度标题
        progress_title = ttk.Label(progress_card, text="⏳ 查询状态", 
                                  background='white',
                                  foreground='#2c3e50',
                                  font=('Arial', 12, 'bold'))
        progress_title.pack(anchor=tk.W, pady=(0, 10))
        
        # 状态标签
        self.progress_var = tk.StringVar(value="准备就绪")
        self.progress_label = ttk.Label(progress_card, 
                                       textvariable=self.progress_var,
                                       background='white',
                                       foreground='#27ae60',
                                       font=('Arial', 10))
        self.progress_label.pack(anchor=tk.W, pady=(0, 10))
        
        # 进度条
        self.progress_bar = ttk.Progressbar(progress_card, mode='indeterminate',
                                          style='TProgressbar')
        self.progress_bar.pack(fill=tk.X)
    
    def create_model_selection(self, parent):
        """创建模型选择区域"""
        model_card = ttk.Frame(parent, style='Card.TFrame', padding=20)
        model_card.pack(fill=tk.BOTH, expand=True)
        
        # 模型选择标题
        model_title = ttk.Label(model_card, text="🎯 模型选择", 
                               background='white',
                               foreground='#2c3e50',
                               font=('Arial', 12, 'bold'))
        model_title.pack(anchor=tk.W, pady=(0, 15))
        
        # 模型统计
        self.model_count_var = tk.StringVar(value="加载中...")
        model_count_label = ttk.Label(model_card, 
                                     textvariable=self.model_count_var,
                                     background='white',
                                     foreground='#7f8c8d',
                                     font=('Arial', 10))
        model_count_label.pack(anchor=tk.W, pady=(0, 10))
        
        # 全选/取消全选按钮
        select_frame = tk.Frame(model_card, bg='white')
        select_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(select_frame, text="全选", 
                  command=self.select_all_models,
                  style='Secondary.TButton').pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(select_frame, text="取消全选", 
                  command=self.deselect_all_models,
                  style='Secondary.TButton').pack(side=tk.LEFT)
        
        # 模型列表容器
        self.models_container = tk.Frame(model_card, bg='white')
        self.models_container.pack(fill=tk.BOTH, expand=True)
        
        self.model_vars = {}
        self.model_checkboxes = {}
    
    def create_right_panel(self, parent):
        """创建右侧面板"""
        right_panel = tk.Frame(parent, bg='#f8f9fa')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 结果显示卡片
        results_card = ttk.Frame(right_panel, style='Card.TFrame', padding=20)
        results_card.pack(fill=tk.BOTH, expand=True)
        
        # 结果标题
        results_title = ttk.Label(results_card, text="📊 模型响应对比", 
                                 background='white',
                                 foreground='#2c3e50',
                                 font=('Arial', 12, 'bold'))
        results_title.pack(anchor=tk.W, pady=(0, 15))
        
        # 创建滚动区域
        canvas = tk.Canvas(results_card, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(results_card, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg='white')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_status_bar(self):
        """创建底部状态栏"""
        status_frame = tk.Frame(self.main_container, bg='#34495e', height=40)
        status_frame.pack(fill=tk.X, pady=(20, 0))
        status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="正在加载模型...")
        self.status_label = ttk.Label(status_frame, 
                                     textvariable=self.status_var,
                                     background='#34495e',
                                     foreground='white',
                                     font=('Arial', 10))
        self.status_label.pack(side=tk.LEFT, padx=15, pady=10)
    
    def load_models(self):
        """加载可用模型"""
        def load_models_thread():
            try:
                self.models = get_available_models(self.base_url)
                if self.models:
                    self.root.after(0, self.update_model_selection)
                    self.root.after(0, lambda: self.status_var.set(f"已加载 {len(self.models)} 个模型"))
                else:
                    self.root.after(0, lambda: self.status_var.set("未找到可用模型"))
                    self.root.after(0, lambda: messagebox.showerror("错误", "未找到可用模型，请检查 Ollama 服务"))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"加载模型失败: {str(e)}"))
                self.root.after(0, lambda: messagebox.showerror("错误", f"加载模型失败: {str(e)}"))
        
        threading.Thread(target=load_models_thread, daemon=True).start()
    
    def update_model_selection(self):
        """更新模型选择界面"""
        # 清除现有的复选框
        for widget in self.models_container.winfo_children():
            widget.destroy()
        
        self.model_vars = {}
        self.model_checkboxes = {}
        
        # 更新模型计数
        self.model_count_var.set(f"已加载 {len(self.models)} 个模型")
        
        # 创建模型复选框网格（垂直布局）
        for i, model in enumerate(self.models):
            var = tk.BooleanVar(value=True)
            self.model_vars[model] = var
            
            # 创建模型项框架
            model_frame = tk.Frame(self.models_container, bg='white')
            model_frame.pack(fill=tk.X, pady=2)
            
            cb = ttk.Checkbutton(model_frame, text=model, variable=var,
                               style='TCheckbutton')
            cb.pack(side=tk.LEFT, anchor=tk.W)
            self.model_checkboxes[model] = cb
    
    def select_all_models(self):
        """全选模型"""
        for var in self.model_vars.values():
            var.set(True)
    
    def deselect_all_models(self):
        """取消全选模型"""
        for var in self.model_vars.values():
            var.set(False)
    
    def get_selected_models(self):
        """获取选中的模型"""
        return [model for model, var in self.model_vars.items() if var.get()]
    
    def clear_prompt(self):
        """清空输入框"""
        self.prompt_text.delete(1.0, tk.END)
    
    def start_query(self):
        """开始查询"""
        prompt = self.prompt_text.get(1.0, tk.END).strip()
        if not prompt:
            messagebox.showwarning("警告", "请输入提示词")
            return
        
        selected_models = self.get_selected_models()
        if not selected_models:
            messagebox.showwarning("警告", "请至少选择一个模型")
            return
        
        # 禁用查询按钮
        self.query_button.config(state=tk.DISABLED)
        self.progress_bar.start()
        self.progress_var.set(f"正在查询 {len(selected_models)} 个模型...")
        
        # 清空之前的结果
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # 在新线程中执行查询
        def query_thread():
            try:
                results = self.parallel_query_all_models_gui(selected_models, prompt)
                self.root.after(0, lambda: self.display_results_gui(results, prompt))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"查询失败: {str(e)}"))
            finally:
                self.root.after(0, self.query_finished)
        
        threading.Thread(target=query_thread, daemon=True).start()
    
    def query_finished(self):
        """查询完成"""
        self.progress_bar.stop()
        self.progress_var.set("查询完成")
        self.query_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
    
    def parallel_query_all_models_gui(self, models: List[str], prompt: str) -> List[Dict[str, Any]]:
        """并行查询所有模型（GUI版本）"""
        results = []
        completed = 0
        
        def update_progress():
            nonlocal completed
            completed += 1
            progress_text = f"正在查询模型... ({completed}/{len(models)})"
            self.root.after(0, lambda: self.progress_var.set(progress_text))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_model = {
                executor.submit(query_model, self.base_url, model, prompt): model 
                for model in models
            }
            
            for future in concurrent.futures.as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    results.append(result)
                    update_progress()
                except Exception as e:
                    results.append({
                        "model": model,
                        "success": False,
                        "error": str(e)
                    })
                    update_progress()
        
        # 按模型名称排序
        results.sort(key=lambda x: x["model"])
        self.current_results = results
        return results
    
    def display_results_gui(self, results: List[Dict[str, Any]], prompt: str):
        """在GUI中显示结果"""
        # 清除之前的结果
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # 显示提示词
        prompt_frame = tk.Frame(self.scrollable_frame, bg='#f8f9fa', relief='flat', 
                               borderwidth=1, highlightbackground='#e9ecef', 
                               highlightthickness=1)
        prompt_frame.pack(fill=tk.X, pady=(0, 15), padx=10)
        
        prompt_content = tk.Frame(prompt_frame, bg='#f8f9fa')
        prompt_content.pack(fill=tk.X, padx=15, pady=15)
        
        prompt_title = tk.Label(prompt_content, text="📝 输入提示词", 
                               bg='#f8f9fa', fg='#2c3e50', 
                               font=('Arial', 12, 'bold'))
        prompt_title.pack(anchor=tk.W, pady=(0, 10))
        
        prompt_label = tk.Label(prompt_content, text=prompt, wraplength=1200,
                               bg='#f8f9fa', fg='#34495e', 
                               font=('Arial', 11), justify=tk.LEFT)
        prompt_label.pack(anchor=tk.W)
        
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        # 显示成功的结果
        for idx, result in enumerate(successful_results, 1):
            self.create_result_frame(result, idx)
        
        # 显示失败的结果
        if failed_results:
            failed_card = tk.Frame(self.scrollable_frame, bg='#ffeaa7', relief='flat', 
                                  borderwidth=1, highlightbackground='#e17055', 
                                  highlightthickness=1)
            failed_card.pack(fill=tk.X, pady=(0, 15), padx=10)
            
            failed_content = tk.Frame(failed_card, bg='#ffeaa7')
            failed_content.pack(fill=tk.X, padx=15, pady=15)
            
            failed_title = tk.Label(failed_content, text="❌ 失败的模型", 
                                   bg='#ffeaa7', fg='#d63031', 
                                   font=('Arial', 12, 'bold'))
            failed_title.pack(anchor=tk.W, pady=(0, 10))
            
            for result in failed_results:
                error_label = tk.Label(failed_content, 
                                      text=f"• {result['model']}: {result.get('error', 'Unknown error')}",
                                      bg='#ffeaa7', fg='#d63031', font=('Arial', 10))
                error_label.pack(anchor=tk.W, pady=2)
        
        # 显示统计信息
        if successful_results:
            self.create_summary_frame(successful_results, len(results))
    
    def create_result_frame(self, result: Dict[str, Any], idx: int):
        """创建单个结果框架"""
        # 创建卡片容器
        card_frame = tk.Frame(self.scrollable_frame, bg='white', relief='flat', 
                             borderwidth=1, highlightbackground='#e9ecef', 
                             highlightthickness=1)
        card_frame.pack(fill=tk.X, pady=(0, 15), padx=10)
        
        # 卡片内容
        content_frame = tk.Frame(card_frame, bg='white')
        content_frame.pack(fill=tk.X, padx=15, pady=15)
        
        # 模型标题
        title_frame = tk.Frame(content_frame, bg='white')
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        model_title = tk.Label(title_frame, text=f"[{idx}] {result['model']}", 
                              bg='white', fg='#2c3e50', font=('Arial', 12, 'bold'))
        model_title.pack(side=tk.LEFT)
        
        # 统计信息
        stats_frame = tk.Frame(content_frame, bg='white')
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 创建统计信息标签
        stats_data = [
            (f"⏱️ {result['duration']:.2f}s", "#f39c12"),
            (f"📊 {result['tokens']} tokens", "#3498db"),
            (f"🚀 {result['tokens_per_second']:.2f}/s", "#27ae60")
        ]
        
        for i, (text, color) in enumerate(stats_data):
            stat_label = tk.Label(stats_frame, text=text, bg='white', 
                                 fg=color, font=('Arial', 9, 'bold'))
            stat_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # 响应内容
        response_frame = tk.Frame(content_frame, bg='white')
        response_frame.pack(fill=tk.BOTH, expand=True)
        
        response_text = scrolledtext.ScrolledText(response_frame, height=8, wrap=tk.WORD, 
                                                font=('Consolas', 10),
                                                bg='#f8f9fa', fg='#2c3e50',
                                                relief='flat', borderwidth=1)
        response_text.pack(fill=tk.BOTH, expand=True)
        response_text.insert(1.0, result['response'])
        response_text.config(state=tk.DISABLED)
    
    def create_summary_frame(self, successful_results: List[Dict[str, Any]], total_models: int):
        """创建统计摘要框架"""
        # 创建摘要卡片
        summary_card = tk.Frame(self.scrollable_frame, bg='#f8f9fa', relief='flat', 
                               borderwidth=1, highlightbackground='#3498db', 
                               highlightthickness=2)
        summary_card.pack(fill=tk.X, pady=(10, 0), padx=10)
        
        # 摘要内容
        content_frame = tk.Frame(summary_card, bg='#f8f9fa')
        content_frame.pack(fill=tk.X, padx=15, pady=15)
        
        # 摘要标题
        summary_title = tk.Label(content_frame, text="📈 统计摘要", 
                                bg='#f8f9fa', fg='#2c3e50', 
                                font=('Arial', 12, 'bold'))
        summary_title.pack(anchor=tk.W, pady=(0, 10))
        
        speeds = [r['tokens_per_second'] for r in successful_results]
        durations = [r['duration'] for r in successful_results]
        
        # 找到最快和最慢的模型
        fastest_model = successful_results[speeds.index(max(speeds))]
        slowest_model = successful_results[speeds.index(min(speeds))]
        fastest_duration_model = successful_results[durations.index(min(durations))]
        slowest_duration_model = successful_results[durations.index(max(durations))]
        
        # 创建统计网格
        stats_grid = tk.Frame(content_frame, bg='#f8f9fa')
        stats_grid.pack(fill=tk.X)
        
        # 第一行：总体统计
        overall_frame = tk.Frame(stats_grid, bg='#f8f9fa')
        overall_frame.pack(fill=tk.X, pady=(0, 10))
        
        overall_stats = [
            (f"总模型: {total_models}", "#34495e"),
            (f"成功: {len(successful_results)}", "#27ae60"),
            (f"失败: {total_models - len(successful_results)}", "#e74c3c")
        ]
        
        for text, color in overall_stats:
            stat_label = tk.Label(overall_frame, text=text, bg='#f8f9fa', 
                                 fg=color, font=('Arial', 10, 'bold'))
            stat_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # 第二行：速度统计
        speed_frame = tk.Frame(stats_grid, bg='#f8f9fa')
        speed_frame.pack(fill=tk.X, pady=(0, 10))
        
        speed_stats = [
            (f"🥇 最快: {max(speeds):.2f}/s ({fastest_model['model'][:15]}...)", "#27ae60"),
            (f"🐌 最慢: {min(speeds):.2f}/s ({slowest_model['model'][:15]}...)", "#e74c3c"),
            (f"📊 平均: {sum(speeds)/len(speeds):.2f}/s", "#3498db")
        ]
        
        for text, color in speed_stats:
            stat_label = tk.Label(speed_frame, text=text, bg='#f8f9fa', 
                                 fg=color, font=('Arial', 9))
            stat_label.pack(side=tk.LEFT, padx=(0, 15))
        
        # 第三行：耗时统计
        duration_frame = tk.Frame(stats_grid, bg='#f8f9fa')
        duration_frame.pack(fill=tk.X)
        
        duration_stats = [
            (f"⚡ 最快: {min(durations):.2f}s ({fastest_duration_model['model'][:15]}...)", "#27ae60"),
            (f"⏳ 最慢: {max(durations):.2f}s ({slowest_duration_model['model'][:15]}...)", "#e74c3c"),
            (f"📊 平均: {sum(durations)/len(durations):.2f}s", "#3498db")
        ]
        
        for text, color in duration_stats:
            stat_label = tk.Label(duration_frame, text=text, bg='#f8f9fa', 
                                 fg=color, font=('Arial', 9))
            stat_label.pack(side=tk.LEFT, padx=(0, 15))
    
    def save_results(self):
        """保存结果到文件"""
        if not self.current_results:
            messagebox.showwarning("警告", "没有结果可保存")
            return
        
        prompt = self.prompt_text.get(1.0, tk.END).strip()
        if not prompt:
            prompt = "未保存的提示词"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interactive_test_{timestamp}.md"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown files", "*.md"), ("All files", "*.*")],
            initialvalue=filename,
            initialdir=self.output_dir
        )
        
        if filepath:
            save_results_to_file(self.current_results, prompt, Path(filepath).parent, 
                               Path(filepath).name)
            messagebox.showinfo("成功", f"结果已保存到: {filepath}")
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()


def main():
    """主函数"""
    app = ModelComparisonGUI()
    app.run()


if __name__ == "__main__":
    main()