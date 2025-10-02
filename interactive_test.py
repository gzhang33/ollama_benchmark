#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Model Comparison Tool with GUI
可视化交互式模型对比工具 - 单界面输入，所有模型并行回答
"""

import concurrent.futures
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog


class MarkdownRenderer:
    """Markdown渲染器类 - 将Markdown文本渲染到tkinter.Text组件"""
    
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.setup_tags()
    
    def setup_tags(self):
        """设置文本标签样式"""
        # 标题样式
        self.text_widget.tag_configure("h1", font=('Arial', 16, 'bold'), foreground='#2c3e50')
        self.text_widget.tag_configure("h2", font=('Arial', 14, 'bold'), foreground='#2c3e50')
        self.text_widget.tag_configure("h3", font=('Arial', 12, 'bold'), foreground='#2c3e50')
        
        # 代码样式
        self.text_widget.tag_configure("code", font=('Consolas', 10), 
                                      background='#f8f9fa', foreground='#e74c3c')
        self.text_widget.tag_configure("codeblock", font=('Consolas', 10), 
                                      background='#f8f9fa', foreground='#2c3e50',
                                      relief='sunken', borderwidth=1)
        
        # 强调样式
        self.text_widget.tag_configure("bold", font=('Arial', 10, 'bold'))
        self.text_widget.tag_configure("italic", font=('Arial', 10, 'italic'))
        
        # 列表样式
        self.text_widget.tag_configure("list_item", lmargin1=20, lmargin2=40)
        self.text_widget.tag_configure("list_bullet", foreground='#3498db')
        
        # 引用样式
        self.text_widget.tag_configure("quote", foreground='#7f8c8d', 
                                      lmargin1=20, lmargin2=40)
    
    def render_markdown(self, markdown_text: str):
        """渲染Markdown文本到Text组件"""
        self.text_widget.delete(1.0, tk.END)
        
        lines = markdown_text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].rstrip()
            
            # 处理标题
            if line.startswith('# '):
                self.render_heading(line[2:], "h1")
            elif line.startswith('## '):
                self.render_heading(line[3:], "h2")
            elif line.startswith('### '):
                self.render_heading(line[4:], "h3")
            
            # 处理代码块
            elif line.startswith('```'):
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    code_lines.append(lines[i])
                    i += 1
                self.render_code_block('\n'.join(code_lines))
            
            # 处理列表
            elif line.startswith('- ') or line.startswith('* '):
                self.render_list_item(line[2:])
            
            # 处理数字列表
            elif re.match(r'^\d+\. ', line):
                self.render_list_item(line[re.search(r'^\d+\. ', line).end():], numbered=True)
            
            # 处理引用
            elif line.startswith('> '):
                self.render_quote(line[2:])
            
            # 处理普通段落
            elif line.strip():
                self.render_paragraph(line)
            
            # 空行
            else:
                self.text_widget.insert(tk.END, '\n')
            
            i += 1
    
    def render_heading(self, text: str, tag: str):
        """渲染标题"""
        start = self.text_widget.index(tk.END + '-1c')
        self.text_widget.insert(tk.END, text + '\n\n')
        end = self.text_widget.index(tk.END + '-2c')
        self.text_widget.tag_add(tag, start, end)
    
    def render_code_block(self, code: str):
        """渲染代码块"""
        start = self.text_widget.index(tk.END + '-1c')
        self.text_widget.insert(tk.END, code + '\n\n')
        end = self.text_widget.index(tk.END + '-2c')
        self.text_widget.tag_add("codeblock", start, end)
    
    def render_list_item(self, text: str, numbered: bool = False):
        """渲染列表项"""
        start = self.text_widget.index(tk.END + '-1c')
        bullet = "• " if not numbered else ""
        self.text_widget.insert(tk.END, bullet + text + '\n')
        end = self.text_widget.index(tk.END + '-1c')
        self.text_widget.tag_add("list_item", start, end)
        if bullet:
            bullet_end = self.text_widget.index(f"{start}+{len(bullet)}c")
            self.text_widget.tag_add("list_bullet", start, bullet_end)
    
    def render_quote(self, text: str):
        """渲染引用"""
        start = self.text_widget.index(tk.END + '-1c')
        self.text_widget.insert(tk.END, text + '\n')
        end = self.text_widget.index(tk.END + '-1c')
        self.text_widget.tag_add("quote", start, end)
    
    def render_paragraph(self, text: str):
        """渲染段落（处理内联样式）"""
        self.render_inline_styles(text)
        self.text_widget.insert(tk.END, '\n')
    
    def render_inline_styles(self, text: str):
        """渲染内联样式（粗体、斜体、代码）"""
        # 处理代码
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        
        # 处理粗体
        text = re.sub(r'\*\*([^*]+)\*\*', r'<bold>\1</bold>', text)
        text = re.sub(r'__([^_]+)__', r'<bold>\1</bold>', text)
        
        # 处理斜体
        text = re.sub(r'\*([^*]+)\*', r'<italic>\1</italic>', text)
        text = re.sub(r'_([^_]+)_', r'<italic>\1</italic>', text)
        
        # 解析并应用标签
        parts = re.split(r'(<[^>]+>)', text)
        for part in parts:
            if part.startswith('<code>'):
                content = part[6:-7]  # 移除标签
                start = self.text_widget.index(tk.END + '-1c')
                self.text_widget.insert(tk.END, content)
                end = self.text_widget.index(tk.END + '-1c')
                self.text_widget.tag_add("code", start, end)
            elif part.startswith('<bold>'):
                content = part[6:-7]
                start = self.text_widget.index(tk.END + '-1c')
                self.text_widget.insert(tk.END, content)
                end = self.text_widget.index(tk.END + '-1c')
                self.text_widget.tag_add("bold", start, end)
            elif part.startswith('<italic>'):
                content = part[8:-9]
                start = self.text_widget.index(tk.END + '-1c')
                self.text_widget.insert(tk.END, content)
                end = self.text_widget.index(tk.END + '-1c')
                self.text_widget.tag_add("italic", start, end)
            elif not part.startswith('<'):
                self.text_widget.insert(tk.END, part)


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
        self.view_mode = "vertical"  # "vertical" 或 "horizontal"
        
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
        
        # 结果标题和控制区域
        title_frame = tk.Frame(results_card, bg='white')
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        results_title = ttk.Label(title_frame, text="📊 模型响应对比", 
                                 background='white',
                                 foreground='#2c3e50',
                                 font=('Arial', 12, 'bold'))
        results_title.pack(side=tk.LEFT)
        
        # 视图切换按钮
        view_frame = tk.Frame(title_frame, bg='white')
        view_frame.pack(side=tk.RIGHT)
        
        self.view_mode_var = tk.StringVar(value="vertical")
        ttk.Radiobutton(view_frame, text="垂直视图", variable=self.view_mode_var, 
                       value="vertical", command=self.switch_view_mode,
                       style='TRadiobutton').pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(view_frame, text="横向对比", variable=self.view_mode_var, 
                       value="horizontal", command=self.switch_view_mode,
                       style='TRadiobutton').pack(side=tk.LEFT)
        
        # 创建结果显示容器
        self.results_container = tk.Frame(results_card, bg='white')
        self.results_container.pack(fill=tk.BOTH, expand=True)
        
        # 垂直视图容器
        self.vertical_container = tk.Frame(self.results_container, bg='white')
        
        # 创建滚动区域（垂直视图）
        self.vertical_canvas = tk.Canvas(self.vertical_container, bg='white', highlightthickness=0)
        self.vertical_scrollbar = ttk.Scrollbar(self.vertical_container, orient="vertical", command=self.vertical_canvas.yview)
        self.scrollable_frame = tk.Frame(self.vertical_canvas, bg='white')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.vertical_canvas.configure(scrollregion=self.vertical_canvas.bbox("all"))
        )
        
        self.vertical_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.vertical_canvas.configure(yscrollcommand=self.vertical_scrollbar.set)
        
        # 横向视图容器（Notebook）
        self.horizontal_notebook = ttk.Notebook(self.results_container)
        
        # 默认显示垂直视图
        self.vertical_container.pack(fill=tk.BOTH, expand=True)
    
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
    
    def switch_view_mode(self):
        """切换视图模式"""
        new_mode = self.view_mode_var.get()
        if new_mode != self.view_mode:
            self.view_mode = new_mode
            
            # 清除当前显示
            for widget in self.results_container.winfo_children():
                widget.pack_forget()
            
            if self.view_mode == "vertical":
                self.vertical_container.pack(fill=tk.BOTH, expand=True)
                self.vertical_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                self.vertical_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            else:
                self.horizontal_notebook.pack(fill=tk.BOTH, expand=True)
            
            # 重新显示当前结果
            if self.current_results:
                prompt = self.prompt_text.get(1.0, tk.END).strip()
                self.display_results_gui(self.current_results, prompt)
    
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
        if self.view_mode == "vertical":
            self.display_vertical_results(results, prompt)
        else:
            self.display_horizontal_results(results, prompt)
    
    def display_vertical_results(self, results: List[Dict[str, Any]], prompt: str):
        """显示垂直布局结果"""
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
            self.create_vertical_result_frame(result, idx)
        
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
    
    def display_horizontal_results(self, results: List[Dict[str, Any]], prompt: str):
        """显示横向布局结果（并排显示模式）"""
        # 清除之前的结果
        for widget in self.horizontal_notebook.winfo_children():
            widget.destroy()
        
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        if not successful_results:
            return
        
        # 创建主容器
        main_frame = tk.Frame(self.horizontal_notebook, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 显示提示词
        self.create_horizontal_prompt_section(main_frame, prompt)
        
        # 创建模型结果并排显示区域
        self.create_horizontal_models_section(main_frame, successful_results)
        
        # 显示失败模型
        if failed_results:
            self.create_horizontal_failed_section(main_frame, failed_results)
        
        # 显示统计摘要
        self.create_horizontal_summary_section(main_frame, successful_results, len(results))
    
    def create_vertical_result_frame(self, result: Dict[str, Any], idx: int):
        """创建垂直布局的结果框架"""
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
        
        # 响应内容（支持Markdown渲染）
        response_frame = tk.Frame(content_frame, bg='white')
        response_frame.pack(fill=tk.BOTH, expand=True)
        
        response_text = scrolledtext.ScrolledText(response_frame, height=8, wrap=tk.WORD, 
                                                font=('Arial', 10),
                                                bg='#f8f9fa', fg='#2c3e50',
                                                relief='flat', borderwidth=1)
        response_text.pack(fill=tk.BOTH, expand=True)
        
        # 使用Markdown渲染器
        renderer = MarkdownRenderer(response_text)
        renderer.render_markdown(result['response'])
        response_text.config(state=tk.DISABLED)
    
    def create_horizontal_result_tab(self, result: Dict[str, Any], idx: int, prompt: str):
        """创建横向布局的结果标签页"""
        # 创建标签页框架
        tab_frame = tk.Frame(self.horizontal_notebook, bg='white')
        
        # 添加标签页
        tab_title = f"[{idx}] {result['model'][:15]}{'...' if len(result['model']) > 15 else ''}"
        self.horizontal_notebook.add(tab_frame, text=tab_title)
        
        # 创建滚动区域
        canvas = tk.Canvas(tab_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 提示词区域
        prompt_frame = tk.Frame(scrollable_frame, bg='#f8f9fa', relief='flat', 
                               borderwidth=1, highlightbackground='#e9ecef', 
                               highlightthickness=1)
        prompt_frame.pack(fill=tk.X, pady=(0, 15), padx=10)
        
        prompt_content = tk.Frame(prompt_frame, bg='#f8f9fa')
        prompt_content.pack(fill=tk.X, padx=15, pady=15)
        
        prompt_title = tk.Label(prompt_content, text="📝 输入提示词", 
                               bg='#f8f9fa', fg='#2c3e50', 
                               font=('Arial', 12, 'bold'))
        prompt_title.pack(anchor=tk.W, pady=(0, 10))
        
        prompt_label = tk.Label(prompt_content, text=prompt, wraplength=800,
                               bg='#f8f9fa', fg='#34495e', 
                               font=('Arial', 11), justify=tk.LEFT)
        prompt_label.pack(anchor=tk.W)
        
        # 统计信息区域
        stats_frame = tk.Frame(scrollable_frame, bg='#e8f4fd', relief='flat', 
                              borderwidth=1, highlightbackground='#3498db', 
                              highlightthickness=1)
        stats_frame.pack(fill=tk.X, pady=(0, 15), padx=10)
        
        stats_content = tk.Frame(stats_frame, bg='#e8f4fd')
        stats_content.pack(fill=tk.X, padx=15, pady=15)
        
        stats_title = tk.Label(stats_content, text="📊 性能统计", 
                              bg='#e8f4fd', fg='#2c3e50', 
                              font=('Arial', 12, 'bold'))
        stats_title.pack(anchor=tk.W, pady=(0, 10))
        
        stats_data = [
            (f"⏱️ 耗时: {result['duration']:.2f}s", "#f39c12"),
            (f"📊 Token数: {result['tokens']}", "#3498db"),
            (f"🚀 速度: {result['tokens_per_second']:.2f} tokens/s", "#27ae60")
        ]
        
        stats_grid = tk.Frame(stats_content, bg='#e8f4fd')
        stats_grid.pack(fill=tk.X)
        
        for i, (text, color) in enumerate(stats_data):
            stat_label = tk.Label(stats_grid, text=text, bg='#e8f4fd', 
                                 fg=color, font=('Arial', 10, 'bold'))
            stat_label.pack(side=tk.LEFT, padx=(0, 30))
        
        # 响应内容区域
        response_frame = tk.Frame(scrollable_frame, bg='white', relief='flat', 
                                 borderwidth=1, highlightbackground='#e9ecef', 
                                 highlightthickness=1)
        response_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        response_content = tk.Frame(response_frame, bg='white')
        response_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        response_title = tk.Label(response_content, text="🤖 模型响应", 
                                 bg='white', fg='#2c3e50', 
                                 font=('Arial', 12, 'bold'))
        response_title.pack(anchor=tk.W, pady=(0, 10))
        
        # 响应文本（支持Markdown渲染）
        response_text = scrolledtext.ScrolledText(response_content, wrap=tk.WORD, 
                                                font=('Arial', 10),
                                                bg='#f8f9fa', fg='#2c3e50',
                                                relief='flat', borderwidth=1)
        response_text.pack(fill=tk.BOTH, expand=True)
        
        # 使用Markdown渲染器
        renderer = MarkdownRenderer(response_text)
        renderer.render_markdown(result['response'])
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
    
    def create_failed_models_tab(self, failed_results: List[Dict[str, Any]]):
        """创建失败模型标签页"""
        tab_frame = tk.Frame(self.horizontal_notebook, bg='white')
        self.horizontal_notebook.add(tab_frame, text="❌ 失败模型")
        
        # 创建滚动区域
        canvas = tk.Canvas(tab_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 失败信息区域
        failed_frame = tk.Frame(scrollable_frame, bg='#ffeaa7', relief='flat', 
                               borderwidth=1, highlightbackground='#e17055', 
                               highlightthickness=1)
        failed_frame.pack(fill=tk.X, pady=15, padx=15)
        
        failed_content = tk.Frame(failed_frame, bg='#ffeaa7')
        failed_content.pack(fill=tk.X, padx=15, pady=15)
        
        failed_title = tk.Label(failed_content, text="❌ 失败的模型", 
                               bg='#ffeaa7', fg='#d63031', 
                               font=('Arial', 14, 'bold'))
        failed_title.pack(anchor=tk.W, pady=(0, 15))
        
        for result in failed_results:
            error_frame = tk.Frame(failed_content, bg='#ffeaa7')
            error_frame.pack(fill=tk.X, pady=5)
            
            model_name = tk.Label(error_frame, text=f"• {result['model']}", 
                                 bg='#ffeaa7', fg='#d63031', 
                                 font=('Arial', 11, 'bold'))
            model_name.pack(anchor=tk.W)
            
            error_msg = tk.Label(error_frame, text=f"  错误: {result.get('error', 'Unknown error')}", 
                                bg='#ffeaa7', fg='#8b4513', 
                                font=('Arial', 10))
            error_msg.pack(anchor=tk.W, pady=(2, 10))
    
    def create_summary_tab(self, successful_results: List[Dict[str, Any]], total_models: int):
        """创建统计摘要标签页"""
        tab_frame = tk.Frame(self.horizontal_notebook, bg='white')
        self.horizontal_notebook.add(tab_frame, text="📈 统计摘要")
        
        # 创建滚动区域
        canvas = tk.Canvas(tab_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 摘要卡片
        summary_card = tk.Frame(scrollable_frame, bg='#f8f9fa', relief='flat', 
                               borderwidth=1, highlightbackground='#3498db', 
                               highlightthickness=2)
        summary_card.pack(fill=tk.X, pady=15, padx=15)
        
        summary_content = tk.Frame(summary_card, bg='#f8f9fa')
        summary_content.pack(fill=tk.X, padx=20, pady=20)
        
        summary_title = tk.Label(summary_content, text="📈 统计摘要", 
                                bg='#f8f9fa', fg='#2c3e50', 
                                font=('Arial', 14, 'bold'))
        summary_title.pack(anchor=tk.W, pady=(0, 15))
        
        speeds = [r['tokens_per_second'] for r in successful_results]
        durations = [r['duration'] for r in successful_results]
        
        # 找到最快和最慢的模型
        fastest_model = successful_results[speeds.index(max(speeds))]
        slowest_model = successful_results[speeds.index(min(speeds))]
        fastest_duration_model = successful_results[durations.index(min(durations))]
        slowest_duration_model = successful_results[durations.index(max(durations))]
        
        # 总体统计
        overall_frame = tk.Frame(summary_content, bg='#f8f9fa')
        overall_frame.pack(fill=tk.X, pady=(0, 15))
        
        overall_title = tk.Label(overall_frame, text="📊 总体统计", 
                                bg='#f8f9fa', fg='#2c3e50', 
                                font=('Arial', 12, 'bold'))
        overall_title.pack(anchor=tk.W, pady=(0, 10))
        
        overall_stats = [
            (f"总模型数: {total_models}", "#34495e"),
            (f"成功: {len(successful_results)}", "#27ae60"),
            (f"失败: {total_models - len(successful_results)}", "#e74c3c")
        ]
        
        for text, color in overall_stats:
            stat_label = tk.Label(overall_frame, text=text, bg='#f8f9fa', 
                                 fg=color, font=('Arial', 11, 'bold'))
            stat_label.pack(anchor=tk.W, pady=2)
        
        # 速度统计
        speed_frame = tk.Frame(summary_content, bg='#f8f9fa')
        speed_frame.pack(fill=tk.X, pady=(0, 15))
        
        speed_title = tk.Label(speed_frame, text="🚀 速度统计", 
                              bg='#f8f9fa', fg='#2c3e50', 
                              font=('Arial', 12, 'bold'))
        speed_title.pack(anchor=tk.W, pady=(0, 10))
        
        speed_stats = [
            (f"🥇 最快速度: {max(speeds):.2f} tokens/s ({fastest_model['model']})", "#27ae60"),
            (f"🐌 最慢速度: {min(speeds):.2f} tokens/s ({slowest_model['model']})", "#e74c3c"),
            (f"📊 平均速度: {sum(speeds)/len(speeds):.2f} tokens/s", "#3498db")
        ]
        
        for text, color in speed_stats:
            stat_label = tk.Label(speed_frame, text=text, bg='#f8f9fa', 
                                 fg=color, font=('Arial', 10))
            stat_label.pack(anchor=tk.W, pady=2)
        
        # 耗时统计
        duration_frame = tk.Frame(summary_content, bg='#f8f9fa')
        duration_frame.pack(fill=tk.X)
        
        duration_title = tk.Label(duration_frame, text="⏱️ 耗时统计", 
                                 bg='#f8f9fa', fg='#2c3e50', 
                                 font=('Arial', 12, 'bold'))
        duration_title.pack(anchor=tk.W, pady=(0, 10))
        
        duration_stats = [
            (f"⚡ 最快完成: {min(durations):.2f}s ({fastest_duration_model['model']})", "#27ae60"),
            (f"⏳ 最慢完成: {max(durations):.2f}s ({slowest_duration_model['model']})", "#e74c3c"),
            (f"📊 平均耗时: {sum(durations)/len(durations):.2f}s", "#3498db")
        ]
        
        for text, color in duration_stats:
            stat_label = tk.Label(duration_frame, text=text, bg='#f8f9fa', 
                                 fg=color, font=('Arial', 10))
            stat_label.pack(anchor=tk.W, pady=2)
    
    def create_horizontal_prompt_section(self, parent, prompt: str):
        """创建横向视图的提示词区域"""
        prompt_frame = tk.Frame(parent, bg='#f8f9fa', relief='flat', 
                               borderwidth=1, highlightbackground='#e9ecef', 
                               highlightthickness=1)
        prompt_frame.pack(fill=tk.X, pady=(0, 15), padx=15)
        
        prompt_content = tk.Frame(prompt_frame, bg='#f8f9fa')
        prompt_content.pack(fill=tk.X, padx=15, pady=15)
        
        prompt_title = tk.Label(prompt_content, text="📝 输入提示词", 
                               bg='#f8f9fa', fg='#2c3e50', 
                               font=('Arial', 14, 'bold'))
        prompt_title.pack(anchor=tk.W, pady=(0, 10))
        
        prompt_label = tk.Label(prompt_content, text=prompt, wraplength=1200,
                               bg='#f8f9fa', fg='#34495e', 
                               font=('Arial', 12), justify=tk.LEFT)
        prompt_label.pack(anchor=tk.W)
    
    def create_horizontal_models_section(self, parent, successful_results: List[Dict[str, Any]]):
        """创建横向视图的模型结果并排显示区域"""
        # 创建滚动区域
        canvas = tk.Canvas(parent, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建模型结果并排显示
        models_frame = tk.Frame(scrollable_frame, bg='white')
        models_frame.pack(fill=tk.X, pady=(0, 15), padx=15)
        
        # 标题
        models_title = tk.Label(models_frame, text="🤖 模型响应对比", 
                               bg='white', fg='#2c3e50', 
                               font=('Arial', 14, 'bold'))
        models_title.pack(anchor=tk.W, pady=(0, 15))
        
        # 计算每行显示的模型数量（根据屏幕宽度调整）
        models_per_row = min(3, len(successful_results))  # 最多3个模型并排
        
        # 创建模型结果网格
        for i, result in enumerate(successful_results):
            row = i // models_per_row
            col = i % models_per_row
            
            # 创建单个模型结果框架
            model_frame = self.create_horizontal_single_model_frame(models_frame, result, i + 1)
            model_frame.grid(row=row, column=col, sticky=(tk.W, tk.E, tk.N, tk.S), 
                           padx=5, pady=5)
        
        # 配置网格权重
        for i in range(models_per_row):
            models_frame.columnconfigure(i, weight=1)
    
    def create_horizontal_single_model_frame(self, parent, result: Dict[str, Any], idx: int):
        """创建单个模型的横向显示框架"""
        # 创建模型卡片
        model_card = tk.Frame(parent, bg='white', relief='flat', 
                             borderwidth=1, highlightbackground='#e9ecef', 
                             highlightthickness=1)
        
        # 模型标题和统计信息
        header_frame = tk.Frame(model_card, bg='#f8f9fa')
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 模型名称
        model_title = tk.Label(header_frame, text=f"[{idx}] {result['model']}", 
                              bg='#f8f9fa', fg='#2c3e50', 
                              font=('Arial', 12, 'bold'))
        model_title.pack(anchor=tk.W, pady=(0, 5))
        
        # 统计信息
        stats_frame = tk.Frame(header_frame, bg='#f8f9fa')
        stats_frame.pack(fill=tk.X)
        
        stats_data = [
            (f"⏱️ {result['duration']:.2f}s", "#f39c12"),
            (f"📊 {result['tokens']}", "#3498db"),
            (f"🚀 {result['tokens_per_second']:.2f}/s", "#27ae60")
        ]
        
        for text, color in stats_data:
            stat_label = tk.Label(stats_frame, text=text, bg='#f8f9fa', 
                                 fg=color, font=('Arial', 9, 'bold'))
            stat_label.pack(anchor=tk.W, pady=1)
        
        # 响应内容区域
        response_frame = tk.Frame(model_card, bg='white')
        response_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # 响应文本（支持Markdown渲染）
        response_text = scrolledtext.ScrolledText(response_frame, height=12, wrap=tk.WORD, 
                                                font=('Arial', 10),
                                                bg='#f8f9fa', fg='#2c3e50',
                                                relief='flat', borderwidth=1)
        response_text.pack(fill=tk.BOTH, expand=True)
        
        # 使用Markdown渲染器
        renderer = MarkdownRenderer(response_text)
        renderer.render_markdown(result['response'])
        response_text.config(state=tk.DISABLED)
        
        return model_card
    
    def create_horizontal_failed_section(self, parent, failed_results: List[Dict[str, Any]]):
        """创建横向视图的失败模型区域"""
        failed_frame = tk.Frame(parent, bg='#ffeaa7', relief='flat', 
                               borderwidth=1, highlightbackground='#e17055', 
                               highlightthickness=1)
        failed_frame.pack(fill=tk.X, pady=(0, 15), padx=15)
        
        failed_content = tk.Frame(failed_frame, bg='#ffeaa7')
        failed_content.pack(fill=tk.X, padx=15, pady=15)
        
        failed_title = tk.Label(failed_content, text="❌ 失败的模型", 
                               bg='#ffeaa7', fg='#d63031', 
                               font=('Arial', 14, 'bold'))
        failed_title.pack(anchor=tk.W, pady=(0, 10))
        
        for result in failed_results:
            error_frame = tk.Frame(failed_content, bg='#ffeaa7')
            error_frame.pack(fill=tk.X, pady=2)
            
            model_name = tk.Label(error_frame, text=f"• {result['model']}", 
                                 bg='#ffeaa7', fg='#d63031', 
                                 font=('Arial', 11, 'bold'))
            model_name.pack(anchor=tk.W)
            
            error_msg = tk.Label(error_frame, text=f"  错误: {result.get('error', 'Unknown error')}", 
                                bg='#ffeaa7', fg='#8b4513', 
                                font=('Arial', 10))
            error_msg.pack(anchor=tk.W, pady=(2, 5))
    
    def create_horizontal_summary_section(self, parent, successful_results: List[Dict[str, Any]], total_models: int):
        """创建横向视图的统计摘要区域"""
        summary_frame = tk.Frame(parent, bg='#f8f9fa', relief='flat', 
                                borderwidth=1, highlightbackground='#3498db', 
                                highlightthickness=2)
        summary_frame.pack(fill=tk.X, pady=(0, 15), padx=15)
        
        summary_content = tk.Frame(summary_frame, bg='#f8f9fa')
        summary_content.pack(fill=tk.X, padx=20, pady=20)
        
        summary_title = tk.Label(summary_content, text="📈 统计摘要", 
                                bg='#f8f9fa', fg='#2c3e50', 
                                font=('Arial', 14, 'bold'))
        summary_title.pack(anchor=tk.W, pady=(0, 15))
        
        speeds = [r['tokens_per_second'] for r in successful_results]
        durations = [r['duration'] for r in successful_results]
        
        # 找到最快和最慢的模型
        fastest_model = successful_results[speeds.index(max(speeds))]
        slowest_model = successful_results[speeds.index(min(speeds))]
        fastest_duration_model = successful_results[durations.index(min(durations))]
        slowest_duration_model = successful_results[durations.index(max(durations))]
        
        # 创建统计网格
        stats_grid = tk.Frame(summary_content, bg='#f8f9fa')
        stats_grid.pack(fill=tk.X)
        
        # 第一行：总体统计
        overall_frame = tk.Frame(stats_grid, bg='#f8f9fa')
        overall_frame.pack(fill=tk.X, pady=(0, 10))
        
        overall_title = tk.Label(overall_frame, text="📊 总体统计", 
                                bg='#f8f9fa', fg='#2c3e50', 
                                font=('Arial', 12, 'bold'))
        overall_title.pack(anchor=tk.W, pady=(0, 5))
        
        overall_stats = [
            (f"总模型数: {total_models}", "#34495e"),
            (f"成功: {len(successful_results)}", "#27ae60"),
            (f"失败: {total_models - len(successful_results)}", "#e74c3c")
        ]
        
        for text, color in overall_stats:
            stat_label = tk.Label(overall_frame, text=text, bg='#f8f9fa', 
                                 fg=color, font=('Arial', 11, 'bold'))
            stat_label.pack(anchor=tk.W, pady=1)
        
        # 第二行：速度统计
        speed_frame = tk.Frame(stats_grid, bg='#f8f9fa')
        speed_frame.pack(fill=tk.X, pady=(0, 10))
        
        speed_title = tk.Label(speed_frame, text="🚀 速度统计", 
                              bg='#f8f9fa', fg='#2c3e50', 
                              font=('Arial', 12, 'bold'))
        speed_title.pack(anchor=tk.W, pady=(0, 5))
        
        speed_stats = [
            (f"🥇 最快速度: {max(speeds):.2f} tokens/s ({fastest_model['model']})", "#27ae60"),
            (f"🐌 最慢速度: {min(speeds):.2f} tokens/s ({slowest_model['model']})", "#e74c3c"),
            (f"📊 平均速度: {sum(speeds)/len(speeds):.2f} tokens/s", "#3498db")
        ]
        
        for text, color in speed_stats:
            stat_label = tk.Label(speed_frame, text=text, bg='#f8f9fa', 
                                 fg=color, font=('Arial', 10))
            stat_label.pack(anchor=tk.W, pady=1)
        
        # 第三行：耗时统计
        duration_frame = tk.Frame(stats_grid, bg='#f8f9fa')
        duration_frame.pack(fill=tk.X)
        
        duration_title = tk.Label(duration_frame, text="⏱️ 耗时统计", 
                                 bg='#f8f9fa', fg='#2c3e50', 
                                 font=('Arial', 12, 'bold'))
        duration_title.pack(anchor=tk.W, pady=(0, 5))
        
        duration_stats = [
            (f"⚡ 最快完成: {min(durations):.2f}s ({fastest_duration_model['model']})", "#27ae60"),
            (f"⏳ 最慢完成: {max(durations):.2f}s ({slowest_duration_model['model']})", "#e74c3c"),
            (f"📊 平均耗时: {sum(durations)/len(durations):.2f}s", "#3498db")
        ]
        
        for text, color in duration_stats:
            stat_label = tk.Label(duration_frame, text=text, bg='#f8f9fa', 
                                 fg=color, font=('Arial', 10))
            stat_label.pack(anchor=tk.W, pady=1)
    
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