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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog


@dataclass
class ModelResponse:
    """Data model for a single model's response"""
    model: str
    success: bool
    response: str = ""
    duration: float = 0.0
    tokens: int = 0
    tokens_per_second: float = 0.0
    error: str = ""
    
    def is_loading(self) -> bool:
        """Check if response is still loading"""
        return not self.success and not self.error
    
    def has_error(self) -> bool:
        """Check if response has an error"""
        return bool(self.error)
    
    def to_markdown(self) -> str:
        """Convert response to markdown format"""
        if self.has_error():
            return f"**Error**: {self.error}"
        
        md = f"**{self.model}**\n\n"
        md += f"- Duration: {self.duration:.2f}s\n"
        md += f"- Tokens: {self.tokens}\n"
        md += f"- Speed: {self.tokens_per_second:.2f} tokens/s\n\n"
        md += f"```\n{self.response}\n```\n"
        return md


@dataclass
class ConversationEntry:
    """Data model for a conversation entry (query + responses)"""
    query: str
    timestamp: datetime
    responses: List[ModelResponse] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """Convert conversation entry to markdown format"""
        md = f"## Query ({self.timestamp.strftime('%Y-%m-%d %H:%M:%S')})\n\n"
        md += f"{self.query}\n\n"
        md += "### Responses\n\n"
        for response in self.responses:
            md += response.to_markdown() + "\n\n"
        return md


class StateManager:
    """Manages application state including conversation history and view state"""
    
    def __init__(self):
        self.current_view = "landing"  # "landing" or "chat"
        self.conversation_history: List[ConversationEntry] = []
        self.selected_models: List[str] = []
    
    def add_conversation_entry(self, query: str, responses: List[Dict[str, Any]]) -> None:
        """Add a new conversation entry to history"""
        model_responses = []
        for resp in responses:
            model_responses.append(ModelResponse(
                model=resp.get("model", ""),
                success=resp.get("success", False),
                response=resp.get("response", ""),
                duration=resp.get("duration", 0.0),
                tokens=resp.get("tokens", 0),
                tokens_per_second=resp.get("tokens_per_second", 0.0),
                error=resp.get("error", "")
            ))
        
        entry = ConversationEntry(
            query=query,
            timestamp=datetime.now(),
            responses=model_responses
        )
        self.conversation_history.append(entry)
    
    def clear_conversation(self) -> None:
        """Clear all conversation history"""
        self.conversation_history = []
    
    def get_conversation_history(self) -> List[ConversationEntry]:
        """Get the full conversation history"""
        return self.conversation_history
    
    def set_selected_models(self, models: List[str]) -> None:
        """Set the selected models"""
        self.selected_models = models
    
    def get_selected_models(self) -> List[str]:
        """Get the selected models"""
        return self.selected_models
    
    def transition_to_chat(self) -> None:
        """Transition to chat view"""
        self.current_view = "chat"
    
    def transition_to_landing(self) -> None:
        """Transition to landing view"""
        self.current_view = "landing"
    
    def is_chat_view(self) -> bool:
        """Check if current view is chat"""
        return self.current_view == "chat"
    
    def is_landing_view(self) -> bool:
        """Check if current view is landing"""
        return self.current_view == "landing"


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


class LandingPageView:
    """Landing page view with centered input and model selection"""
    
    def __init__(self, parent, on_submit_callback, on_model_select_callback):
        self.parent = parent
        self.on_submit_callback = on_submit_callback
        self.on_model_select_callback = on_model_select_callback
        self.selected_model_count = 0
        
        self.container = tk.Frame(parent, bg='#f8f9fa')
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the landing page UI"""
        # Center container with max width
        center_frame = tk.Frame(self.container, bg='#f8f9fa')
        center_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Model icons placeholder (simple text for MVP)
        icons_label = tk.Label(center_frame, text="🤖 💬 🧠", 
                              font=('Arial', 32),
                              bg='#f8f9fa', fg='#2c3e50')
        icons_label.pack(pady=(0, 20))
        
        # Title
        title_label = tk.Label(center_frame, text="Find the best AI for you",
                              font=('Arial', 24, 'bold'),
                              bg='#f8f9fa', fg='#2c3e50')
        title_label.pack(pady=(0, 10))
        
        # Subtitle
        subtitle_label = tk.Label(center_frame, 
                                 text="Compare multiple AI models side-by-side",
                                 font=('Arial', 12),
                                 bg='#f8f9fa', fg='#7f8c8d')
        subtitle_label.pack(pady=(0, 40))
        
        # Input frame with icons
        input_frame = tk.Frame(center_frame, bg='white', relief='solid', borderwidth=1)
        input_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
# Attachment functionality removed
        
        # Input entry
        self.input_entry = tk.Entry(input_frame, font=('Arial', 12),
                                   bg='white', fg='#2c3e50',
                                   relief='flat', borderwidth=0)
        self.input_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=12)
        self.input_entry.insert(0, "Ask anything...")
        self.input_entry.bind('<FocusIn>', self._on_entry_focus_in)
        self.input_entry.bind('<FocusOut>', self._on_entry_focus_out)
        self.input_entry.bind('<Return>', lambda e: self._on_submit())
        
        # Send button (right)
        send_btn = tk.Button(input_frame, text="➤", font=('Arial', 14),
                           bg='white', fg='#3498db', relief='flat',
                           cursor='hand2', padx=10,
                           command=self._on_submit)
        send_btn.pack(side=tk.RIGHT, padx=(0, 5))
        
        # Model selection dropdown button
        model_btn_frame = tk.Frame(center_frame, bg='#f8f9fa')
        model_btn_frame.pack(pady=(0, 10))
        
        self.model_select_btn = tk.Button(model_btn_frame, 
                                         text="🎯 Select Models ▼",
                                         font=('Arial', 11),
                                         bg='#3498db', fg='white',
                                         relief='flat', cursor='hand2',
                                         padx=20, pady=8,
                                         command=self.toggle_model_dropdown)
        self.model_select_btn.pack()
        
        # Dropdown menu container (initially hidden)
        self.dropdown_container = tk.Frame(center_frame, bg='white',
                                          relief='solid', borderwidth=1)
        self.dropdown_visible = False
        
        # Selected model count display
        self.model_count_label = tk.Label(center_frame,
                                         text="No models selected",
                                         font=('Arial', 10),
                                         bg='#f8f9fa', fg='#7f8c8d')
        self.model_count_label.pack()
    
    def toggle_model_dropdown(self):
        """Toggle the model selection dropdown"""
        if self.dropdown_visible:
            self.hide_dropdown()
        else:
            self.show_dropdown()
    
    def show_dropdown(self):
        """Show the model selection dropdown"""
        self.dropdown_visible = True
        self.model_select_btn.config(text="🎯 Select Models ▲")
        self.dropdown_container.pack(pady=(10, 0))
        
        # Trigger callback to populate dropdown
        self.on_model_select_callback()
    
    def hide_dropdown(self):
        """Hide the model selection dropdown"""
        self.dropdown_visible = False
        self.model_select_btn.config(text="🎯 Select Models ▼")
        self.dropdown_container.pack_forget()
    
    def populate_dropdown(self, models: List[str], selected_models: List[str], on_change_callback):
        """Populate the dropdown with model checkboxes"""
        # Clear existing widgets
        for widget in self.dropdown_container.winfo_children():
            widget.destroy()
        
        # Create scrollable frame
        canvas = tk.Canvas(self.dropdown_container, bg='white', 
                          highlightthickness=0, height=200, width=300)
        scrollbar = ttk.Scrollbar(self.dropdown_container, orient=tk.VERTICAL,
                                 command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add checkboxes for each model
        self.model_vars = {}
        for model in models:
            var = tk.BooleanVar(value=model in selected_models)
            self.model_vars[model] = var
            
            cb = ttk.Checkbutton(scrollable_frame, text=model, 
                               variable=var,
                               command=lambda: on_change_callback(self.get_selected_models()))
            cb.pack(anchor=tk.W, pady=2, padx=10)
        
        # Buttons frame
        btn_frame = tk.Frame(self.dropdown_container, bg='white')
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(btn_frame, text="Select All",
                 font=('Arial', 9),
                 bg='#3498db', fg='white',
                 relief='flat', cursor='hand2',
                 command=lambda: self.select_all_models(on_change_callback)).pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(btn_frame, text="Clear All",
                 font=('Arial', 9),
                 bg='#95a5a6', fg='white',
                 relief='flat', cursor='hand2',
                 command=lambda: self.clear_all_models(on_change_callback)).pack(side=tk.LEFT)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def get_selected_models(self) -> List[str]:
        """Get currently selected models from dropdown"""
        if not hasattr(self, 'model_vars'):
            return []
        return [model for model, var in self.model_vars.items() if var.get()]
    
    def select_all_models(self, on_change_callback):
        """Select all models in dropdown"""
        for var in self.model_vars.values():
            var.set(True)
        on_change_callback(self.get_selected_models())
    
    def clear_all_models(self, on_change_callback):
        """Clear all model selections"""
        for var in self.model_vars.values():
            var.set(False)
        on_change_callback(self.get_selected_models())
    
    def _on_entry_focus_in(self, event):
        """Clear placeholder text on focus"""
        if self.input_entry.get() == "Ask anything...":
            self.input_entry.delete(0, tk.END)
            self.input_entry.config(fg='#2c3e50')
    
    def _on_entry_focus_out(self, event):
        """Restore placeholder text if empty"""
        if not self.input_entry.get():
            self.input_entry.insert(0, "Ask anything...")
            self.input_entry.config(fg='#7f8c8d')
    
    def _on_submit(self):
        """Handle submit button click"""
        query = self.input_entry.get().strip()
        if query and query != "Ask anything...":
            self.on_submit_callback(query)
    
    def update_model_count(self, count: int):
        """Update the selected model count display"""
        self.selected_model_count = count
        if count == 0:
            self.model_count_label.config(text="No models selected")
        elif count == 1:
            self.model_count_label.config(text="1 model selected")
        else:
            self.model_count_label.config(text=f"{count} models selected")
    
    def get_input_text(self) -> str:
        """Get the current input text"""
        text = self.input_entry.get().strip()
        return text if text != "Ask anything..." else ""
    
    def clear_input(self):
        """Clear the input field"""
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, "Ask anything...")
        self.input_entry.config(fg='#7f8c8d')
    
    def show(self):
        """Show the landing page"""
        self.container.pack(fill=tk.BOTH, expand=True)
    
    def hide(self):
        """Hide the landing page"""
        self.container.pack_forget()


class BottomInputBar:
    """Bottom input bar for follow-up queries in chat interface"""
    
    def __init__(self, parent, on_submit_callback):
        self.parent = parent
        self.on_submit_callback = on_submit_callback
        self.is_enabled = True
        
        self.container = tk.Frame(parent, bg='white', height=70)
        self.container.pack_propagate(False)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the bottom input bar UI"""
        # Inner frame with border
        inner_frame = tk.Frame(self.container, bg='white', 
                              relief='solid', borderwidth=1)
        inner_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
# Attachment functionality removed
        
        # Input entry
        self.input_entry = tk.Entry(inner_frame, font=('Arial', 12),
                                   bg='white', fg='#2c3e50',
                                   relief='flat', borderwidth=0)
        self.input_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, 
                             padx=10, pady=8)
        self.input_entry.insert(0, "Ask followup...")
        self.input_entry.bind('<FocusIn>', self._on_entry_focus_in)
        self.input_entry.bind('<FocusOut>', self._on_entry_focus_out)
        self.input_entry.bind('<Return>', lambda e: self._on_submit())
        
        # Send button (right)
        self.send_btn = tk.Button(inner_frame, text="➤", 
                                 font=('Arial', 14),
                                 bg='white', fg='#3498db', 
                                 relief='flat',
                                 cursor='hand2', padx=10,
                                 command=self._on_submit)
        self.send_btn.pack(side=tk.RIGHT, padx=(0, 5))
    
    def _on_entry_focus_in(self, event):
        """Clear placeholder text on focus"""
        if self.input_entry.get() == "Ask followup...":
            self.input_entry.delete(0, tk.END)
            self.input_entry.config(fg='#2c3e50')
    
    def _on_entry_focus_out(self, event):
        """Restore placeholder text if empty"""
        if not self.input_entry.get():
            self.input_entry.insert(0, "Ask followup...")
            self.input_entry.config(fg='#7f8c8d')
    
    def _on_submit(self):
        """Handle submit button click"""
        if not self.is_enabled:
            return
        
        query = self.input_entry.get().strip()
        if query and query != "Ask followup...":
            self.on_submit_callback(query)
    
    def get_input_text(self) -> str:
        """Get the current input text"""
        text = self.input_entry.get().strip()
        return text if text != "Ask followup..." else ""
    
    def clear_input(self):
        """Clear the input field"""
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, "Ask followup...")
        self.input_entry.config(fg='#7f8c8d')
    
    def disable(self):
        """Disable input during query processing"""
        self.is_enabled = False
        self.input_entry.config(state=tk.DISABLED)
        self.send_btn.config(state=tk.DISABLED)
    
    def enable(self):
        """Re-enable input after query completion"""
        self.is_enabled = True
        self.input_entry.config(state=tk.NORMAL)
        self.send_btn.config(state=tk.NORMAL)
        self.input_entry.focus_set()
    
    def show(self):
        """Show the input bar"""
        self.container.pack(side=tk.BOTTOM, fill=tk.X)
    
    def hide(self):
        """Hide the input bar"""
        self.container.pack_forget()


class UserQueryBubble:
    """User query bubble component for chat interface"""
    
    def __init__(self, parent, query_text: str):
        self.parent = parent
        self.query_text = query_text
        
        self.container = tk.Frame(parent, bg='white')
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the query bubble UI"""
        # Right-aligned container
        bubble_container = tk.Frame(self.container, bg='white')
        bubble_container.pack(anchor=tk.E, padx=20, pady=10)
        
        # Query bubble with background
        bubble_frame = tk.Frame(bubble_container, bg='#e3f2fd', 
                               relief='flat', borderwidth=0)
        bubble_frame.pack()
        
        # Query text with wrapping
        query_label = tk.Label(bubble_frame, text=self.query_text,
                              font=('Arial', 11),
                              bg='#e3f2fd', fg='#2c3e50',
                              wraplength=400, justify=tk.LEFT,
                              padx=15, pady=12)
        query_label.pack()
    
    def pack(self, **kwargs):
        """Pack the container"""
        self.container.pack(**kwargs)


class ModelResponseColumn:
    """Model response column component showing individual model response"""
    
    def __init__(self, parent, model_response: ModelResponse):
        self.parent = parent
        self.model_response = model_response
        
        self.container = tk.Frame(parent, bg='white', 
                                 relief='solid', borderwidth=1,
                                 width=350)
        # Remove pack_propagate(False) to allow natural sizing
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the column UI"""
        # Header with model name
        header_frame = tk.Frame(self.container, bg='#f8f9fa', height=40)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        model_label = tk.Label(header_frame, 
                              text=self.model_response.model,
                              font=('Arial', 11, 'bold'),
                              bg='#f8f9fa', fg='#2c3e50',
                              padx=10)
        model_label.pack(side=tk.LEFT, pady=10)
        
        # Content area (no individual scrolling) - expand to fill available space
        self.content_text = tk.Text(
            self.container,
            font=('Arial', 10),
            bg='white', fg='#2c3e50',
            wrap=tk.WORD,
            relief='flat',
            padx=10, pady=10,
            width=40    # Fixed width for consistent column sizing
        )
        self.content_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))
        
        # Footer with metrics
        footer_frame = tk.Frame(self.container, bg='#f8f9fa', height=30)
        footer_frame.pack(fill=tk.X, pady=(0, 5))
        footer_frame.pack_propagate(False)
        
        self.metrics_label = tk.Label(footer_frame,
                                     text="",
                                     font=('Arial', 9),
                                     bg='#f8f9fa', fg='#7f8c8d',
                                     padx=10)
        self.metrics_label.pack(side=tk.LEFT, pady=5)
        
        # Render content based on state
        self.render_content()
    
    def render_content(self):
        """Render content based on response state"""
        print(f"[DEBUG] render_content: {self.model_response.model}, has_error={self.model_response.has_error()}, is_loading={self.model_response.is_loading()}, success={self.model_response.success}")
        
        # Ensure widget is in normal state before modifying
        self.content_text.config(state=tk.NORMAL)
        self.content_text.delete(1.0, tk.END)
        
        if self.model_response.has_error():
            # Error state
            print(f"[DEBUG] Rendering error state for {self.model_response.model}: {self.model_response.error}")
            self.content_text.insert(tk.END, f"❌ Error\n\n{self.model_response.error}")
            self.content_text.config(fg='#e74c3c')
            self.metrics_label.config(text="Failed")
        elif self.model_response.is_loading():
            # Loading state
            print(f"[DEBUG] Rendering loading state for {self.model_response.model}")
            self.content_text.insert(tk.END, "⏳ Loading...")
            self.content_text.config(fg='#7f8c8d')
            self.metrics_label.config(text="Processing...")
        else:
            # Success state
            print(f"[DEBUG] Rendering success state for {self.model_response.model}, response_len={len(self.model_response.response)}")
            self.content_text.insert(tk.END, self.model_response.response)
            self.content_text.config(fg='#2c3e50')
            
            # Format metrics
            metrics_text = (f"{self.model_response.duration:.1f}s | "
                          f"{self.model_response.tokens} tokens | "
                          f"{self.model_response.tokens_per_second:.1f} tok/s")
            self.metrics_label.config(text=metrics_text)
            print(f"[DEBUG] Metrics set for {self.model_response.model}: {metrics_text}")
        
        # Keep widget in NORMAL state and force multiple updates
        self.content_text.config(state=tk.NORMAL)
        self.content_text.update_idletasks()
        self.content_text.update()
        
        # Force parent container updates
        self.container.update_idletasks()
        self.container.update()
        
        print(f"[DEBUG] render_content completed for {self.model_response.model}")
        print(f"[DEBUG] Text widget content length: {len(self.content_text.get(1.0, tk.END))}")
        
        # Additional debug: check if widget is visible
        try:
            print(f"[DEBUG] Widget visibility - winfo_viewable: {self.content_text.winfo_viewable()}")
            print(f"[DEBUG] Widget geometry - width: {self.content_text.winfo_width()}, height: {self.content_text.winfo_height()}")
        except:
            print(f"[DEBUG] Could not get widget visibility info")
    
    def update_response(self, model_response: ModelResponse):
        """Update the response and re-render"""
        print(f"[DEBUG] ModelResponseColumn.update_response: {model_response.model}, success={model_response.success}")
        self.model_response = model_response
        self.content_text.config(state=tk.NORMAL)
        self.render_content()
        # Keep the widget in NORMAL state so content is visible
        self.content_text.config(state=tk.NORMAL)
        print(f"[DEBUG] ModelResponseColumn.update_response completed for {model_response.model}")
    
    def pack(self, **kwargs):
        """Pack the container"""
        self.container.pack(**kwargs)


class ModelResponseRow:
    """Horizontal row of model response columns"""
    
    def __init__(self, parent, responses: List[ModelResponse]):
        self.parent = parent
        self.responses = responses
        self.columns = []
        
        # Remove fixed height - let it be adaptive
        self.container = tk.Frame(parent, bg='white')
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the horizontal response row UI with horizontal scrolling"""
        print(f"[DEBUG] ModelResponseRow.setup_ui: Creating UI for {len(self.responses)} responses")
        
        # Create horizontal scrollable canvas for multiple models
        self.canvas = tk.Canvas(self.container, bg='white', highlightthickness=0)
        self.h_scrollbar = ttk.Scrollbar(self.container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.columns_frame = tk.Frame(self.canvas, bg='white')
        
        # Configure scrolling
        self.columns_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas_window = self.canvas.create_window((0, 0), window=self.columns_frame, anchor=tk.NW)
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set)
        
        # Bind canvas resize to update frame height
        def on_canvas_configure(event):
            # Update the frame height to match canvas height
            self.canvas.itemconfig(self.canvas_window, height=event.height)
        
        self.canvas.bind('<Configure>', on_canvas_configure)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create columns for each response
        for i, response in enumerate(self.responses):
            print(f"[DEBUG] Creating column {i} for {response.model}")
            column = ModelResponseColumn(self.columns_frame, response)
            # Set fixed width for consistent layout
            column.container.config(width=350)
            column.container.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
            column.container.pack_propagate(False)  # Maintain consistent width
            self.columns.append(column)
            print(f"[DEBUG] Column {i} created and packed")
        
        print(f"[DEBUG] ModelResponseRow.setup_ui completed with {len(self.columns)} columns")
    
    def update_responses(self, responses: List[ModelResponse]):
        """Update all responses"""
        print(f"[DEBUG] ModelResponseRow.update_responses called with {len(responses)} responses")
        print(f"[DEBUG] Current columns count: {len(self.columns)}")
        
        self.responses = responses
        for i, response in enumerate(responses):
            print(f"[DEBUG] Updating column {i}: {response.model}")
            if i < len(self.columns):
                self.columns[i].update_response(response)
                print(f"[DEBUG] Column {i} updated successfully")
            else:
                print(f"[DEBUG] WARNING: No column {i} available for response {response.model}")
        
        # Force comprehensive UI update
        self.force_ui_refresh()
        print(f"[DEBUG] Forced UI update completed")
    
    def force_ui_refresh(self):
        """Force a comprehensive UI refresh"""
        try:
            # Update all child widgets
            for column in self.columns:
                column.content_text.update_idletasks()
                column.content_text.update()
                column.container.update_idletasks()
                column.container.update()
            
            # Update frames
            self.columns_frame.update_idletasks()
            self.columns_frame.update()
            self.container.update_idletasks()
            self.container.update()
            
            print(f"[DEBUG] UI refresh completed successfully")
        except Exception as e:
            print(f"[DEBUG] UI refresh failed: {e}")
    
    def pack(self, **kwargs):
        """Pack the container"""
        self.container.pack(**kwargs)


class ChatInterfaceView:
    """Chat interface view with conversation history and bottom input"""
    
    def __init__(self, parent, on_submit_callback, on_clear_callback):
        self.parent = parent
        self.on_submit_callback = on_submit_callback
        self.on_clear_callback = on_clear_callback
        
        self.container = tk.Frame(parent, bg='white')
        self.conversation_widgets = []
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the chat interface UI"""
        # Top toolbar
        toolbar_frame = tk.Frame(self.container, bg='#f8f9fa', height=50)
        toolbar_frame.pack(fill=tk.X)
        toolbar_frame.pack_propagate(False)
        
        # Clear conversation button
        clear_btn = tk.Button(toolbar_frame, text="🔄 New Conversation",
                            font=('Arial', 10),
                            bg='#f8f9fa', fg='#3498db',
                            relief='flat', cursor='hand2',
                            padx=15, pady=10,
                            command=self.on_clear_callback)
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Progress bar and status (initially hidden)
        self.progress_frame = tk.Frame(toolbar_frame, bg='#f8f9fa')
        
        self.progress_label = tk.Label(self.progress_frame, 
                                      text="Processing...",
                                      font=('Arial', 9),
                                      bg='#f8f9fa', fg='#7f8c8d')
        self.progress_label.pack(side=tk.LEFT, padx=(10, 5))
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, 
                                          mode='indeterminate',
                                          length=200)
        self.progress_bar.pack(side=tk.LEFT)
        
        # Scrollable conversation area
        conversation_container = tk.Frame(self.container, bg='white')
        conversation_container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for scrolling
        self.canvas = tk.Canvas(conversation_container, bg='white',
                               highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(conversation_container, 
                                        orient=tk.VERTICAL,
                                        command=self.canvas.yview)
        
        self.scrollable_frame = tk.Frame(self.canvas, bg='white')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, 
                                 anchor=tk.NW, width=self.canvas.winfo_width())
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind canvas resize to update window width
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Bottom input bar
        self.bottom_input = BottomInputBar(self.container, self.on_submit_callback)
        self.bottom_input.show()
    
    def _on_canvas_configure(self, event):
        """Handle canvas resize"""
        self.canvas.itemconfig(self.canvas.find_withtag("all")[0], 
                              width=event.width)
    
    def add_user_query(self, query: str):
        """Add a user query bubble to the conversation"""
        query_bubble = UserQueryBubble(self.scrollable_frame, query)
        # Don't expand the query bubble, just fill horizontally
        query_bubble.pack(fill=tk.X, pady=(10, 5))
        self.conversation_widgets.append(query_bubble)
        self.auto_scroll_to_bottom()
    
    def add_model_responses(self, responses: List[ModelResponse]):
        """Add model responses row to the conversation"""
        response_row = ModelResponseRow(self.scrollable_frame, responses)
        # Make the response row expand to fill available vertical space
        response_row.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
        self.conversation_widgets.append(response_row)
        self.auto_scroll_to_bottom()
    
    def render_conversation(self, conversation_history: List[ConversationEntry]):
        """Render the full conversation history"""
        # Clear existing widgets
        for widget in self.conversation_widgets:
            if hasattr(widget, 'container'):
                widget.container.destroy()
        self.conversation_widgets = []
        
        # Render each conversation entry
        for entry in conversation_history:
            self.add_user_query(entry.query)
            self.add_model_responses(entry.responses)
    
    def auto_scroll_to_bottom(self):
        """Auto-scroll to the bottom of the conversation"""
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)
    
    def clear_conversation(self):
        """Clear all conversation widgets"""
        for widget in self.conversation_widgets:
            if hasattr(widget, 'container'):
                widget.container.destroy()
        self.conversation_widgets = []
    
    def disable_input(self):
        """Disable input during query processing"""
        self.bottom_input.disable()
    
    def enable_input(self):
        """Enable input after query completion"""
        self.bottom_input.enable()
    
    def clear_input(self):
        """Clear the input field"""
        self.bottom_input.clear_input()
    
    def show_progress(self, message: str = "Processing..."):
        """Show progress bar with message"""
        self.progress_label.config(text=message)
        self.progress_frame.pack(side=tk.RIGHT, padx=10)
        self.progress_bar.start()
    
    def hide_progress(self):
        """Hide progress bar"""
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
    
    def update_progress(self, message: str):
        """Update progress message"""
        self.progress_label.config(text=message)
    
    def show(self):
        """Show the chat interface"""
        self.container.pack(fill=tk.BOTH, expand=True)
    
    def hide(self):
        """Hide the chat interface"""
        self.container.pack_forget()


class ModelSelectionPanel:
    """Modal dialog for model selection"""
    
    def __init__(self, parent, available_models: List[str], 
                 selected_models: List[str], on_confirm_callback):
        self.parent = parent
        self.available_models = available_models
        self.selected_models = selected_models.copy()
        self.on_confirm_callback = on_confirm_callback
        self.model_vars = {}
        
        # Create modal dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Select Models")
        self.dialog.geometry("400x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the model selection dialog UI"""
        # Header
        header_frame = tk.Frame(self.dialog, bg='#f8f9fa', height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, 
                              text="Select models to compare",
                              font=('Arial', 14, 'bold'),
                              bg='#f8f9fa', fg='#2c3e50')
        title_label.pack(pady=20)
        
        # Button row for Select All / Deselect All
        button_row = tk.Frame(self.dialog, bg='white', height=50)
        button_row.pack(fill=tk.X, padx=20, pady=(10, 5))
        button_row.pack_propagate(False)
        
        select_all_btn = tk.Button(button_row, text="Select All",
                                   font=('Arial', 10),
                                   bg='#3498db', fg='white',
                                   relief='flat', cursor='hand2',
                                   padx=15, pady=5,
                                   command=self.select_all)
        select_all_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        deselect_all_btn = tk.Button(button_row, text="Deselect All",
                                     font=('Arial', 10),
                                     bg='#95a5a6', fg='white',
                                     relief='flat', cursor='hand2',
                                     padx=15, pady=5,
                                     command=self.deselect_all)
        deselect_all_btn.pack(side=tk.LEFT)
        
        # Model count label
        self.count_label = tk.Label(button_row,
                                   text=f"{len(self.selected_models)} selected",
                                   font=('Arial', 10),
                                   bg='white', fg='#7f8c8d')
        self.count_label.pack(side=tk.RIGHT)
        
        # Scrollable checkbox list
        list_container = tk.Frame(self.dialog, bg='white')
        list_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        canvas = tk.Canvas(list_container, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL,
                                 command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create checkboxes for each model
        for model in self.available_models:
            var = tk.BooleanVar(value=model in self.selected_models)
            self.model_vars[model] = var
            
            cb = ttk.Checkbutton(scrollable_frame, text=model, 
                               variable=var,
                               command=self.update_count)
            cb.pack(anchor=tk.W, pady=2, padx=5)
        
        # Confirm button at bottom
        confirm_frame = tk.Frame(self.dialog, bg='white', height=70)
        confirm_frame.pack(fill=tk.X, padx=20, pady=10)
        confirm_frame.pack_propagate(False)
        
        confirm_btn = tk.Button(confirm_frame, text="Confirm Selection",
                               font=('Arial', 11, 'bold'),
                               bg='#27ae60', fg='white',
                               relief='flat', cursor='hand2',
                               padx=30, pady=10,
                               command=self.confirm)
        confirm_btn.pack(expand=True)
    
    def select_all(self):
        """Select all models"""
        for var in self.model_vars.values():
            var.set(True)
        self.update_count()
    
    def deselect_all(self):
        """Deselect all models"""
        for var in self.model_vars.values():
            var.set(False)
        self.update_count()
    
    def update_count(self):
        """Update the selected count label"""
        count = sum(1 for var in self.model_vars.values() if var.get())
        self.count_label.config(text=f"{count} selected")
    
    def confirm(self):
        """Confirm selection and close dialog"""
        # Get selected models
        selected = [model for model, var in self.model_vars.items() 
                   if var.get()]
        
        # Validate at least one model is selected
        if not selected:
            messagebox.showwarning("Warning", 
                                 "Please select at least one model",
                                 parent=self.dialog)
            return
        
        # Update selected models and call callback
        self.selected_models = selected
        self.on_confirm_callback(selected)
        self.dialog.destroy()
    
    def show(self):
        """Show the dialog (modal)"""
        self.dialog.wait_window()


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
    """Main GUI application with chat-based interface"""
    
    def __init__(self):
        # Core properties
        self.base_url = "http://localhost:11434"
        self.output_dir = Path("interactive_test_results")
        self.output_dir.mkdir(exist_ok=True)
        self.models = []
        
        # State management
        self.state_manager = StateManager()
        
        # Setup GUI
        self.setup_gui()
        self.load_models()
    
    def setup_gui(self):
        """Setup the main GUI"""
        self.root = tk.Tk()
        self.root.title("AI Model Comparison - Find the best AI for you")
        # Remove hardcoded window size - let it be resizable and adaptive
        self.root.state('zoomed')  # Start maximized on Windows, or use normal size on other platforms
        self.root.configure(bg='#f8f9fa')
        
        # Main container
        self.main_container = tk.Frame(self.root, bg='#f8f9fa')
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create views
        self.landing_view = LandingPageView(
            self.main_container,
            on_submit_callback=self.handle_landing_submit,
            on_model_select_callback=self.show_model_selection
        )
        
        self.chat_view = ChatInterfaceView(
            self.main_container,
            on_submit_callback=self.handle_chat_submit,
            on_clear_callback=self.handle_clear_conversation
        )
        
        # Show landing view initially
        self.switch_to_landing()
    
    def switch_to_landing(self):
        """Switch to landing page view"""
        self.chat_view.hide()
        self.landing_view.show()
        self.state_manager.transition_to_landing()
    
    def switch_to_chat(self):
        """Switch to chat interface view"""
        self.landing_view.hide()
        self.chat_view.show()
        self.state_manager.transition_to_chat()
    
    def show_model_selection(self):
        """Show model selection dropdown"""
        if not self.models:
            messagebox.showwarning("Warning", "No models available. Please wait for models to load.")
            self.landing_view.hide_dropdown()
            return
        
        # Populate the dropdown in landing view
        self.landing_view.populate_dropdown(
            self.models,
            self.state_manager.get_selected_models(),
            self.on_models_selected
        )
    
    def on_models_selected(self, selected_models: List[str]):
        """Handle model selection change"""
        self.state_manager.set_selected_models(selected_models)
        self.landing_view.update_model_count(len(selected_models))
    
    def handle_landing_submit(self, query: str):
        """Handle query submission from landing page"""
        # Validate models are selected
        if not self.state_manager.get_selected_models():
            messagebox.showwarning("Warning", "Please select at least one model first.")
            return
        
        # Switch to chat view
        self.switch_to_chat()
        
        # Execute query
        self.execute_query(query)
    
    def handle_chat_submit(self, query: str):
        """Handle query submission from chat interface"""
        self.execute_query(query)
    
    def handle_clear_conversation(self):
        """Handle clear conversation request"""
        self.state_manager.clear_conversation()
        self.chat_view.clear_conversation()
        self.switch_to_landing()
    
    def execute_query(self, query: str):
        """Execute a query against selected models"""
        selected_models = self.state_manager.get_selected_models()
        
        print(f"[DEBUG] Executing query: {query}")
        print(f"[DEBUG] Selected models: {selected_models}")
        
        if not selected_models:
            messagebox.showwarning("Warning", "No models selected")
            return
        
        # Add query bubble to chat
        self.chat_view.add_user_query(query)
        
        # Create loading responses
        loading_responses = [
            ModelResponse(model=model, success=False, error="", response="Loading...")
            for model in selected_models
        ]
        print(f"[DEBUG] Created {len(loading_responses)} loading responses")
        self.chat_view.add_model_responses(loading_responses)
        
        # Disable input and show progress
        self.chat_view.disable_input()
        self.chat_view.show_progress(f"Querying {len(selected_models)} models...")
        
        # Execute queries in background thread
        def query_thread():
            try:
                print(f"[DEBUG] Starting parallel query...")
                self.root.after(0, lambda: self.chat_view.update_progress("Processing queries..."))
                
                results = self.parallel_query_models(selected_models, query)
                print(f"[DEBUG] Query complete. Results: {len(results)} responses")
                for r in results:
                    print(f"[DEBUG]   - {r.get('model')}: success={r.get('success')}, response_len={len(r.get('response', ''))}")
                
                # Add to conversation history
                self.root.after(0, lambda: self.chat_view.update_progress("Updating interface..."))
                self.state_manager.add_conversation_entry(query, results)
                
                # Update UI with results
                print(f"[DEBUG] Updating UI with results...")
                self.root.after(0, lambda: self.update_latest_responses(results))
            except Exception as e:
                print(f"[DEBUG] Query failed with error: {e}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror("Error", f"Query failed: {str(e)}"))
            finally:
                self.root.after(0, self.query_complete)
        
        threading.Thread(target=query_thread, daemon=True).start()
    
    def parallel_query_models(self, models: List[str], prompt: str) -> List[Dict[str, Any]]:
        """Query multiple models in parallel"""
        results = []
        
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
                except Exception as e:
                    results.append({
                        "model": model,
                        "success": False,
                        "error": str(e)
                    })
        
        # Sort by model name
        results.sort(key=lambda x: x["model"])
        return results
    
    def update_latest_responses(self, results: List[Dict[str, Any]]):
        """Update the latest response row with actual results"""
        print(f"[DEBUG] update_latest_responses called with {len(results)} results")
        
        # Convert to ModelResponse objects
        model_responses = []
        for resp in results:
            mr = ModelResponse(
                model=resp.get("model", ""),
                success=resp.get("success", False),
                response=resp.get("response", ""),
                duration=resp.get("duration", 0.0),
                tokens=resp.get("tokens", 0),
                tokens_per_second=resp.get("tokens_per_second", 0.0),
                error=resp.get("error", "")
            )
            model_responses.append(mr)
            print(f"[DEBUG]   Created ModelResponse: {mr.model}, success={mr.success}, response_len={len(mr.response)}, error={mr.error}")
        
        # Update the last response row
        print(f"[DEBUG] Conversation widgets count: {len(self.chat_view.conversation_widgets)}")
        if self.chat_view.conversation_widgets:
            last_widget = self.chat_view.conversation_widgets[-1]
            print(f"[DEBUG] Last widget type: {type(last_widget)}")
            if isinstance(last_widget, ModelResponseRow):
                print(f"[DEBUG] Updating ModelResponseRow with {len(model_responses)} responses")
                last_widget.update_responses(model_responses)
            else:
                print(f"[DEBUG] WARNING: Last widget is not ModelResponseRow!")
        else:
            print(f"[DEBUG] WARNING: No conversation widgets found!")
    

    
    def query_complete(self):
        """Handle query completion"""
        self.chat_view.hide_progress()
        self.chat_view.enable_input()
        self.chat_view.clear_input()
    
    def load_models(self):
        """Load available models from Ollama"""
        def load_thread():
            try:
                self.models = get_available_models(self.base_url)
                if self.models:
                    # Select all models by default
                    self.root.after(0, lambda: self.state_manager.set_selected_models(self.models))
                    self.root.after(0, lambda: self.landing_view.update_model_count(len(self.models)))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "No models found. Please check Ollama service."))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load models: {str(e)}"))
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def save_conversation(self):
        """Save the current conversation to a markdown file"""
        conversation_history = self.state_manager.get_conversation_history()
        
        if not conversation_history:
            messagebox.showinfo("Info", "No conversation to save")
            return
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.md"
        filepath = self.output_dir / filename
        
        # Write conversation to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# AI Model Comparison - Conversation\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Models**: {', '.join(self.state_manager.get_selected_models())}\n\n")
            f.write(f"---\n\n")
            
            # Write each conversation entry
            for entry in conversation_history:
                f.write(entry.to_markdown())
                f.write(f"\n---\n\n")
        
        messagebox.showinfo("Success", f"Conversation saved to:\n{filepath}")
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


# === OLD CODE REMOVED ===
# All old methods from the previous implementation have been removed
# The new chat-based interface uses the component classes defined above


# Skipping to the correct main function below
# (Old methods removed - approximately 1100 lines)


# PLACEHOLDER - will be replaced with correct main function


def main():
    """主函数"""
    app = ModelComparisonGUI()
    app.run()


if __name__ == "__main__":
    main()