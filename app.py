#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Model Comparison Tool with GUI
Interactive model comparison tool - single interface input, all models respond in parallel
"""

import concurrent.futures
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

from ollama_client import OllamaClient, get_available_models, query_model
from ollama_utils import create_timestamp


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
        title_label = tk.Label(center_frame, text="AI Model Comparison Tool",
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
        self.input_entry.insert(0, "Enter your question...")
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
        if self.input_entry.get() == "Enter your question...":
            self.input_entry.delete(0, tk.END)
            self.input_entry.config(fg='#2c3e50')
    
    def _on_entry_focus_out(self, event):
        """Restore placeholder text if empty"""
        if not self.input_entry.get():
            self.input_entry.insert(0, "Enter your question...")
            self.input_entry.config(fg='#7f8c8d')
    
    def _on_submit(self):
        """Handle submit button click"""
        query = self.input_entry.get().strip()
        if query and query != "Enter your question...":
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
        return text if text != "Enter your question..." else ""
    
    def clear_input(self):
        """Clear the input field"""
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, "Enter your question...")
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
        self.input_entry.insert(0, "Ask follow-up question...")
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
        if self.input_entry.get() == "Ask follow-up question...":
            self.input_entry.delete(0, tk.END)
            self.input_entry.config(fg='#2c3e50')
    
    def _on_entry_focus_out(self, event):
        """Restore placeholder text if empty"""
        if not self.input_entry.get():
            self.input_entry.insert(0, "Ask follow-up question...")
            self.input_entry.config(fg='#7f8c8d')
    
    def _on_submit(self):
        """Handle submit button click"""
        if not self.is_enabled:
            return
        
        query = self.input_entry.get().strip()
        if query and query != "Ask follow-up question...":
            self.on_submit_callback(query)
    
    def get_input_text(self) -> str:
        """Get the current input text"""
        text = self.input_entry.get().strip()
        return text if text != "Ask follow-up question..." else ""
    
    def clear_input(self):
        """Clear the input field"""
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, "Ask follow-up question...")
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
        self.min_width = 300
        self.max_width = 600
        self.preferred_width = 400
        
        self.container = tk.Frame(parent, bg='white', 
                                 relief='solid', borderwidth=1,
                                 width=self.preferred_width,
                                 height=400)  # Set minimum height
        self.container.pack_propagate(False)  # Maintain consistent width and height
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the column UI with adaptive sizing"""
        # Header with model name - more compact
        header_frame = tk.Frame(self.container, bg='#f8f9fa', height=35)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        model_label = tk.Label(header_frame, 
                              text=self.model_response.model,
                              font=('Arial', 10, 'bold'),
                              bg='#f8f9fa', fg='#2c3e50',
                              padx=8, wraplength=350)
        model_label.pack(side=tk.LEFT, pady=8)
        
        # Content area with better text handling
        self.content_text = tk.Text(
            self.container,
            font=('Arial', 9),
            bg='white', fg='#2c3e50',
            wrap=tk.WORD,
            relief='flat',
            padx=8, pady=8,
            width=50,  # Increased width for better text display
            height=15  # Set reasonable minimum height
        )
        self.content_text.pack(fill=tk.BOTH, expand=True, padx=3, pady=(3, 0))
        
        # Footer with metrics - more compact
        footer_frame = tk.Frame(self.container, bg='#f8f9fa', height=28)
        footer_frame.pack(fill=tk.X, pady=(0, 3))
        footer_frame.pack_propagate(False)
        
        self.metrics_label = tk.Label(footer_frame,
                                     text="",
                                     font=('Arial', 8),
                                     bg='#f8f9fa', fg='#7f8c8d',
                                     padx=8, wraplength=350)
        self.metrics_label.pack(side=tk.LEFT, pady=4)
        
        # Render content based on state
        self.render_content()
    
    def render_content(self):
        """Render content based on response state with improved text formatting"""
        # Ensure widget is in normal state before modifying
        self.content_text.config(state=tk.NORMAL)
        self.content_text.delete(1.0, tk.END)
        
        if self.model_response.has_error():
            # Error state
            self.content_text.insert(tk.END, f"❌ Error\n\n{self.model_response.error}")
            self.content_text.config(fg='#e74c3c')
            self.metrics_label.config(text="Failed")
        elif self.model_response.is_loading():
            # Loading state
            self.content_text.insert(tk.END, "⏳ Loading...")
            self.content_text.config(fg='#7f8c8d')
            self.metrics_label.config(text="Processing...")
        else:
            # Success state - format text for better readability
            response_text = self.model_response.response.strip()
            if response_text:
                # Insert response with proper formatting
                self.content_text.insert(tk.END, response_text)
            else:
                self.content_text.insert(tk.END, "No response received")
            
            self.content_text.config(fg='#2c3e50')
            
            # Format metrics more compactly
            if self.model_response.duration > 0 and self.model_response.tokens > 0:
                metrics_text = (f"{self.model_response.duration:.1f}s | "
                              f"{self.model_response.tokens} tokens | "
                              f"{self.model_response.tokens_per_second:.1f} tok/s")
            else:
                metrics_text = "Completed"
            self.metrics_label.config(text=metrics_text)
        
        # Configure text widget for better display
        self.content_text.config(state=tk.NORMAL)
        
        # Auto-resize text widget based on content
        self.auto_resize_text_widget()
        
        # Update display
        self.content_text.update_idletasks()
        self.container.update_idletasks()
    
    def auto_resize_text_widget(self):
        """Automatically resize text widget to fill available space"""
        try:
            # Get the line count of the content
            line_count = int(self.content_text.index('end-1c').split('.')[0])
            
            # Calculate appropriate height to fill the container
            container_height = self.container.winfo_height()
            available_height = container_height - 70  # Account for header and footer
            
            # Calculate lines that fit in available space
            line_height = 14  # Approximate line height in pixels
            max_lines = max(10, available_height // line_height)
            
            # Use the smaller of content lines or available space
            content_height = min(max_lines, max(10, line_count + 2))
            
            # Update text widget height
            self.content_text.config(height=content_height)
            
        except Exception as e:
            # Fallback to reasonable height if calculation fails
            self.content_text.config(height=15)
    
    def update_response(self, model_response: ModelResponse):
        """Update the response and re-render"""
        self.model_response = model_response
        self.render_content()
    
    def resize_column(self, new_width: int):
        """Resize column to new width"""
        self.preferred_width = max(self.min_width, min(self.max_width, new_width))
        self.container.config(width=self.preferred_width)
    
    def pack(self, **kwargs):
        """Pack the container"""
        self.container.pack(**kwargs)


class ModelResponseRow:
    """Horizontal row of model response columns with adaptive layout"""
    
    def __init__(self, parent, responses: List[ModelResponse]):
        self.parent = parent
        self.responses = responses
        self.columns = []
        self.column_width = 400  # Default column width
        self.min_columns_visible = 2  # Minimum columns to show without scrolling
        
        self.container = tk.Frame(parent, bg='white')
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the horizontal response row UI with adaptive scrolling"""
        # Create horizontal scrollable canvas for multiple models
        self.canvas = tk.Canvas(self.container, bg='white', highlightthickness=0)
        self.h_scrollbar = ttk.Scrollbar(self.container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.columns_frame = tk.Frame(self.canvas, bg='white')
        
        # Configure container to expand and fill available space
        self.container.pack_propagate(False)
        
        # Configure scrolling
        self.columns_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas_window = self.canvas.create_window((0, 0), window=self.columns_frame, anchor=tk.NW)
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set)
        
        # Bind canvas resize to update frame height and column widths
        def on_canvas_configure(event):
            # Update the frame height to match canvas height - use full height
            canvas_height = max(400, event.height)  # Minimum 400px height
            self.canvas.itemconfig(self.canvas_window, height=canvas_height)
            
            # Calculate optimal column width based on available space
            available_width = event.width
            num_columns = len(self.columns)
            
            if num_columns > 0:
                # Calculate optimal width per column
                optimal_width = max(300, min(600, available_width // min(num_columns, self.min_columns_visible)))
                self.update_column_widths(optimal_width)
        
        self.canvas.bind('<Configure>', on_canvas_configure)
        
        # Pack canvas and scrollbar with full expansion
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create columns for each response
        self.create_columns()
    
    def create_columns(self):
        """Create columns for all responses"""
        for i, response in enumerate(self.responses):
            column = ModelResponseColumn(self.columns_frame, response)
            # Set initial width and height
            column.container.config(width=self.column_width, height=400)
            column.container.pack(side=tk.LEFT, padx=3, pady=3, fill=tk.BOTH, expand=True)
            column.container.pack_propagate(False)
            self.columns.append(column)
    
    def update_column_widths(self, new_width: int):
        """Update all column widths to new value"""
        if new_width != self.column_width:
            self.column_width = new_width
            for column in self.columns:
                column.resize_column(new_width)
    
    def update_responses(self, responses: List[ModelResponse]):
        """Update all responses"""
        self.responses = responses
        
        # Update existing columns
        for i, response in enumerate(responses):
            if i < len(self.columns):
                self.columns[i].update_response(response)
        
        # If we have more responses than columns, create new columns
        if len(responses) > len(self.columns):
            for i in range(len(self.columns), len(responses)):
                column = ModelResponseColumn(self.columns_frame, responses[i])
                column.container.config(width=self.column_width, height=400)
                column.container.pack(side=tk.LEFT, padx=3, pady=3, fill=tk.BOTH, expand=True)
                column.container.pack_propagate(False)
                self.columns.append(column)
        
        # Update UI
        self.refresh_layout()
    
    def refresh_layout(self):
        """Refresh the layout after updates"""
        try:
            # Update all child widgets
            for column in self.columns:
                column.content_text.update_idletasks()
                column.container.update_idletasks()
            
            # Update frames
            self.columns_frame.update_idletasks()
            self.container.update_idletasks()
            
            # Update canvas scroll region and force full height usage
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            # Ensure columns use full height
            canvas_height = self.canvas.winfo_height()
            if canvas_height > 400:
                self.canvas.itemconfig(self.canvas_window, height=canvas_height)
            
        except Exception as e:
            print(f"[DEBUG] Layout refresh failed: {e}")
    
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
        """Setup the chat interface UI with responsive design"""
        # Top toolbar with flexible layout
        toolbar_frame = tk.Frame(self.container, bg='#f8f9fa', height=45)
        toolbar_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        toolbar_frame.pack_propagate(False)
        
        # Clear conversation button
        clear_btn = tk.Button(toolbar_frame, text="🔄 New Conversation",
                            font=('Arial', 9),
                            bg='#f8f9fa', fg='#3498db',
                            relief='flat', cursor='hand2',
                            padx=12, pady=8,
                            command=self.on_clear_callback)
        clear_btn.pack(side=tk.LEFT, padx=8)
        
        # Progress bar and status (initially hidden)
        self.progress_frame = tk.Frame(toolbar_frame, bg='#f8f9fa')
        
        self.progress_label = tk.Label(self.progress_frame, 
                                      text="Processing...",
                                      font=('Arial', 8),
                                      bg='#f8f9fa', fg='#7f8c8d')
        self.progress_label.pack(side=tk.LEFT, padx=(8, 4))
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, 
                                          mode='indeterminate',
                                          length=180)
        self.progress_bar.pack(side=tk.LEFT)
        
        # Scrollable conversation area with minimal padding
        conversation_container = tk.Frame(self.container, bg='white')
        conversation_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Canvas for scrolling with improved configuration
        self.canvas = tk.Canvas(conversation_container, bg='white',
                               highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(conversation_container, 
                                        orient=tk.VERTICAL,
                                        command=self.canvas.yview)
        
        self.scrollable_frame = tk.Frame(self.canvas, bg='white')
        
        # Configure scrolling with dynamic width
        self.scrollable_frame.bind(
            "<Configure>",
            self._on_scrollable_configure
        )
        
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, 
                                 anchor=tk.NW)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)
        
        # Pack with full expansion to use all available space
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind canvas resize to update window width
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Bottom input bar
        self.bottom_input = BottomInputBar(self.container, self.on_submit_callback)
        self.bottom_input.show()
    
    def _on_canvas_configure(self, event):
        """Handle canvas resize"""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def _on_scrollable_configure(self, event):
        """Handle scrollable frame configuration"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def add_user_query(self, query: str):
        """Add a user query bubble to the conversation"""
        query_bubble = UserQueryBubble(self.scrollable_frame, query)
        # Pack with minimal padding to save space
        query_bubble.pack(fill=tk.X, pady=(5, 2))
        self.conversation_widgets.append(query_bubble)
        self.auto_scroll_to_bottom()
    
    def add_model_responses(self, responses: List[ModelResponse]):
        """Add model responses row to the conversation"""
        response_row = ModelResponseRow(self.scrollable_frame, responses)
        # Pack with full expansion and minimal padding to maximize space usage
        response_row.pack(fill=tk.BOTH, expand=True, padx=2, pady=(2, 5))
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


# Model querying functions moved to ollama_client.py


def save_results_to_file(
    results: List[Dict[str, Any]],
    prompt: str,
    output_dir: Path,
    filename: str = None
) -> None:
    """Save results to a markdown file."""
    if filename is None:
        timestamp = create_timestamp()
        filename = f"app_{timestamp}.md"
    
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
        self.client = OllamaClient(self.base_url)
        self.output_dir = Path("app_results")
        self.output_dir.mkdir(exist_ok=True)
        self.models = []
        
        # State management
        self.state_manager = StateManager()
        
        # Setup GUI
        self.setup_gui()
        self.load_models()
    
    def setup_gui(self):
        """Setup the main GUI with responsive design"""
        self.root = tk.Tk()
        self.root.title("AI Model Comparison Tool")
        
        # Set minimum window size and start maximized
        self.root.minsize(800, 600)
        self.root.state('zoomed')  # Start maximized on Windows
        self.root.configure(bg='#f8f9fa')
        
        # Handle window resize events
        self.root.bind('<Configure>', self.on_window_resize)
        
        # Main container with minimal padding to maximize space
        self.main_container = tk.Frame(self.root, bg='#f8f9fa')
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
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
    
    def on_window_resize(self, event):
        """Handle window resize events"""
        if event.widget == self.root:
            # Window was resized, update layouts
            self.root.after_idle(self.update_layouts)
    
    def update_layouts(self):
        """Update all layouts after window resize"""
        try:
            if hasattr(self, 'chat_view') and self.state_manager.is_chat_view():
                # Update chat view layouts
                for widget in self.chat_view.conversation_widgets:
                    if hasattr(widget, 'refresh_layout'):
                        widget.refresh_layout()
                
                # Force conversation area to use full height
                self.chat_view.canvas.update_idletasks()
                canvas_height = self.chat_view.canvas.winfo_height()
                if canvas_height > 400:
                    self.chat_view.canvas.itemconfig(self.chat_view.canvas_window, height=canvas_height)
                    
        except Exception as e:
            print(f"[DEBUG] Layout update failed: {e}")
    
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
                executor.submit(self.client.query_model, model, prompt): model
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
                self.models = self.client.get_available_models()
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
        timestamp = create_timestamp()
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