# Design Document: Interactive UI Optimization

## Overview

This design document outlines the transformation of the interactive_test.py UI from a left-panel input interface to a modern chat-based interface. The redesign prioritizes functional correctness and proper layout for the MVP, with visual polish deferred to future iterations.

### Design Goals

1. **Chat-First Experience**: Transform the interface to match familiar AI chat patterns (ChatGPT, Claude, etc.)
2. **Improved Comparison**: Enable easier side-by-side model comparison through horizontal response layout
3. **Conversational Flow**: Support follow-up questions in a natural conversation thread
4. **MVP Focus**: Prioritize working functionality over visual aesthetics
5. **Maintain Core Features**: Preserve existing parallel execution, result saving, and model selection capabilities

### Key Design Decisions

**Decision 1: Single-Line Bottom Input**
- **Rationale**: Modern chat interfaces universally place input at the bottom, creating a familiar UX pattern. This also maximizes vertical space for viewing responses.
- **Trade-off**: Multi-line input requires Shift+Enter, which may not be immediately discoverable.

**Decision 2: Landing Page with Model Selection**
- **Rationale**: Separates model configuration from the chat experience, reducing visual clutter and providing a clear starting point.
- **Trade-off**: Adds one extra step before first query, but improves overall workflow clarity.

**Decision 3: Horizontal Response Layout**
- **Rationale**: Side-by-side comparison is more effective than vertical stacking for comparing similar content.
- **Trade-off**: Requires horizontal scrolling for >3 models, but this is acceptable for the comparison use case.

**Decision 4: Tkinter-Based Implementation**
- **Rationale**: Maintains compatibility with existing codebase and Windows environment without introducing new dependencies.
- **Trade-off**: Limited styling capabilities compared to web-based solutions, but sufficient for MVP.

## Architecture

### Component Structure

```
ModelComparisonGUI (Main Application)
├── LandingPageView (Initial State)
│   ├── ModelSelectionPanel
│   ├── CenterInputBox
│   └── WelcomeContent
├── ChatInterfaceView (Active State)
│   ├── ConversationScrollArea
│   │   ├── UserQueryBubble (right-aligned)
│   │   └── ModelResponseRow (horizontal layout)
│   │       ├── ModelResponseColumn (per model)
│   │       │   ├── ModelHeader
│   │       │   ├── ResponseContent (scrollable)
│   │       │   └── MetricsFooter
│   │       └── ... (additional columns)
│   └── BottomInputBar
│       ├── AttachmentButton
│       ├── InputField
│       └── SendButton
└── StateManager
    ├── ConversationHistory
    ├── SelectedModels
    └── CurrentView
```

### State Management

The application operates in two primary states:

1. **Landing State**: No conversation history, displays landing page
2. **Chat State**: Has conversation history, displays chat interface

State transitions:
- Landing → Chat: When first query is submitted
- Chat → Landing: When conversation is cleared
- Chat → Chat: When follow-up queries are submitted

### Data Flow

```
User Input → StateManager → Query Executor (Parallel) → Response Aggregator → UI Renderer
     ↓                                                           ↓
Model Selection                                          Conversation History
```

## Components and Interfaces

### 1. LandingPageView

**Purpose**: Provide initial interface for model selection and first query input.

**Layout**:
```
┌─────────────────────────────────────────┐
|                                         |
|                                         |
│         [Model Icons]                   │
│                                         │
│    Find the best AI for you             │
│    Compare responses from multiple      │
│    models side-by-side                  │
│                                         │
│  ┌────────────────────────────────────┐ │
│  │ Ask anything...              [📎][→]││
│  └────────────────────────────────────┘ │
│  [Select Models Button]                 |      
|                                         |
|                                         |
└─────────────────────────────────────────┘
```

**Key Methods**:
- `render_landing_page()`: Display landing page content
- `show_model_selector()`: Open model selection modal
- `handle_first_query()`: Transition to chat interface

**Styling**:
- Centered content with max-width constraint
- Large, prominent input box
- Minimal visual elements (MVP focus)

### 2. ChatInterfaceView

**Purpose**: Display conversation history and enable follow-up queries.

**Layout**:
```
┌─────────────────────────────────────────┐
│  [Scrollable Content Area]              │
│                                          │
│                    ┌──────────────────┐ │
│                    │ User Query       │ │
│                    └──────────────────┘ │
│                                          │
│  ┌──────────┬──────────┬──────────┐    │
│  │ Model A  │ Model B  │ Model C  │    │
│  ├──────────┼──────────┼──────────┤    │
│  │ Response │ Response │ Response │    │
│  │ ...      │ ...      │ ...      │    │
│  │ [Scroll] │ [Scroll] │ [Scroll] │    │
│  │          │          │          │    │
│  │ 2.5s     │ 3.1s     │ 2.8s     │    │
│  │ 45 tok/s │ 38 tok/s │ 42 tok/s │    │
│  └──────────┴──────────┴──────────┘    │
│                                          │
├─────────────────────────────────────────┤
│  [📎] Ask followup...              [→]  │
└─────────────────────────────────────────┘
```

**Key Methods**:
- `render_conversation()`: Display all queries and responses
- `add_user_query()`: Append new query to conversation
- `add_model_responses()`: Append response row to conversation
- `auto_scroll_to_bottom()`: Scroll to latest content

**Styling**:
- Fixed bottom input bar (always visible)
- Scrollable content area above input
- Clear visual separation between queries and responses

### 3. UserQueryBubble

**Purpose**: Display user's query in chat-like format.

**Properties**:
- Right-aligned layout
- Distinct background color (light blue/gray)
- Rounded corners
- Timestamp (optional for MVP)

**Implementation**:
```python
class UserQueryBubble:
    def __init__(self, parent, query_text):
        self.frame = tk.Frame(parent, bg='white')
        self.bubble = tk.Frame(self.frame, bg='#e3f2fd', 
                               relief='flat', borderwidth=1)
        self.bubble.pack(side=tk.RIGHT, padx=20, pady=10)
        
        self.label = tk.Label(self.bubble, text=query_text,
                             bg='#e3f2fd', fg='#1976d2',
                             font=('Arial', 11), wraplength=400,
                             justify=tk.RIGHT, padx=15, pady=10)
        self.label.pack()
```

### 4. ModelResponseRow

**Purpose**: Display responses from all models in horizontal layout.

**Layout Strategy**:
- Equal-width columns for each model
- Horizontal scrollbar if >3 models
- Independent vertical scrolling per column

**Implementation**:
```python
class ModelResponseRow:
    def __init__(self, parent, responses):
        self.frame = tk.Frame(parent, bg='white')
        
        # Create horizontal scroll container
        self.canvas = tk.Canvas(self.frame, bg='white', 
                               highlightthickness=0)
        self.h_scrollbar = ttk.Scrollbar(self.frame, 
                                        orient='horizontal',
                                        command=self.canvas.xview)
        
        self.columns_frame = tk.Frame(self.canvas, bg='white')
        
        # Create column for each model
        for response in responses:
            col = ModelResponseColumn(self.columns_frame, response)
            col.pack(side=tk.LEFT, fill=tk.BOTH, 
                    expand=True, padx=5)
```

### 5. ModelResponseColumn

**Purpose**: Display single model's response with metrics.

**Structure**:
- Header: Model name
- Content: Response text (scrollable)
- Footer: Performance metrics

**States**:
- Loading: Show spinner/progress indicator
- Success: Show response and metrics
- Error: Show error message

**Implementation**:
```python
class ModelResponseColumn:
    def __init__(self, parent, response_data):
        self.frame = tk.Frame(parent, bg='white', 
                             relief='solid', borderwidth=1,
                             width=350)
        self.frame.pack_propagate(False)
        
        # Header
        self.header = tk.Label(self.frame, 
                              text=response_data['model'],
                              bg='#f5f5f5', font=('Arial', 11, 'bold'))
        self.header.pack(fill=tk.X, pady=(0, 5))
        
        # Content (scrollable)
        self.content = scrolledtext.ScrolledText(
            self.frame, wrap=tk.WORD, font=('Arial', 10),
            bg='white', relief='flat')
        self.content.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Metrics footer
        self.metrics = tk.Label(self.frame,
                               text=self.format_metrics(response_data),
                               bg='#fafafa', font=('Arial', 9),
                               fg='#666')
        self.metrics.pack(fill=tk.X)
    
    def format_metrics(self, data):
        return f"{data['duration']:.1f}s | " \
               f"{data['tokens']} tokens | " \
               f"{data['tokens_per_second']:.1f} tok/s"
```

### 6. BottomInputBar

**Purpose**: Provide consistent input interface across chat session.

**Features**:
- Single-line input (expandable with Shift+Enter)
- Attachment button (left)
- Send button (right)
- Placeholder text
- Auto-focus after query completion

**Implementation**:
```python
class BottomInputBar:
    def __init__(self, parent, on_submit_callback):
        self.frame = tk.Frame(parent, bg='white', 
                             relief='solid', borderwidth=1,
                             height=60)
        self.frame.pack_propagate(False)
        
        # Attachment button
        self.attach_btn = tk.Button(self.frame, text='📎',
                                   font=('Arial', 14),
                                   relief='flat', bg='white')
        self.attach_btn.pack(side=tk.LEFT, padx=10)
        
        # Input field
        self.input = tk.Entry(self.frame, font=('Arial', 11),
                             relief='flat', bg='#f8f9fa')
        self.input.pack(side=tk.LEFT, fill=tk.BOTH, 
                       expand=True, padx=5)
        self.input.bind('<Return>', self.handle_submit)
        
        # Send button
        self.send_btn = tk.Button(self.frame, text='→',
                                 font=('Arial', 16),
                                 relief='flat', bg='#3498db',
                                 fg='white', command=on_submit_callback)
        self.send_btn.pack(side=tk.RIGHT, padx=10)
```

### 7. ModelSelectionPanel

**Purpose**: Allow users to select which models to compare.

**Display Options**:
- Modal dialog (overlay)
- Side panel (slide-in)

**MVP Decision**: Use modal dialog for simplicity.

**Features**:
- Checkbox list of available models
- Select All / Deselect All buttons
- Model count display
- Validation (prevent empty selection)

**Implementation**:
```python
class ModelSelectionPanel:
    def __init__(self, parent, available_models, 
                 current_selection):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Select Models")
        self.dialog.geometry("400x600")
        
        # Header
        header = tk.Label(self.dialog, 
                         text="Select models to compare",
                         font=('Arial', 12, 'bold'))
        header.pack(pady=10)
        
        # Select all/none buttons
        btn_frame = tk.Frame(self.dialog)
        btn_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Button(btn_frame, text="Select All",
                 command=self.select_all).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Deselect All",
                 command=self.deselect_all).pack(side=tk.LEFT, padx=5)
        
        # Model list (scrollable)
        list_frame = tk.Frame(self.dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        
        self.canvas = tk.Canvas(list_frame)
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical',
                                 command=self.canvas.yview)
        
        self.model_vars = {}
        for model in available_models:
            var = tk.BooleanVar(value=model in current_selection)
            self.model_vars[model] = var
            cb = ttk.Checkbutton(self.canvas, text=model, 
                               variable=var)
            cb.pack(anchor=tk.W, pady=2)
        
        # Confirm button
        tk.Button(self.dialog, text="Confirm",
                 command=self.confirm).pack(pady=10)
```

### 8. StateManager

**Purpose**: Manage application state and conversation history.

**Responsibilities**:
- Track current view (landing vs chat)
- Store conversation history
- Manage selected models
- Coordinate view transitions

**Data Structures**:
```python
class ConversationEntry:
    query: str
    timestamp: datetime
    responses: List[ModelResponse]

class ModelResponse:
    model: str
    success: bool
    response: str
    duration: float
    tokens: int
    tokens_per_second: float
    error: Optional[str]

class StateManager:
    current_view: str  # 'landing' or 'chat'
    conversation_history: List[ConversationEntry]
    selected_models: List[str]
    
    def add_conversation_entry(self, entry: ConversationEntry)
    def clear_conversation(self)
    def get_conversation_history(self) -> List[ConversationEntry]
    def set_selected_models(self, models: List[str])
    def transition_to_chat(self)
    def transition_to_landing(self)
```

## Data Models

### ConversationHistory

```python
@dataclass
class ConversationHistory:
    entries: List[ConversationEntry] = field(default_factory=list)
    
    def add_entry(self, query: str, responses: List[Dict]):
        entry = ConversationEntry(
            query=query,
            timestamp=datetime.now(),
            responses=[ModelResponse(**r) for r in responses]
        )
        self.entries.append(entry)
    
    def clear(self):
        self.entries.clear()
    
    def to_markdown(self) -> str:
        """Export conversation to markdown format"""
        pass
```

### ModelResponse

```python
@dataclass
class ModelResponse:
    model: str
    success: bool
    response: str = ""
    duration: float = 0.0
    tokens: int = 0
    tokens_per_second: float = 0.0
    error: Optional[str] = None
    
    def is_loading(self) -> bool:
        return not self.success and self.error is None
    
    def has_error(self) -> bool:
        return not self.success and self.error is not None
```

## Error Handling

### Error Categories

1. **Model Query Errors**
   - Timeout: Display "Request timed out" in model column
   - Connection Error: Display "Connection failed" with retry option
   - API Error: Display error message from Ollama

2. **Validation Errors**
   - Empty query: Show warning dialog
   - No models selected: Show warning dialog
   - Invalid model selection: Prevent submission

3. **State Errors**
   - Failed to load models: Show error dialog with retry
   - Failed to save results: Show error dialog, keep results in memory

### Error Display Strategy

**In Model Columns**:
```
┌──────────────┐
│ Model Name   │
├──────────────┤
│              │
│   ⚠️ Error   │
│              │
│ [Error Msg]  │
│              │
│ [Retry]      │
│              │
└──────────────┘
```

**Loading State**:
```
┌──────────────┐
│ Model Name   │
├──────────────┤
│              │
│   ⏳         │
│ Processing...│
│              │
└──────────────┘
```

### Error Recovery

- **Retry Mechanism**: Allow individual model retry without re-querying all models
- **Graceful Degradation**: Show successful responses even if some models fail
- **State Preservation**: Maintain conversation history even after errors

## Testing Strategy

### Unit Testing Focus

Given the MVP scope and optional testing approach, focus on:

1. **State Management Logic**
   - Conversation history operations
   - State transitions
   - Model selection validation

2. **Data Transformation**
   - Response formatting
   - Markdown export
   - Metrics calculation

3. **Error Handling**
   - Timeout handling
   - Invalid input handling
   - State recovery

### Manual Testing Checklist

**Landing Page**:
- [ ] Landing page displays correctly on startup
- [ ] Model selection opens and closes properly
- [ ] First query transitions to chat interface
- [ ] Model selection validation works

**Chat Interface**:
- [ ] User queries display right-aligned
- [ ] Model responses display in horizontal layout
- [ ] Scrolling works (vertical and horizontal)
- [ ] Follow-up queries append correctly
- [ ] Bottom input bar remains fixed

**Model Responses**:
- [ ] Loading states display correctly
- [ ] Successful responses render properly
- [ ] Error states display with messages
- [ ] Metrics display accurately
- [ ] Individual column scrolling works

**State Management**:
- [ ] Conversation history persists across queries
- [ ] Clear conversation returns to landing page
- [ ] Model selection changes apply to new queries only
- [ ] Results can be saved to file

**Edge Cases**:
- [ ] >3 models trigger horizontal scrolling
- [ ] Long responses scroll within columns
- [ ] Window resize handles gracefully
- [ ] Empty query prevented
- [ ] No models selected prevented

### Integration Testing

**End-to-End Flows**:
1. Launch → Select models → Submit query → View responses → Follow-up → Save
2. Launch → Submit with default models → Change models → Submit again
3. Launch → Submit → Clear conversation → Return to landing
4. Launch → Model query timeout → View error → Retry

### Performance Testing

**Metrics to Monitor**:
- UI responsiveness during parallel queries
- Scroll performance with long conversations
- Memory usage with multiple queries
- Rendering time for large responses

**Acceptance Criteria**:
- UI remains responsive during queries (no freezing)
- Smooth scrolling with <100ms lag
- Memory growth <50MB per 10 queries
- Response rendering <500ms

## MVP Scope and Future Enhancements

### MVP Includes

✅ Landing page with model selection
✅ Bottom input bar (single-line)
✅ Chat interface with conversation history
✅ Side-by-side model response display
✅ Basic styling (functional, not polished)
✅ Loading and error states
✅ Result saving to markdown
✅ Conversation clearing

### MVP Excludes (Future Enhancements)

❌ Markdown rendering in responses (show plain text)
❌ Code syntax highlighting
❌ Model icons/avatars
❌ Dark mode
❌ Response copying/sharing
❌ Query editing/regeneration
❌ Response rating/feedback
❌ Advanced input features (file upload, voice input)
❌ Conversation search/filtering
❌ Export to multiple formats
❌ Keyboard shortcuts (beyond Enter to send)

### Future Enhancement Priorities

**Phase 2** (Post-MVP):
1. Markdown rendering with code blocks
2. Copy response button
3. Model icons
4. Keyboard shortcuts

**Phase 3** (Polish):
1. Dark mode
2. Response regeneration
3. Query editing
4. Advanced styling

**Phase 4** (Advanced Features):
1. File attachments
2. Conversation search
3. Response rating
4. Export formats

## Implementation Notes

### Tkinter Constraints

**Limitations**:
- Limited CSS-like styling
- No native markdown rendering
- Manual scroll management required
- Fixed layout calculations

**Workarounds**:
- Use Frame backgrounds for "cards"
- Plain text for MVP (defer markdown)
- Canvas + Scrollbar for custom scrolling
- Pack/Grid managers for responsive layout

### Windows Compatibility

**Considerations**:
- Test with Windows default fonts
- Ensure proper DPI scaling
- Handle Windows-specific path separators
- Test with Windows theme (light/dark)

### Performance Optimization

**Strategies**:
- Lazy rendering for long conversations
- Virtual scrolling for >20 queries (future)
- Debounce scroll events
- Reuse widget instances where possible

### Code Organization

**File Structure**:
```
interactive_test.py
├── # UI Components
├── class LandingPageView
├── class ChatInterfaceView
├── class UserQueryBubble
├── class ModelResponseRow
├── class ModelResponseColumn
├── class BottomInputBar
├── class ModelSelectionPanel
├── # State Management
├── class StateManager
├── class ConversationHistory
├── # Data Models
├── @dataclass ConversationEntry
├── @dataclass ModelResponse
├── # Utilities
├── def get_available_models()
├── def query_model()
├── def save_results_to_file()
└── # Main
    └── class ModelComparisonGUI
```

**Refactoring Strategy**:
- Extract UI components into separate classes
- Maintain existing query_model() and parallel execution logic
- Preserve result saving functionality
- Keep backward compatibility with existing result format

## Migration Path

### Phase 1: Component Extraction
1. Extract current UI code into legacy methods
2. Create new component classes (empty shells)
3. Implement StateManager
4. Add view switching logic

### Phase 2: Landing Page
1. Implement LandingPageView
2. Implement ModelSelectionPanel
3. Wire up first query submission
4. Test landing → chat transition

### Phase 3: Chat Interface
1. Implement ChatInterfaceView structure
2. Implement UserQueryBubble
3. Implement BottomInputBar
4. Test basic chat layout

### Phase 4: Response Display
1. Implement ModelResponseRow
2. Implement ModelResponseColumn
3. Add loading/error states
4. Test horizontal scrolling

### Phase 5: Integration
1. Connect all components
2. Implement conversation history
3. Add follow-up query support
4. Test complete flow

### Phase 6: Polish & Testing
1. Refine styling
2. Fix edge cases
3. Manual testing
4. Documentation

## Conclusion

This design provides a clear path to transform the interactive_test.py UI into a modern chat-based interface while maintaining the MVP focus on functionality over aesthetics. The modular component structure allows for incremental implementation and future enhancements without major refactoring.

Key success factors:
- Clear separation between landing and chat states
- Reusable component architecture
- Preservation of existing parallel query logic
- Focus on core functionality for MVP
- Extensible design for future features
