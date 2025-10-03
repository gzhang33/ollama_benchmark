# Implementation Plan

## Overview

This implementation plan transforms the existing left-panel input interface into a modern chat-based interface with landing page and conversation flow. The current implementation has a working left-panel UI with model selection and vertical/horizontal result display. We need to refactor this into a chat-first experience with bottom input and conversational history.

---

- [x] 1. Create state management foundation

  - Extract conversation history tracking into StateManager class
  - Implement view state management (landing vs chat)
  - Add data models for ConversationEntry and ModelResponse
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 1.1 Implement StateManager class

  - Create StateManager with current_view, conversation_history, and selected_models properties
  - Add methods: add_conversation_entry(), clear_conversation(), get_conversation_history(), set_selected_models()
  - Add view transition methods: transition_to_chat(), transition_to_landing()
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 1.2 Create data model classes

  - Implement ConversationEntry dataclass with query, timestamp, and responses
  - Implement ModelResponse dataclass with model, success, response, duration, tokens, tokens_per_second, error
  - Add helper methods: is_loading(), has_error(), to_markdown()
  - _Requirements: 6.1, 6.2, 8.1, 8.2, 8.3_

- [ ]\* 1.3 Write unit tests for state management

  - Test conversation history operations (add, clear, retrieve)
  - Test state transitions (landing to chat, chat to landing)
  - Test model selection validation
  - _Requirements: 6.1, 6.2, 6.5, 6.6_

- [x] 2. Build LandingPageView component

  - Create LandingPageView class with centered layout
  - Implement welcome content with title and subtitle
  - Add center input box with placeholder "Ask anything..."
  - Wire up model selection button
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8_

- [x] 2.1 Implement landing page layout

  - Create centered container with max-width constraint
  - Add model icons placeholder at top (simple text for MVP)
  - Display title "Find the best AI for you" and subtitle
  - Position single-line input box in center with attachment and send icons
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.8_

- [x] 2.2 Connect model selection to landing page

  - Add "Select Models" button below input
  - Wire button to open ModelSelectionPanel
  - Display selected model count on landing page
  - _Requirements: 1.6, 1.7, 5.1, 5.2_

- [x] 2.3 Handle first query submission

  - Validate input is not empty
  - Validate at least one model is selected
  - Transition from landing to chat interface on submit
  - _Requirements: 1.5, 3.1, 5.7_

- [x] 3. Create BottomInputBar component

  - Build fixed bottom input bar with single-line entry
  - Add attachment button (left) and send button (right)
  - Implement Enter key submission
  - Add placeholder text "Ask followup..."
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6, 2.8_

- [x] 3.1 Implement input bar layout and styling

  - Create fixed-height frame (60px) at bottom
  - Add attachment button with icon on left
  - Add Entry widget with proper styling in center
  - Add send button with icon on right
  - Apply MVP styling (simple colors, flat design)
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 3.2 Wire up input submission logic

  - Bind Enter key to submit callback
  - Disable input during query processing
  - Re-enable and auto-focus after completion
  - Clear input after successful submission
  - _Requirements: 2.6, 8.1, 8.4_

- [ ]\* 3.3 Add multi-line input support (optional enhancement)

  - Implement Shift+Enter for multi-line input
  - Auto-expand input height for multi-line content
  - _Requirements: 2.7_

- [x] 4. Build ChatInterfaceView component

  - Create scrollable conversation area above bottom input
  - Implement conversation rendering with queries and responses
  - Add auto-scroll to latest content
  - Maintain fixed bottom input bar
  - _Requirements: 3.1, 3.2, 3.4, 3.5, 3.6, 3.7, 7.2, 7.4, 7.5_

- [x] 4.1 Implement chat layout structure

  - Create main container with scrollable content area
  - Position BottomInputBar at bottom (always visible)
  - Set up Canvas + Scrollbar for conversation scrolling
  - Configure proper padding and spacing
  - _Requirements: 3.4, 7.2, 7.4_

- [x] 4.2 Implement conversation rendering

  - Create render_conversation() method to display all entries
  - Add add_user_query() to append new query bubbles
  - Add add_model_responses() to append response rows
  - Implement auto_scroll_to_bottom() for new content
  - _Requirements: 3.5, 3.6, 3.7, 7.5_

- [x] 5. Create UserQueryBubble component

  - Build right-aligned query display
  - Apply chat bubble styling (background, rounded corners)
  - Add proper padding and margins
  - _Requirements: 3.2, 3.3_

- [x] 5.1 Implement query bubble layout and styling

  - Create Frame with right-aligned packing
  - Apply light blue/gray background (#e3f2fd)
  - Set text color and font (Arial 11)
  - Add padding (15px) and word wrapping (400px max width)
  - _Requirements: 3.2, 3.3_

- [x] 6. Build ModelResponseRow component

  - Create horizontal layout container for model responses
  - Implement equal-width columns for each model
  - Add horizontal scrollbar for >3 models
  - Support independent vertical scrolling per column
  - _Requirements: 4.1, 4.2, 4.6, 4.9_

- [x] 6.1 Implement horizontal response layout

  - Create Canvas with horizontal scrolling capability
  - Add horizontal scrollbar
  - Create columns_frame for model columns
  - Configure equal-width column distribution
  - _Requirements: 4.1, 4.2, 4.9_

- [x] 6.2 Add ModelResponseColumn instances

  - Iterate through responses and create column for each
  - Pack columns side-by-side with proper spacing (5px padding)
  - Set fixed column width (350px for MVP)
  - _Requirements: 4.2, 4.6_

- [x] 7. Create ModelResponseColumn component

  - Build column structure with header, content, and footer
  - Implement scrollable content area
  - Add loading, success, and error states
  - Display performance metrics
  - _Requirements: 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_

- [x] 7.1 Implement column layout structure

  - Create fixed-width Frame (350px) with border
  - Add model name header with bold font
  - Add ScrolledText widget for response content
  - Add metrics footer with performance data
  - _Requirements: 4.3, 4.4, 4.5_

- [x] 7.2 Implement state-based rendering

  - Add loading state with spinner/progress indicator
  - Add success state with response text and metrics
  - Add error state with error message
  - Format metrics: "2.5s | 45 tokens | 45 tok/s"
  - _Requirements: 4.7, 4.8, 8.2, 8.3, 8.5, 8.6_

- [ ]\* 7.3 Add response content formatting

  - Display plain text for MVP (defer markdown rendering)
  - Ensure proper text wrapping and scrolling
  - _Requirements: 4.4, 9.3_

- [x] 8. Refactor ModelSelectionPanel

  - Update to work as modal dialog
  - Add Select All / Deselect All buttons
  - Implement validation (prevent empty selection)
  - Show model count display
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_

- [x] 8.1 Implement modal dialog layout

  - Create Toplevel window (400x600)
  - Add header with "Select models to compare" title
  - Add Select All / Deselect All button row
  - Create scrollable checkbox list
  - Add Confirm button at bottom
  - _Requirements: 5.1, 5.2, 5.3_

- [x] 8.2 Add selection validation

  - Prevent confirmation with no models selected
  - Show warning dialog if validation fails
  - Update selected_models in StateManager on confirm
  - _Requirements: 5.7, 5.8_

- [x] 9. Integrate components into main GUI


  - Refactor ModelComparisonGUI to use new components
  - Implement view switching (landing ↔ chat)
  - Wire up all event handlers and callbacks
  - Preserve existing parallel query logic
  - _Requirements: 3.1, 6.4, 6.6_

- [x] 9.1 Refactor main GUI initialization


  - Initialize StateManager in **init**
  - Create both LandingPageView and ChatInterfaceView
  - Set initial view to landing
  - Preserve existing base_url, output_dir, and models properties

  - _Requirements: 6.4_


- [ ] 9.2 Implement view switching logic

  - Add switch_to_landing() method to show landing page
  - Add switch_to_chat() method to show chat interface
  - Hide/show appropriate views based on state
  - Trigger view switch on first query and conversation clear
  - _Requirements: 3.1, 6.4, 6.6_


- [ ] 9.3 Wire up query submission flow

  - Connect landing page input to query execution
  - Connect bottom input bar to follow-up queries
  - Append queries and responses to conversation history
  - Update chat interface with new content

  - _Requirements: 3.6, 3.7, 6.1, 6.2_

- [ ] 9.4 Preserve existing functionality

  - Keep parallel_query_all_models() logic unchanged
  - Maintain result saving to markdown files


  - Preserve model loading and error handling
  - Keep existing query_model() and get_available_models() functions
  - _Requirements: 9.4_

- [ ] 10. Implement conversation management


  - Add clear conversation functionality
  - Implement conversation history persistence
  - Add result export to markdown
  - _Requirements: 6.5, 6.6, 9.4_

- [ ] 10.1 Add clear conversation feature


  - Add "New Conversation" or "Clear" button to chat interface
  - Clear conversation history in StateManager
  - Transition back to landing page
  - Reset UI state
  - _Requirements: 6.5, 6.6_


- [ ] 10.2 Update result saving for conversation format

  - Modify save_results_to_file() to handle conversation history
  - Include all queries and responses in chronological order
  - Maintain existing markdown format compatibility
  - Add conversation metadata (timestamp, model selections)
  - _Requirements: 9.4_


- [ ] 11. Add responsive layout and scrolling

  - Implement proper window resize handling
  - Configure scrolling for all scrollable areas
  - Test with different window sizes
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_


- [ ] 11.1 Configure scrolling behavior

  - Set up vertical scrolling for conversation area
  - Set up horizontal scrolling for response rows (>3 models)
  - Set up independent vertical scrolling for each response column

  - Ensure bottom input bar remains fixed during scrolling
  - _Requirements: 7.2, 7.3, 7.4, 7.6_

- [ ] 11.2 Implement auto-scroll functionality




  - Auto-scroll to bottom when new content is added
  - Preserve scroll position when user manually scrolls up
  - Test scroll performance with long conversations
  - _Requirements: 7.5_

- [ ] 11.3 Handle window resize


  - Configure frames to resize proportionally
  - Test layout with minimum window size (800x600)
  - Ensure all content remains accessible after resize
  - _Requirements: 7.1_

- [ ] 12. Polish UI and fix edge cases


  - Apply MVP styling consistently
  - Test and fix edge cases
  - Add loading indicators
  - Improve error messages
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 9.1, 9.2, 9.5_

- [ ] 12.1 Apply consistent MVP styling

  - Use simple color palette (blues, grays, whites)

  - Apply flat design (minimal borders, shadows)
  - Use consistent fonts (Arial for text, Consolas for code)
  - Add proper spacing and padding throughout
  - _Requirements: 9.1, 9.2_

- [ ] 12.2 Test and fix edge cases

  - Test with empty query submission
  - Test with no models selected
  - Test with >3 models (horizontal scrolling)
  - Test with very long responses (column scrolling)
  - Test with all models failing
  - Test with mixed success/failure responses
  - _Requirements: 8.5, 8.6_

- [ ] 12.3 Improve loading and error states

  - Add clear loading indicators during query processing
  - Show progress for parallel queries
  - Display user-friendly error messages
  - Add retry capability for failed models (optional)
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [ ]\* 12.4 Manual testing checklist
  - Test complete flow: launch → select models → query → follow-up → save
  - Test model selection changes mid-session
  - Test conversation clear and return to landing
  - Test window resize and scrolling
  - Test with different numbers of models (1, 3, 5+)
  - Test error scenarios (timeout, connection failure)
  - _Requirements: All_

---

## Notes

- **MVP Focus**: This task list prioritizes functional correctness over visual polish
- **Plain Text**: Markdown rendering is deferred to future iterations (show plain text for MVP)
- **Existing Code**: Preserve all existing query logic, parallel execution, and result saving
- **Testing**: Unit tests are marked as optional (\*) to focus on core implementation
- **Incremental**: Each task builds on previous tasks to ensure working state at each step

## Implementation Strategy

1. **Phase 1 (Tasks 1-2)**: Build foundation with state management and landing page
2. **Phase 2 (Tasks 3-5)**: Create input and query display components
3. **Phase 3 (Tasks 6-8)**: Build response display and model selection
4. **Phase 4 (Tasks 9-10)**: Integrate everything and add conversation management
5. **Phase 5 (Tasks 11-12)**: Polish, test, and fix edge cases
