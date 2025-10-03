# Requirements Document

## Introduction

This document outlines the requirements for optimizing the interactive_test.py UI interface. The goal is to create an MVP version that prioritizes functional correctness and proper layout over visual polish. The redesign transforms the current left-panel input interface into a modern chat-based interface with a landing page and conversation view, similar to popular AI chat interfaces.

Key changes include:
- Moving the input prompt to the bottom as a single-line input
- Creating a landing page with model selection
- Implementing a chat interface that displays user queries and model responses side-by-side
- Supporting follow-up questions in a conversational flow

## Requirements

### Requirement 1: Landing Page Interface

**User Story:** As a user, I want to see a clean landing page when I first open the application, so that I can easily understand the purpose and select models before starting.

#### Acceptance Criteria

1. WHEN the application starts THEN the system SHALL display a landing page with centered content
2. WHEN the landing page is displayed THEN the system SHALL show AI model icons at the top
3. WHEN the landing page is displayed THEN the system SHALL show a title "Find the best AI for you" or similar
4. WHEN the landing page is displayed THEN the system SHALL show a subtitle describing the comparison functionality
5. WHEN the landing page is displayed THEN the system SHALL display a single-line input box in the center with placeholder text "Ask anything..."
6. WHEN the landing page is displayed THEN the system SHALL show a model selection button that allows users to choose which models to compare
7. WHEN the user clicks the model selection button THEN the system SHALL display a modal or panel with checkboxes for available models
8. WHEN the landing page is displayed THEN the system SHALL show attachment and send icons near the input box

### Requirement 2: Bottom Input Bar

**User Story:** As a user, I want the input prompt to be at the bottom of the screen as a single line, so that I can have a familiar chat-like experience and see more response content.

#### Acceptance Criteria

1. WHEN the chat interface is active THEN the system SHALL display a input box at the bottom of the window
2. WHEN the input box is displayed THEN the system SHALL position it with appropriate padding from the window edges
3. WHEN the input box is displayed THEN the system SHALL show a send button on the right side
4. WHEN the input box is displayed THEN the system SHALL show attachment/options icons on the left side
6. WHEN the user presses Enter THEN the system SHALL submit the query
7. WHEN the user presses Shift+Enter THEN the system SHALL allow multi-line input (optional enhancement)
8. WHEN the input box is empty THEN the system SHALL display placeholder text like "Ask followup..."

### Requirement 3: Chat Interface Layout

**User Story:** As a user, I want to see my questions and model responses in a chat-like interface, so that I can easily follow the conversation flow and compare model outputs.

#### Acceptance Criteria

1. WHEN the user submits a query THEN the system SHALL transition from the landing page to the chat interface
2. WHEN the chat interface is displayed THEN the system SHALL show the user's query in the top-right area
3. WHEN the user's query is displayed THEN the system SHALL use right-aligned styling to distinguish it from model responses
4. WHEN the chat interface is displayed THEN the system SHALL show a scrollable content area above the bottom input bar
5. WHEN multiple queries are submitted THEN the system SHALL display them in chronological order from top to bottom
6. WHEN the chat interface is active THEN the system SHALL maintain the bottom input bar for follow-up questions
7. WHEN the user submits a follow-up question THEN the system SHALL append it to the conversation history

### Requirement 4: Side-by-Side Model Response Display

**User Story:** As a user, I want to see responses from multiple models displayed side-by-side in the same row, so that I can easily compare their outputs for the same query.

#### Acceptance Criteria

1. WHEN models respond to a query THEN the system SHALL display all responses in a horizontal row layout below the user's query
2. WHEN displaying model responses THEN the system SHALL create equal-width columns for each selected model
3. WHEN displaying a model response THEN the system SHALL show the model name as a header
4. WHEN displaying a model response THEN the system SHALL show the response text content
5. WHEN displaying a model response THEN the system SHALL show performance metrics (duration, tokens, tokens/s)
6. WHEN displaying model responses THEN the system SHALL make each column independently scrollable if content is long
7. WHEN a model is still processing THEN the system SHALL show a loading indicator in that model's column
8. WHEN a model fails THEN the system SHALL show an error message in that model's column
9. WHEN more than 3 models are selected THEN the system SHALL allow horizontal scrolling to view all model columns

### Requirement 5: Model Selection Management

**User Story:** As a user, I want to select which models to compare before or during my session, so that I can focus on the models that are most relevant to my needs.

#### Acceptance Criteria

1. WHEN the user clicks the model selection button THEN the system SHALL display a list of available models
2. WHEN the model selection interface is displayed THEN the system SHALL show checkboxes for each available model
3. WHEN the model selection interface is displayed THEN the system SHALL show "Select All" and "Deselect All" buttons
4. WHEN the user checks/unchecks a model THEN the system SHALL update the selection state
5. WHEN the user confirms model selection THEN the system SHALL close the selection interface
6. WHEN the user confirms model selection THEN the system SHALL use the selected models for subsequent queries
7. WHEN no models are selected THEN the system SHALL prevent query submission and show a warning
8. WHEN the user changes model selection mid-session THEN the system SHALL apply the new selection to future queries only

### Requirement 6: Conversation History and State Management

**User Story:** As a user, I want the application to maintain my conversation history, so that I can review previous queries and responses.

#### Acceptance Criteria

1. WHEN the user submits a query THEN the system SHALL store the query in conversation history
2. WHEN models respond THEN the system SHALL store all responses with their associated query
3. WHEN the user scrolls up THEN the system SHALL display previous queries and responses
4. WHEN the application has conversation history THEN the system SHALL maintain the chat interface (not return to landing page)
5. WHEN the user wants to start fresh THEN the system SHALL provide a way to clear conversation history
6. WHEN conversation history is cleared THEN the system SHALL return to the landing page

### Requirement 7: Responsive Layout and Scrolling

**User Story:** As a user, I want the interface to handle different window sizes and content lengths gracefully, so that I can use the application comfortably.

#### Acceptance Criteria

1. WHEN the window is resized THEN the system SHALL adjust the layout proportionally
2. WHEN content exceeds the visible area THEN the system SHALL provide vertical scrolling for the main content area
3. WHEN model responses are wide THEN the system SHALL provide horizontal scrolling for the response row
4. WHEN the user scrolls THEN the system SHALL keep the bottom input bar fixed at the bottom
5. WHEN new content is added THEN the system SHALL auto-scroll to show the latest content
6. WHEN individual model responses are long THEN the system SHALL make each column independently scrollable

### Requirement 8: Visual Feedback and Loading States

**User Story:** As a user, I want clear visual feedback during model processing, so that I know the system is working and can see progress.

#### Acceptance Criteria

1. WHEN a query is submitted THEN the system SHALL disable the input box until processing begins
2. WHEN models are processing THEN the system SHALL show loading indicators in each model's column
3. WHEN a model completes THEN the system SHALL remove the loading indicator and show the response
4. WHEN all models complete THEN the system SHALL re-enable the input box for follow-up questions
5. WHEN a model times out THEN the system SHALL show a timeout message in that model's column
6. WHEN a model errors THEN the system SHALL show an error message with details in that model's column

### Requirement 9: MVP Scope and Future Enhancements

**User Story:** As a developer, I want to focus on core functionality for the MVP, so that we can deliver a working product quickly and iterate based on feedback.

#### Acceptance Criteria

1. WHEN implementing the MVP THEN the system SHALL prioritize functional correctness over visual polish
2. WHEN implementing the MVP THEN the system SHALL use simple styling (basic colors, fonts, spacing)
3. WHEN implementing the MVP THEN the system SHALL defer advanced features like markdown rendering to future iterations
4. WHEN implementing the MVP THEN the system SHALL maintain existing functionality (model querying, parallel execution, result saving)
5. WHEN implementing the MVP THEN the system SHALL ensure the new UI works on Windows with Tkinter
6. IF time permits THEN the system MAY add basic markdown rendering for code blocks
7. IF time permits THEN the system MAY add model icons or avatars
8. IF time permits THEN the system MAY add dark mode support
