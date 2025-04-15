# LangGraph Error Handling Examples

This document provides concrete code examples for implementing error handling in a LangGraph application.

## 1. State-Based Error Tracking

### Enhanced State Type Definition

```python
from typing import Optional, Dict, Any, List
from typing_extensions import TypedDict
from pydantic import BaseModel

class ResearchGraphState(TypedDict):
    topic: str  # Research topic
    max_analysts: int  # Number of analysts
    analysts: List[Any]  # Analyst asking questions
    sections: List[str]  # Report sections
    
    # Error tracking
    error: Optional[Dict[str, Any]]  # Error information if present
```

### Node Function with Error Detection

```python
def check_for_errors(state: Dict[str, Any]):
    """
    Conditional router that checks for errors and routes accordingly
    """
    if "error" in state and state["error"] is not None:
        error_type = state["error"].get("error_type", "unknown")
        
        # Different routing based on error type
        if error_type == "LLMError":
            return "handle_llm_error"
        elif error_type == "APIError":
            return "handle_api_error"
        else:
            return "handle_general_error"
            
    # No errors, continue normal flow
    return "continue_normal_flow"
```

### Error Recording in Node

```python
def create_analysts(state: Dict[str, Any]):
    """Node to create analysts with error handling"""
    try:
        # Normal function logic
        topic = state["topic"]
        max_analysts = state["max_analysts"]
        
        # Generate analysts with LLM
        analysts = generate_analysts(topic, max_analysts)
        
        return {"analysts": analysts}
        
    except Exception as e:
        # Record error in state
        return {
            "error": {
                "error_type": type(e).__name__,
                "message": str(e),
                "node": "create_analysts",
                "timestamp": time.time()
            }
        }
```

## 2. Try-Except Blocks in Node Functions

### Simple Error Handling in Node

```python
def search_web(state: Dict[str, Any]):
    """Search web with graceful error handling"""
    try:
        # Extract query
        query = state.get("search_query", "")
        
        # Perform search
        results = search_service.search(query)
        
        return {"search_results": results}
        
    except ConnectionError:
        # Handle connection issues
        logger.warning("Search service unavailable, using cached results")
        return {"search_results": get_cached_results(query)}
        
    except Exception as e:
        # Generic error handling
        logger.error(f"Search failed: {str(e)}")
        return {"search_results": [], "search_error": str(e)}
```

### Handling LLM Errors

```python
def generate_report(state: Dict[str, Any]):
    """Generate report with LLM error handling"""
    try:
        # Extract content
        sections = state.get("sections", [])
        content = "\n\n".join(sections)
        
        # Generate report with LLM
        system_message = "Summarize these sections into a cohesive report"
        report = llm.invoke([system_message, content])
        
        return {"final_report": report.content}
        
    except TokenLimitError:
        # Handle token limit exceeded
        logger.warning("Token limit exceeded, chunking content")
        chunks = chunk_content(content)
        summaries = []
        
        for chunk in chunks:
            summary = llm.invoke([system_message, chunk])
            summaries.append(summary.content)
            
        combined_summary = "\n\n".join(summaries)
        final_report = llm.invoke([
            "Combine these summaries into a single cohesive report", 
            combined_summary
        ])
        
        return {"final_report": final_report.content}
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return {"final_report": "Error generating report"}
```

## 3. Decorator-Based Error Handling

### Safe Node Decorator

```python
import functools
import logging
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("langgraph")

def safe_node(
    max_retries: int = 0,
    retry_delay: float = 1.0,
    error_key: str = "error"
):
    """
    Decorator that makes a node function safe by catching exceptions
    
    Args:
        max_retries: Number of times to retry on failure
        retry_delay: Delay between retries in seconds
        error_key: Key to use in state for error information
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(state: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
            # Get node name from function name
            node_name = func.__name__
            
            # Try with retries
            for attempt in range(max_retries + 1):
                try:
                    # Call the original function
                    result = func(state, *args, **kwargs)
                    return result
                    
                except Exception as e:
                    # Log the error
                    logger.error(f"Error in {node_name} (attempt {attempt+1}/{max_retries+1}): {str(e)}")
                    
                    # Check if we should retry
                    if attempt < max_retries:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                        
                    # No more retries, record the error
                    error_info = {
                        "error_type": type(e).__name__,
                        "message": str(e),
                        "node": node_name,
                        "timestamp": time.time()
                    }
                    
                    return {error_key: error_info}
                    
        return wrapper
    return decorator
```

### Using the Decorator

```python
@safe_node(max_retries=3, retry_delay=2.0)
def call_llm(state: Dict[str, Any]):
    """Call LLM with retry logic"""
    # Function implementation
    query = state.get("query", "")
    response = llm.invoke(query)
    return {"response": response.content}
```

## 4. Checkpoint and Recovery System

### Checkpoint Manager

```python
import json
import os
import time
from typing import Any, Dict, Optional

class CheckpointManager:
    """Manages checkpoints for LangGraph state recovery"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, state: Dict[str, Any], checkpoint_id: Optional[str] = None) -> str:
        """
        Save state to a checkpoint file
        
        Args:
            state: State dictionary to save
            checkpoint_id: Optional ID for the checkpoint
            
        Returns:
            Path to the checkpoint file
        """
        # Generate checkpoint ID if not provided
        if checkpoint_id is None:
            checkpoint_id = f"checkpoint_{int(time.time())}"
            
        # Ensure .json extension
        if not checkpoint_id.endswith(".json"):
            checkpoint_id = f"{checkpoint_id}.json"
            
        # Create the full path
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_id)
        
        # Prepare serializable state
        serializable_state = self._prepare_for_serialization(state)
        
        # Write to file
        with open(checkpoint_path, 'w') as f:
            json.dump(serializable_state, f, indent=2)
            
        return checkpoint_path
        
    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load state from a checkpoint file
        
        Args:
            checkpoint_id: ID of the checkpoint to load
            
        Returns:
            The loaded state dictionary
        """
        # Ensure .json extension
        if not checkpoint_id.endswith(".json"):
            checkpoint_id = f"{checkpoint_id}.json"
            
        # Create the full path
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_id)
        
        # Check if file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
            
        # Load from file
        with open(checkpoint_path, 'r') as f:
            state = json.load(f)
            
        return state
        
    def list_checkpoints(self):
        """List all available checkpoints"""
        return [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".json")]
        
    def _prepare_for_serialization(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable form"""
        if isinstance(obj, dict):
            return {k: self._prepare_for_serialization(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_serialization(item) for item in obj]
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        else:
            # Try to convert to primitive type
            try:
                json.dumps(obj)
                return obj
            except TypeError:
                return str(obj)
```

### Checkpoint Node

```python
# Create checkpoint manager
checkpoint_manager = CheckpointManager()

def save_checkpoint_node(state: Dict[str, Any]):
    """Node that saves a checkpoint of the current state"""
    # Save checkpoint
    checkpoint_path = checkpoint_manager.save_checkpoint(state)
    
    # Return state with checkpoint information
    return {"last_checkpoint": os.path.basename(checkpoint_path)}

def restore_from_checkpoint(state: Dict[str, Any]):
    """Node that restores state from the latest checkpoint"""
    try:
        # Get list of checkpoints
        checkpoints = checkpoint_manager.list_checkpoints()
        
        if not checkpoints:
            # No checkpoints available
            return {"error": "No checkpoints available for recovery"}
            
        # Sort by timestamp (assuming checkpoint_{timestamp}.json format)
        latest_checkpoint = sorted(checkpoints)[-1]
        
        # Load checkpoint
        recovered_state = checkpoint_manager.load_checkpoint(latest_checkpoint)
        
        # Add recovery information
        recovered_state["_recovered"] = True
        recovered_state["_recovery_source"] = latest_checkpoint
        
        return recovered_state
        
    except Exception as e:
        return {"error": f"Recovery failed: {str(e)}"}
```

## 5. Graph-Level Error Routing

### Research Graph with Error Handling

```python
from langgraph.graph import StateGraph
from langgraph.constants import START, END

def create_research_graph_with_error_handling():
    """Create research graph with error handling paths"""
    # Create state graph
    builder = StateGraph(ResearchGraphState)
    
    # Add normal nodes
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("conduct_interview", create_interview_graph())
    builder.add_node("write_report", write_report)
    builder.add_node("finalize_report", finalize_report)
    
    # Add error handling nodes
    builder.add_node("handle_error", handle_error)
    builder.add_node("retry_analysis", retry_analysis)
    builder.add_node("fallback_report", fallback_report)
    
    # Start with analyst creation
    builder.add_edge(START, "create_analysts")
    
    # Check for errors after creating analysts
    builder.add_conditional_edges(
        "create_analysts",
        check_for_errors,
        {
            "handle_llm_error": "retry_analysis",
            "handle_api_error": "handle_error",
            "handle_general_error": "handle_error",
            "continue_normal_flow": "conduct_interview"
        }
    )
    
    # After handling errors, retry or use fallback
    builder.add_edge("retry_analysis", "create_analysts")
    builder.add_edge("handle_error", "fallback_report")
    
    # Continue with normal flow
    builder.add_edge("conduct_interview", "write_report")
    
    # Check for errors after writing report
    builder.add_conditional_edges(
        "write_report",
        check_for_errors,
        {
            "handle_error": "fallback_report",
            "continue_normal_flow": "finalize_report"
        }
    )
    
    # Finalize and end
    builder.add_edge("finalize_report", END)
    builder.add_edge("fallback_report", END)
    
    return builder.compile()
```

## 6. LLM Retry with Exponential Backoff

```python
import time
import random
from typing import Any, Callable, TypeVar, cast

T = TypeVar('T')

def retry_with_exponential_backoff(
    func: Callable[..., T],
    max_retries: int = 5,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> Callable[..., T]:
    """
    Retry a function with exponential backoff
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter to delay
        
    Returns:
        Wrapped function with retry logic
    """
    def wrapper(*args: Any, **kwargs: Any) -> T:
        # Initialize variables
        num_retries = 0
        delay = initial_delay
        
        # Loop until max retries
        while True:
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                # Check if we've hit max retries
                num_retries += 1
                if num_retries > max_retries:
                    raise
                    
                # Calculate delay with jitter if enabled
                if jitter:
                    delay = delay * exponential_base * (1 + random.random())
                else:
                    delay = delay * exponential_base
                    
                # Log retry
                print(f"Retry {num_retries}/{max_retries} after {delay:.2f}s due to {e}")
                
                # Sleep before retry
                time.sleep(delay)
                
    return cast(Callable[..., T], wrapper)
```

## 7. Structured Logging

```python
import logging
import json
import time
from typing import Any, Dict, Optional

class StructuredLogger:
    """Logger that produces structured JSON logs"""
    
    def __init__(self, name: str, log_level: int = logging.INFO):
        """
        Initialize structured logger
        
        Args:
            name: Logger name
            log_level: Logging level
        """
        # Set up logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Add handler if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            self.logger.addHandler(handler)
            
    def log(
        self,
        level: int,
        message: str,
        node: Optional[str] = None,
        state_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Log a structured message
        
        Args:
            level: Log level
            message: Log message
            node: Node name
            state_id: State ID
            **kwargs: Additional fields to include
        """
        # Create log entry
        log_entry = {
            "timestamp": time.time(),
            "message": message,
            "level": logging.getLevelName(level),
        }
        
        # Add optional fields
        if node:
            log_entry["node"] = node
            
        if state_id:
            log_entry["state_id"] = state_id
            
        # Add additional fields
        log_entry.update(kwargs)
        
        # Log as JSON
        self.logger.log(level, json.dumps(log_entry))
        
    def info(self, message: str, **kwargs: Any) -> None:
        """Log at INFO level"""
        self.log(logging.INFO, message, **kwargs)
        
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log at WARNING level"""
        self.log(logging.WARNING, message, **kwargs)
        
    def error(self, message: str, **kwargs: Any) -> None:
        """Log at ERROR level"""
        self.log(logging.ERROR, message, **kwargs)
        
    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback"""
        kwargs["traceback"] = logging.traceback.format_exc()
        self.log(logging.ERROR, message, **kwargs)
```

## 8. Graceful Degradation with Fallbacks

```python
def generate_content_with_fallbacks(state: Dict[str, Any]):
    """Generate content with multiple fallback options"""
    try:
        # Attempt to use primary LLM with full context
        topic = state["topic"]
        context = state.get("context", [])
        
        response = primary_llm.invoke(topic, context)
        return {"content": response.content, "source": "primary_llm"}
        
    except Exception as primary_error:
        try:
            # Fallback 1: Try secondary LLM
            response = secondary_llm.invoke(topic)
            return {"content": response.content, "source": "secondary_llm"}
            
        except Exception as secondary_error:
            try:
                # Fallback 2: Try cached response
                cached_content = get_cached_response(topic)
                if cached_content:
                    return {"content": cached_content, "source": "cache"}
                    
                # Fallback 3: Use template response
                template_content = get_template_response(topic)
                return {"content": template_content, "source": "template"}
                
            except Exception as final_error:
                # Final fallback: Return error message
                return {
                    "content": f"Unable to generate content for {topic}",
                    "source": "error_message",
                    "error": str(final_error)
                }
```
