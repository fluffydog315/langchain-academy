import time
from typing import Dict, Any, Optional, Union, List, Type, Tuple
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('langgraph_research')

def record_error(
    state: Dict[str, Any], 
    error: Union[Exception, str],
    node_name: str,
    severity: str = "error",
    context: Optional[Dict[str, Any]] = None,
    add_to_history: bool = True
) -> Dict[str, Any]:
    """
    Create standardized error record and update state
    
    Args:
        state: Current state dictionary
        error: The exception that occurred or error message
        node_name: Name of the node where error occurred
        severity: Error severity ("warning", "error", "critical")
        context: Additional context information
        add_to_history: Whether to add error to error_history
        
    Returns:
        Updated state dictionary with error information
    """
    # Create error info
    if isinstance(error, Exception):
        error_type = type(error).__name__
        message = str(error)
        stack_trace = traceback.format_exc()
    else:
        error_type = "CustomError"
        message = str(error)
        stack_trace = None
    
    # Default context if none provided
    if context is None:
        context = {"state_keys": list(state.keys())}
    
    # Create error info dictionary
    error_info = {
        "error_type": error_type,
        "message": message,
        "node": node_name,
        "timestamp": time.time(),
        "severity": severity,
        "recovery_attempted": False,
        "context": context
    }
    
    # Add stack trace if available
    if stack_trace:
        error_info["stack_trace"] = stack_trace
    
    # Log the error
    logger.error(
        f"Error in node {node_name}: [{error_type}] {message} "
        f"(Severity: {severity})"
    )
    
    # Create state update
    result = {"error": error_info}
    
    # Add to error history if requested
    if add_to_history:
        error_history = state.get("error_history", [])
        result["error_history"] = error_history + [error_info]
    
    return result

def clear_error(state: Dict[str, Any], mark_recovered: bool = True) -> Dict[str, Any]:
    """
    Clear error from state, optionally marking it as recovered
    
    Args:
        state: Current state dictionary
        mark_recovered: Whether to mark the error as recovered in history
        
    Returns:
        Updated state with error cleared
    """
    # Create a new state without the error
    result = {}
    
    # Mark the last error as recovered if requested
    if mark_recovered and "error_history" in state and state["error_history"]:
        error_history = state["error_history"].copy()
        if error_history:
            error_history[-1]["recovery_attempted"] = True
            error_history[-1]["recovered"] = True
            error_history[-1]["recovery_timestamp"] = time.time()
        result["error_history"] = error_history
    
    return result

def is_error_state(state: Dict[str, Any]) -> bool:
    """
    Check if state contains an error
    
    Args:
        state: State dictionary to check
        
    Returns:
        True if state contains an error, False otherwise
    """
    return "error" in state and state["error"] is not None

def get_error_severity(state: Dict[str, Any]) -> Optional[str]:
    """
    Get the severity of the current error
    
    Args:
        state: State dictionary
        
    Returns:
        Severity string or None if no error
    """
    if not is_error_state(state):
        return None
    
    return state["error"].get("severity", "error")

def record_recovery(
    state: Dict[str, Any],
    recovery_method: str,
    success: bool = True,
    details: Optional[str] = None
) -> Dict[str, Any]:
    """
    Record recovery attempt information
    
    Args:
        state: Current state dictionary
        recovery_method: Method used for recovery
        success: Whether recovery was successful
        details: Additional details about recovery
        
    Returns:
        Updated state with recovery information
    """
    # Create recovery info
    recovery_info = {
        "timestamp": time.time(),
        "method": recovery_method,
        "success": success,
        "details": details
    }
    
    # If there's an error, reference it
    if "error" in state and state["error"]:
        recovery_info["error_type"] = state["error"].get("error_type")
        recovery_info["error_node"] = state["error"].get("node")
    
    # Update error history if it exists
    result = {"recovery_info": recovery_info}
    
    if "error_history" in state and state["error_history"]:
        error_history = state["error_history"].copy()
        if error_history:
            error_history[-1]["recovery_attempted"] = True
            error_history[-1]["recovered"] = success
            error_history[-1]["recovery_method"] = recovery_method
            error_history[-1]["recovery_timestamp"] = time.time()
            if details:
                error_history[-1]["recovery_details"] = details
        result["error_history"] = error_history
    
    # Clear error if recovery was successful
    if success:
        result["error"] = None
    
    return result
