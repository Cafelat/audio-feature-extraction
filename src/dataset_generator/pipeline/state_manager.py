"""State management for pipeline execution"""

import time
from typing import Optional, Dict, Any


class StateManager:
    """Pipeline state management
    
    Tracks state transitions and processing time in the pipeline.
    """
    
    def __init__(self):
        """Initialize StateManager"""
        self.states: Dict[str, Any] = {}
        self.start_time = time.time()
    
    def update(self, state_name: str, data: Any) -> None:
        """Update pipeline state
        
        Args:
            state_name: Name of the state
            data: Data associated with state
        """
        self.states[state_name] = data
    
    def get_state(self, state_name: str) -> Optional[Any]:
        """Get pipeline state
        
        Args:
            state_name: Name of the state
            
        Returns:
            Data associated with state, or None if not found
        """
        return self.states.get(state_name)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since initialization
        
        Returns:
            Elapsed time in seconds
        """
        return time.time() - self.start_time
    
    def reset(self) -> None:
        """Reset state manager"""
        self.states.clear()
        self.start_time = time.time()
