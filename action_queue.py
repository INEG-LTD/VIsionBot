"""
Action Queue System for BrowserVisionBot

Provides a thread-safe queue for deferring actions from callbacks.
Actions are automatically processed after each act() call.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from collections import deque


@dataclass
class QueuedAction:
    """Represents an action queued for later execution"""
    action: str
    action_id: Optional[str] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    queued_at: float = field(default_factory=time.time)


class ActionQueue:
    """Thread-safe queue for deferred actions with priority support"""
    
    def __init__(self, max_size: int = 100):
        self._queue = deque()
        self._lock = threading.Lock()
        self._max_size = max_size
        self._queued_ids = set()  # Track queued action IDs to prevent circular dependencies
    
    def enqueue(self, action: str, action_id: Optional[str] = None, 
                priority: int = 0, metadata: Dict[str, Any] = None) -> None:
        """
        Add action to queue with optional priority and metadata.
        
        Args:
            action: The action to execute (e.g., "click: button")
            action_id: Optional action ID for tracking
            priority: Priority level (higher = executed first)
            metadata: Optional metadata dict
        """
        with self._lock:
            if len(self._queue) >= self._max_size:
                raise RuntimeError(f"Action queue is full (max {self._max_size} actions)")
            
            # Check for circular dependencies (only for user-provided IDs)
            if action_id and action_id in self._queued_ids:
                raise ValueError(f"Circular dependency detected: {action_id}")
            
            # Generate action ID if not provided
            if action_id is None:
                action_id = f"queued_{int(time.time() * 1000)}"
            
            queued_action = QueuedAction(
                action=action,
                action_id=action_id,
                priority=priority,
                metadata=metadata or {}
            )
            
            # Insert based on priority (higher priority first)
            inserted = False
            for i, existing_action in enumerate(self._queue):
                if priority > existing_action.priority:
                    self._queue.insert(i, queued_action)
                    inserted = True
                    break
            
            if not inserted:
                self._queue.append(queued_action)
            
            # Only track user-provided IDs to prevent circular dependencies
            if action_id and not action_id.startswith("queued_"):
                self._queued_ids.add(action_id)
    
    def dequeue(self) -> Optional[QueuedAction]:
        """Get next action from queue (FIFO by default, priority-based if needed)"""
        with self._lock:
            if self._queue:
                action = self._queue.popleft()
                self._queued_ids.discard(action.action_id)
                return action
            return None
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        with self._lock:
            return len(self._queue) == 0
    
    def size(self) -> int:
        """Get current queue size"""
        with self._lock:
            return len(self._queue)
    
    def clear(self) -> None:
        """Clear all queued actions"""
        with self._lock:
            self._queue.clear()
            self._queued_ids.clear()
    
    def inspect(self) -> List[QueuedAction]:
        """Inspect queued actions without processing (returns copy)"""
        with self._lock:
            return list(self._queue)
    
    def peek(self) -> Optional[QueuedAction]:
        """Peek at next action without removing it"""
        with self._lock:
            return self._queue[0] if self._queue else None
