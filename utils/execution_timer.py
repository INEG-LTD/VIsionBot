"""
Execution timer utility for tracking performance metrics.
"""
import time
from typing import Optional, Dict, List, Any
from utils.bot_logger import get_logger, LogLevel, LogCategory

logger = get_logger()

class ExecutionTimer:
    """Tracks execution timings for tasks, iterations, and commands"""
    
    def __init__(self):
        self.task_start_time: Optional[float] = None
        self.task_end_time: Optional[float] = None
        self.iteration_start_times: Dict[int, float] = {}
        self.iteration_end_times: Dict[int, float] = {}
        self.command_start_times: Dict[str, float] = {}
        self.command_end_times: Dict[str, float] = {}
        self.command_texts: Dict[str, str] = {}
        self.current_iteration: int = 0
        self.current_command_start: Optional[float] = None
        
    def start_task(self):
        """Start tracking task execution"""
        self.task_start_time = time.time()
        self.task_end_time = None
        self.iteration_start_times = {}
        self.iteration_end_times = {}
        self.command_start_times = {}
        self.command_end_times = {}
        self.current_iteration = 0
        self.current_command_start = None
        
    def end_task(self):
        """End task tracking"""
        self.task_end_time = time.time()
        
    def start_iteration(self):
        """Start tracking an iteration"""
        self.current_iteration += 1
        self.iteration_start_times[self.current_iteration] = time.time()
        
    def end_iteration(self):
        """End current iteration tracking"""
        if self.current_iteration in self.iteration_start_times:
            self.iteration_end_times[self.current_iteration] = time.time()
            
    def start_command(self, command_id: str, command: str):
        """Start tracking a command"""
        now = time.time()
        self.command_start_times[command_id] = now
        self.command_texts[command_id] = command
        self.current_command_start = now
        
    def end_command(self, command_id: Optional[str] = None):
        """End current command tracking"""
        now = time.time()
        if command_id and command_id in self.command_start_times:
            self.command_end_times[command_id] = now
        elif self.current_command_start:
            # If no ID provided, try to find the most recent started command that hasn't ended
            # This is a bit ambiguous without ID, but we can clear the current start flag
            pass
        
        self.current_command_start = None
            
    def set_command_text(self, command_id: str, text: str):
        """Set the command text for a command"""
        self.command_texts[command_id] = text
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all timings"""
        summary = {
            "total_duration": 0.0,
            "iterations": [],
            "commands": []
        }
        
        if self.task_start_time:
            end = self.task_end_time or time.time()
            summary["total_duration"] = end - self.task_start_time
            
        for i, start in self.iteration_start_times.items():
            end = self.iteration_end_times.get(i)
            duration = (end - start) if end else 0.0
            summary["iterations"].append({
                "iteration": i,
                "duration": duration
            })
            
        for cmd_id, start in self.command_start_times.items():
            end = self.command_end_times.get(cmd_id)
            duration = (end - start) if end else 0.0
            summary["commands"].append({
                "id": cmd_id,
                "text": self.command_texts.get(cmd_id, ""),
                "duration": duration
            })
            
        return summary
        
    def _format_duration(self, seconds: float) -> str:
        """Format duration in a human-readable way"""
        if seconds < 1.0:
            return f"{seconds*1000:.0f}ms"
        return f"{seconds:.2f}s"
        
    def log_summary(self):
        """Log timing summary to console"""
        summary = self.get_summary()
        
        logger.info(
            f"Task completed in {self._format_duration(summary['total_duration'])}",
            category=LogCategory.PERFORMANCE
        )
        
        if summary["iterations"]:
            avg_iter = sum(i["duration"] for i in summary["iterations"]) / len(summary["iterations"])
            logger.info(
                f"Average iteration time: {self._format_duration(avg_iter)}",
                category=LogCategory.PERFORMANCE
            )
