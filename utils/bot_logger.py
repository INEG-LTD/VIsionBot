"""
Bot Logger - Comprehensive logging system for the automation bot.

This module provides structured logging for all bot activities including:
- Goal execution and evaluation
- Action execution (clicks, scrolls, etc.)
- Focus operations
- Error handling
- Performance metrics
"""
import os
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class LogLevel(str, Enum):
    """Log levels for different types of events"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class LogCategory(str, Enum):
    """Categories for different types of bot activities"""
    GOAL = "GOAL"
    ACTION = "ACTION"
    FOCUS = "FOCUS"
    NAVIGATION = "NAVIGATION"
    CONDITION = "CONDITION"
    ERROR = "ERROR"
    PERFORMANCE = "PERFORMANCE"
    SYSTEM = "SYSTEM"


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: str
    level: LogLevel
    category: LogCategory
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None
    success: Optional[bool] = None


class BotLogger:
    """Main logging class for the automation bot"""
    
    def __init__(self, log_file: str = "bot_automation.log", max_file_size_mb: int = 10):
        self.log_file = log_file
        self.max_file_size_mb = max_file_size_mb
        self.session_id = f"session_{int(time.time())}"
        self.start_time = time.time()
        self.entries: List[LogEntry] = []
        
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        
        # Initialize log file
        self._write_header()
    
    def _write_header(self):
        """Write session header to log file"""
        header = f"""
{'='*80}
BOT AUTOMATION SESSION STARTED
Session ID: {self.session_id}
Start Time: {datetime.now().isoformat()}
Log File: {self.log_file}
{'='*80}

"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(header)
    
    def _should_rotate_log(self) -> bool:
        """Check if log file should be rotated due to size"""
        try:
            size_mb = os.path.getsize(self.log_file) / (1024 * 1024)
            return size_mb > self.max_file_size_mb
        except:
            return False
    
    def _rotate_log(self):
        """Rotate log file when it gets too large"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{self.log_file}.{timestamp}.bak"
            os.rename(self.log_file, backup_file)
            self._write_header()
        except Exception as e:
            print(f"⚠️ Failed to rotate log file: {e}")
    
    def _format_entry(self, entry: LogEntry) -> str:
        """Format a log entry for output"""
        timestamp = entry.timestamp
        level = entry.level.value
        category = entry.category.value
        message = entry.message
        
        # Base log line
        log_line = f"[{timestamp}] {level:8} {category:12} {message}"
        
        # Add duration if available
        if entry.duration_ms is not None:
            log_line += f" (took {entry.duration_ms:.1f}ms)"
        
        # Add success indicator
        if entry.success is not None:
            status = "✅" if entry.success else "❌"
            log_line += f" {status}"
        
        # Add details if available
        if entry.details:
            details_str = json.dumps(entry.details, indent=2)
            log_line += f"\n    Details: {details_str}"
        
        return log_line
    
    def _write_entry(self, entry: LogEntry):
        """Write a single log entry to file"""
        try:
            # Check if we need to rotate the log
            if self._should_rotate_log():
                self._rotate_log()
            
            # Format and write the entry
            formatted_entry = self._format_entry(entry)
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(formatted_entry + "\n")
            
            # Also store in memory for session analysis
            self.entries.append(entry)
            
        except Exception as e:
            print(f"⚠️ Failed to write log entry: {e}")
    
    def log(self, level: LogLevel, category: LogCategory, message: str, 
            details: Optional[Dict[str, Any]] = None, duration_ms: Optional[float] = None, 
            success: Optional[bool] = None):
        """Log an event with structured information"""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            category=category,
            message=message,
            details=details,
            duration_ms=duration_ms,
            success=success
        )
        
        self._write_entry(entry)
    
    def log_goal_start(self, goal_description: str, goal_type: str = None):
        """Log the start of a goal execution"""
        details = {"goal_description": goal_description}
        if goal_type:
            details["goal_type"] = goal_type
        
        self.log(LogLevel.INFO, LogCategory.GOAL, f"Starting goal: {goal_description}", details)
    
    def log_goal_success(self, goal_description: str, duration_ms: float, details: Dict[str, Any] = None):
        """Log successful goal completion"""
        self.log(LogLevel.SUCCESS, LogCategory.GOAL, f"Goal completed: {goal_description}", 
                details, duration_ms, success=True)
    
    def log_goal_failure(self, goal_description: str, error: str, duration_ms: float = None):
        """Log goal failure"""
        details = {"error": error}
        self.log(LogLevel.ERROR, LogCategory.GOAL, f"Goal failed: {goal_description}", 
                details, duration_ms, success=False)
    
    def log_action(self, action_type: str, target: str, success: bool, duration_ms: float = None, details: Dict[str, Any] = None):
        """Log an action execution (click, scroll, etc.)"""
        status = "successful" if success else "failed"
        message = f"{action_type}: {target} - {status}"
        
        self.log(LogLevel.INFO, LogCategory.ACTION, message, details, duration_ms, success)
    
    def log_focus_operation(self, operation: str, target: str, success: bool, details: Dict[str, Any] = None):
        """Log focus operations"""
        status = "successful" if success else "failed"
        message = f"Focus {operation}: {target} - {status}"
        
        self.log(LogLevel.INFO, LogCategory.FOCUS, message, details, success=success)
    
    def log_condition_evaluation(self, condition: str, result: bool, reasoning: str = None):
        """Log condition evaluation"""
        details = {"condition": condition, "result": result}
        if reasoning:
            details["reasoning"] = reasoning
        
        self.log(LogLevel.INFO, LogCategory.CONDITION, f"Condition evaluated: {condition} = {result}", details)
    
    def log_navigation(self, action: str, target: str, success: bool, details: Dict[str, Any] = None):
        """Log navigation events"""
        status = "successful" if success else "failed"
        message = f"Navigation {action}: {target} - {status}"
        
        self.log(LogLevel.INFO, LogCategory.NAVIGATION, message, details, success=success)
    
    def log_error(self, error: str, context: str = None, details: Dict[str, Any] = None):
        """Log errors with context"""
        message = f"Error: {error}"
        if context:
            message += f" (Context: {context})"
        
        self.log(LogLevel.ERROR, LogCategory.ERROR, message, details)
    
    def log_performance(self, operation: str, duration_ms: float, details: Dict[str, Any] = None):
        """Log performance metrics"""
        message = f"Performance: {operation} took {duration_ms:.1f}ms"
        self.log(LogLevel.INFO, LogCategory.PERFORMANCE, message, details, duration_ms)
    
    def log_system(self, message: str, details: Dict[str, Any] = None):
        """Log system-level events"""
        self.log(LogLevel.INFO, LogCategory.SYSTEM, message, details)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session"""
        total_duration = time.time() - self.start_time
        
        # Count entries by level and category
        level_counts = {}
        category_counts = {}
        success_count = 0
        failure_count = 0
        
        for entry in self.entries:
            level_counts[entry.level.value] = level_counts.get(entry.level.value, 0) + 1
            category_counts[entry.category.value] = category_counts.get(entry.category.value, 0) + 1
            
            if entry.success is True:
                success_count += 1
            elif entry.success is False:
                failure_count += 1
        
        return {
            "session_id": self.session_id,
            "total_duration_seconds": total_duration,
            "total_entries": len(self.entries),
            "level_counts": level_counts,
            "category_counts": category_counts,
            "success_count": success_count,
            "failure_count": failure_count,
            "log_file": self.log_file
        }
    
    def write_session_summary(self):
        """Write session summary to log file"""
        summary = self.get_session_summary()
        
        summary_text = f"""
{'='*80}
SESSION SUMMARY
{'='*80}
Session ID: {summary['session_id']}
Total Duration: {summary['total_duration_seconds']:.1f} seconds
Total Log Entries: {summary['total_entries']}
Success Count: {summary['success_count']}
Failure Count: {summary['failure_count']}

Level Distribution:
"""
        for level, count in summary['level_counts'].items():
            summary_text += f"  {level}: {count}\n"
        
        summary_text += "\nCategory Distribution:\n"
        for category, count in summary['category_counts'].items():
            summary_text += f"  {category}: {count}\n"
        
        summary_text += f"\nLog File: {summary['log_file']}\n"
        summary_text += f"{'='*80}\n"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(summary_text)


# Global logger instance
_global_logger: Optional[BotLogger] = None

def get_logger() -> BotLogger:
    """Get the global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = BotLogger()
    return _global_logger

def set_logger(logger: BotLogger):
    """Set the global logger instance"""
    global _global_logger
    _global_logger = logger
