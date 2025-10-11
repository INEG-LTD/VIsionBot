"""
Command Ledger - Track command execution with IDs and hierarchy.
"""
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path


class CommandStatus(str, Enum):
    """Status of command execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CommandRecord:
    """Record of a command execution"""
    id: str
    command: str
    parent_id: Optional[str] = None
    status: CommandStatus = CommandStatus.PENDING
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    child_ids: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Get execution duration in seconds"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def lineage(self) -> List[str]:
        """Get list of IDs from root to this command"""
        # Note: This will be populated by the ledger
        return getattr(self, '_lineage', [self.id])


class CommandLedger:
    """
    Central ledger for tracking command execution with IDs.
    
    Features:
    - Automatic ID generation if not provided
    - Parent-child relationship tracking
    - Execution status and timing
    - Query capabilities
    """
    
    def __init__(self):
        self.records: Dict[str, CommandRecord] = {}
        self.execution_stack: List[str] = []  # Stack of currently executing command IDs
        self._counter = 0  # For sequential IDs
        self._logger_integration: Optional['LedgerLoggerIntegration'] = None
    
    def generate_id(self, prefix: str = "cmd") -> str:
        """Generate a unique command ID"""
        self._counter += 1
        timestamp = int(time.time() * 1000) % 1000000  # Last 6 digits of timestamp
        return f"{prefix}_{timestamp}_{self._counter}"
    
    def register_command(
        self,
        command: str,
        command_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a new command.
        
        Args:
            command: The command text/description
            command_id: Optional custom ID, auto-generated if None
            parent_id: Optional parent command ID for nested commands
            metadata: Optional metadata dict (e.g., {"source": "act", "type": "click"})
        
        Returns:
            The command ID (generated or provided)
        """
        if command_id is None:
            command_id = self.generate_id()
        
        # Auto-detect parent if not specified and we have an execution stack
        if parent_id is None and self.execution_stack:
            parent_id = self.execution_stack[-1]
        
        record = CommandRecord(
            id=command_id,
            command=command,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        
        self.records[command_id] = record
        
        # Add to parent's children
        if parent_id and parent_id in self.records:
            self.records[parent_id].child_ids.append(command_id)
        
        return command_id
    
    def start_command(self, command_id: str) -> None:
        """Mark a command as started"""
        if command_id in self.records:
            record = self.records[command_id]
            record.status = CommandStatus.RUNNING
            record.started_at = time.time()
            self.execution_stack.append(command_id)
    
    def complete_command(
        self,
        command_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Mark a command as completed"""
        if command_id in self.records:
            record = self.records[command_id]
            record.status = CommandStatus.COMPLETED if success else CommandStatus.FAILED
            record.completed_at = time.time()
            record.error_message = error_message
            
            # Remove from execution stack
            if command_id in self.execution_stack:
                self.execution_stack.remove(command_id)
    
    def get_record(self, command_id: str) -> Optional[CommandRecord]:
        """Get a command record by ID"""
        return self.records.get(command_id)
    
    def get_current_command_id(self) -> Optional[str]:
        """Get the currently executing command ID"""
        return self.execution_stack[-1] if self.execution_stack else None
    
    def get_lineage(self, command_id: str) -> List[str]:
        """Get the full lineage of command IDs from root to this command"""
        lineage = []
        current_id = command_id
        
        # Prevent infinite loops
        visited = set()
        
        while current_id and current_id not in visited:
            lineage.insert(0, current_id)
            visited.add(current_id)
            
            record = self.records.get(current_id)
            if not record or not record.parent_id:
                break
            current_id = record.parent_id
        
        return lineage
    
    def get_lineage_commands(self, command_id: str) -> List[CommandRecord]:
        """Get the full lineage of command records"""
        lineage_ids = self.get_lineage(command_id)
        return [self.records[cid] for cid in lineage_ids if cid in self.records]
    
    def get_children(self, command_id: str, recursive: bool = False) -> List[CommandRecord]:
        """Get child commands of a command"""
        record = self.records.get(command_id)
        if not record:
            return []
        
        children = [self.records[cid] for cid in record.child_ids if cid in self.records]
        
        if recursive:
            for child in list(children):
                children.extend(self.get_children(child.id, recursive=True))
        
        return children
    
    def filter_records(
        self,
        status: Optional[CommandStatus] = None,
        parent_id: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[CommandRecord]:
        """Filter records by criteria"""
        results = []
        
        for record in self.records.values():
            # Filter by status
            if status and record.status != status:
                continue
            
            # Filter by parent
            if parent_id is not None and record.parent_id != parent_id:
                continue
            
            # Filter by metadata
            if metadata_filter:
                match = all(
                    record.metadata.get(k) == v
                    for k, v in metadata_filter.items()
                )
                if not match:
                    continue
            
            results.append(record)
        
        return results
    
    def get_execution_tree(self, root_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a tree representation of command execution"""
        def build_tree(command_id: str) -> Dict[str, Any]:
            record = self.records.get(command_id)
            if not record:
                return {}
            
            return {
                "id": record.id,
                "command": record.command,
                "status": record.status.value,
                "duration": record.duration,
                "children": [build_tree(cid) for cid in record.child_ids],
            }
        
        if root_id:
            return build_tree(root_id)
        
        # Get all root commands (no parent)
        roots = [r for r in self.records.values() if r.parent_id is None]
        return {
            "roots": [build_tree(r.id) for r in roots]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about command execution"""
        total = len(self.records)
        by_status = {}
        for record in self.records.values():
            by_status[record.status.value] = by_status.get(record.status.value, 0) + 1
        
        completed = [r for r in self.records.values() if r.duration is not None]
        total_duration = sum(r.duration for r in completed)
        avg_duration = total_duration / len(completed) if completed else 0
        
        return {
            "total_commands": total,
            "by_status": by_status,
            "total_duration": total_duration,
            "average_duration": avg_duration,
            "currently_executing": len(self.execution_stack),
        }
    
    def clear(self) -> None:
        """Clear all records"""
        self.records.clear()
        self.execution_stack.clear()
        self._counter = 0
    
    def enable_logger_integration(self, bot_logger) -> None:
        """
        Enable automatic logging of command events to bot logger.
        
        Args:
            bot_logger: The BotLogger instance
        """
        if self._logger_integration is None:
            self._logger_integration = LedgerLoggerIntegration(self, bot_logger)
            print("✅ Command ledger logger integration enabled")
    
    def disable_logger_integration(self) -> None:
        """Disable logger integration"""
        if self._logger_integration:
            self._logger_integration.uninstall()
            self._logger_integration = None
            print("✅ Command ledger logger integration disabled")
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __repr__(self) -> str:
        return f"CommandLedger({len(self.records)} commands)"
    
    # =========================================================================
    # Persistence Methods
    # =========================================================================
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the ledger to a JSON file for later analysis.
        
        Args:
            filepath: Path to save the ledger (e.g., "session_2024_01_15.json")
        """
        path = Path(filepath)
        
        # Convert records to dict format
        data = {
            "version": "1.0",
            "saved_at": time.time(),
            "session_start": self.records[list(self.records.keys())[0]].started_at if self.records else None,
            "stats": self.get_stats(),
            "records": {}
        }
        
        for record_id, record in self.records.items():
            data["records"][record_id] = {
                "id": record.id,
                "command": record.command,
                "parent_id": record.parent_id,
                "status": record.status.value,
                "started_at": record.started_at,
                "completed_at": record.completed_at,
                "error_message": record.error_message,
                "metadata": record.metadata,
                "child_ids": record.child_ids,
                "duration": record.duration,
            }
        
        # Save to file
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved command ledger to {filepath}")
    
    def load_from_file(self, filepath: str) -> None:
        """
        Load a ledger from a JSON file.
        
        Args:
            filepath: Path to the ledger file to load
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Ledger file not found: {filepath}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Clear existing records
        self.records.clear()
        
        # Load records
        for record_id, record_data in data["records"].items():
            record = CommandRecord(
                id=record_data["id"],
                command=record_data["command"],
                parent_id=record_data["parent_id"],
                status=CommandStatus(record_data["status"]),
                started_at=record_data["started_at"],
                completed_at=record_data["completed_at"],
                error_message=record_data["error_message"],
                metadata=record_data["metadata"],
                child_ids=record_data["child_ids"],
            )
            self.records[record_id] = record
        
        print(f"📂 Loaded command ledger from {filepath} ({len(self.records)} commands)")
    
    def export_summary(self, filepath: str) -> None:
        """
        Export a human-readable summary of the ledger.
        
        Args:
            filepath: Path to save the summary (e.g., "summary.txt")
        """
        path = Path(filepath)
        stats = self.get_stats()
        
        lines = []
        lines.append("=" * 80)
        lines.append("COMMAND LEDGER SUMMARY")
        lines.append("=" * 80)
        lines.append(f"\nTotal Commands: {stats['total_commands']}")
        lines.append(f"Total Duration: {stats['total_duration']:.2f}s")
        lines.append(f"Average Duration: {stats['average_duration']:.3f}s")
        lines.append(f"\nStatus Breakdown:")
        for status, count in stats['by_status'].items():
            lines.append(f"  {status}: {count}")
        
        lines.append(f"\n{'='*80}")
        lines.append("COMMAND DETAILS")
        lines.append("=" * 80)
        
        # Get root commands
        roots = [r for r in self.records.values() if r.parent_id is None]
        
        def write_tree(record: CommandRecord, indent: int = 0):
            prefix = "  " * indent
            status_icon = {
                CommandStatus.COMPLETED: "✅",
                CommandStatus.FAILED: "❌",
                CommandStatus.RUNNING: "⏳",
                CommandStatus.PENDING: "⏸️",
                CommandStatus.SKIPPED: "⏭️",
            }.get(record.status, "❓")
            
            duration_str = f"{record.duration:.2f}s" if record.duration else "N/A"
            lines.append(f"{prefix}{status_icon} [{record.id}] {record.command} ({duration_str})")
            
            if record.error_message:
                lines.append(f"{prefix}   Error: {record.error_message}")
            
            for child_id in record.child_ids:
                if child_id in self.records:
                    write_tree(self.records[child_id], indent + 1)
        
        for root in roots:
            write_tree(root)
            lines.append("")
        
        # Save to file
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write("\n".join(lines))
        
        print(f"📄 Exported summary to {filepath}")
    
    # =========================================================================
    # Comparison Methods
    # =========================================================================
    
    @staticmethod
    def compare(ledger1: 'CommandLedger', ledger2: 'CommandLedger') -> 'LedgerComparison':
        """
        Compare two command ledgers to identify differences.
        
        Args:
            ledger1: First ledger (e.g., previous run)
            ledger2: Second ledger (e.g., current run)
        
        Returns:
            LedgerComparison object with detailed differences
        """
        comparison = LedgerComparison()
        
        # Get all command texts from both ledgers
        commands1 = {r.command: r for r in ledger1.records.values()}
        commands2 = {r.command: r for r in ledger2.records.values()}
        
        # Find added/removed commands
        comparison.added_commands = [
            commands2[cmd] for cmd in set(commands2.keys()) - set(commands1.keys())
        ]
        comparison.removed_commands = [
            commands1[cmd] for cmd in set(commands1.keys()) - set(commands2.keys())
        ]
        
        # Compare common commands
        common_commands = set(commands1.keys()) & set(commands2.keys())
        
        for cmd in common_commands:
            r1 = commands1[cmd]
            r2 = commands2[cmd]
            
            # Status changes
            if r1.status != r2.status:
                comparison.status_changes.append({
                    "command": cmd,
                    "old_status": r1.status.value,
                    "new_status": r2.status.value,
                })
            
            # Duration changes
            if r1.duration and r2.duration:
                diff_percent = ((r2.duration - r1.duration) / r1.duration) * 100
                
                if diff_percent > 20:  # More than 20% slower
                    comparison.slower_commands.append({
                        "command": cmd,
                        "old_duration": r1.duration,
                        "new_duration": r2.duration,
                        "diff_percent": diff_percent,
                    })
                elif diff_percent < -20:  # More than 20% faster
                    comparison.faster_commands.append({
                        "command": cmd,
                        "old_duration": r1.duration,
                        "new_duration": r2.duration,
                        "diff_percent": diff_percent,
                    })
            
            # Error changes
            if r1.error_message != r2.error_message:
                comparison.error_changes.append({
                    "command": cmd,
                    "old_error": r1.error_message,
                    "new_error": r2.error_message,
                })
        
        # Overall stats comparison
        stats1 = ledger1.get_stats()
        stats2 = ledger2.get_stats()
        
        comparison.stats_diff = {
            "total_commands": stats2["total_commands"] - stats1["total_commands"],
            "total_duration": stats2["total_duration"] - stats1["total_duration"],
            "avg_duration": stats2["average_duration"] - stats1["average_duration"],
        }
        
        return comparison
    
    @staticmethod
    def load_and_compare(filepath1: str, filepath2: str) -> 'LedgerComparison':
        """
        Load two ledgers from files and compare them.
        
        Args:
            filepath1: Path to first ledger file
            filepath2: Path to second ledger file
        
        Returns:
            LedgerComparison object
        """
        ledger1 = CommandLedger()
        ledger1.load_from_file(filepath1)
        
        ledger2 = CommandLedger()
        ledger2.load_from_file(filepath2)
        
        return CommandLedger.compare(ledger1, ledger2)


@dataclass
class LedgerComparison:
    """Results of comparing two command ledgers"""
    added_commands: List[CommandRecord] = field(default_factory=list)
    removed_commands: List[CommandRecord] = field(default_factory=list)
    status_changes: List[Dict[str, Any]] = field(default_factory=list)
    slower_commands: List[Dict[str, Any]] = field(default_factory=list)
    faster_commands: List[Dict[str, Any]] = field(default_factory=list)
    error_changes: List[Dict[str, Any]] = field(default_factory=list)
    stats_diff: Dict[str, float] = field(default_factory=dict)
    
    def print_summary(self) -> None:
        """Print a human-readable summary of the comparison"""
        print("\n" + "=" * 80)
        print("LEDGER COMPARISON SUMMARY")
        print("=" * 80)
        
        if self.added_commands:
            print(f"\n➕ Added Commands ({len(self.added_commands)}):")
            for cmd in self.added_commands[:5]:  # Show first 5
                print(f"   + {cmd.command}")
            if len(self.added_commands) > 5:
                print(f"   ... and {len(self.added_commands) - 5} more")
        
        if self.removed_commands:
            print(f"\n➖ Removed Commands ({len(self.removed_commands)}):")
            for cmd in self.removed_commands[:5]:
                print(f"   - {cmd.command}")
            if len(self.removed_commands) > 5:
                print(f"   ... and {len(self.removed_commands) - 5} more")
        
        if self.status_changes:
            print(f"\n🔄 Status Changes ({len(self.status_changes)}):")
            for change in self.status_changes[:5]:
                print(f"   {change['old_status']} → {change['new_status']}: {change['command']}")
            if len(self.status_changes) > 5:
                print(f"   ... and {len(self.status_changes) - 5} more")
        
        if self.slower_commands:
            print(f"\n🐌 Slower Commands ({len(self.slower_commands)}):")
            for cmd in sorted(self.slower_commands, key=lambda x: x['diff_percent'], reverse=True)[:5]:
                print(f"   +{cmd['diff_percent']:.1f}% ({cmd['old_duration']:.2f}s → {cmd['new_duration']:.2f}s): {cmd['command'][:60]}")
        
        if self.faster_commands:
            print(f"\n⚡ Faster Commands ({len(self.faster_commands)}):")
            for cmd in sorted(self.faster_commands, key=lambda x: x['diff_percent'])[:5]:
                print(f"   {cmd['diff_percent']:.1f}% ({cmd['old_duration']:.2f}s → {cmd['new_duration']:.2f}s): {cmd['command'][:60]}")
        
        if self.error_changes:
            print(f"\n⚠️ Error Changes ({len(self.error_changes)}):")
            for change in self.error_changes[:5]:
                print(f"   {change['command'][:60]}")
                print(f"     Old: {change['old_error']}")
                print(f"     New: {change['new_error']}")
        
        if self.stats_diff:
            print(f"\n📊 Overall Stats:")
            print(f"   Total Commands: {self.stats_diff.get('total_commands', 0):+d}")
            print(f"   Total Duration: {self.stats_diff.get('total_duration', 0):+.2f}s")
            print(f"   Avg Duration: {self.stats_diff.get('avg_duration', 0):+.3f}s")
        
        print("=" * 80 + "\n")


# =============================================================================
# Logger Integration
# =============================================================================

class LedgerLoggerIntegration:
    """
    Integration between CommandLedger and BotLogger.
    Automatically logs command events to the bot's logger.
    """
    
    def __init__(self, command_ledger: CommandLedger, bot_logger):
        """
        Initialize the integration.
        
        Args:
            command_ledger: The CommandLedger instance to monitor
            bot_logger: The BotLogger instance to log to
        """
        self.ledger = command_ledger
        self.logger = bot_logger
        self._original_register = command_ledger.register_command
        self._original_start = command_ledger.start_command
        self._original_complete = command_ledger.complete_command
        
        # Wrap ledger methods to add logging
        self._install_hooks()
    
    def _install_hooks(self) -> None:
        """Install logging hooks into the ledger"""
        
        def logged_register(command: str, command_id: Optional[str] = None, 
                           parent_id: Optional[str] = None, 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
            # Call original method
            cmd_id = self._original_register(command, command_id, parent_id, metadata)
            
            # Log to bot logger
            try:
                from utils.bot_logger import LogCategory
                self.logger.log(
                    "INFO",
                    LogCategory.COMMAND,
                    f"Command registered: {command}",
                    {"command_id": cmd_id, "parent_id": parent_id, "metadata": metadata}
                )
            except Exception:
                pass  # Don't break if logger fails
            
            return cmd_id
        
        def logged_start(command_id: str) -> None:
            # Call original method
            self._original_start(command_id)
            
            # Log to bot logger
            try:
                from utils.bot_logger import LogCategory
                record = self.ledger.get_record(command_id)
                if record:
                    self.logger.log(
                        "INFO",
                        LogCategory.COMMAND,
                        f"Command started: {record.command}",
                        {"command_id": command_id}
                    )
            except Exception:
                pass
        
        def logged_complete(command_id: str, success: bool = True, 
                           error_message: Optional[str] = None) -> None:
            # Call original method
            self._original_complete(command_id, success, error_message)
            
            # Log to bot logger
            try:
                from utils.bot_logger import LogCategory
                record = self.ledger.get_record(command_id)
                if record:
                    level = "INFO" if success else "ERROR"
                    status = "completed" if success else "failed"
                    
                    self.logger.log(
                        level,
                        LogCategory.COMMAND,
                        f"Command {status}: {record.command}",
                        {
                            "command_id": command_id,
                            "duration": record.duration,
                            "error": error_message
                        }
                    )
            except Exception:
                pass
        
        # Replace ledger methods with logged versions
        self.ledger.register_command = logged_register
        self.ledger.start_command = logged_start
        self.ledger.complete_command = logged_complete
    
    def uninstall(self) -> None:
        """Remove logging hooks from the ledger"""
        self.ledger.register_command = self._original_register
        self.ledger.start_command = self._original_start
        self.ledger.complete_command = self._original_complete

