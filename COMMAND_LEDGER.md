# Command Ledger System

The **Command Ledger** tracks all command executions with unique IDs, hierarchical relationships, and execution metadata.

## Features

- ✅ **Automatic ID Generation** - IDs auto-generated if not provided
- ✅ **Parent-Child Tracking** - Track nested commands (loops, conditionals, refs)
- ✅ **Execution Metadata** - Timing, status, errors
- ✅ **Command Lineage** - Full chain from root command to action
- ✅ **Query Capabilities** - Filter, search, and analyze commands
- ✅ **Execution Tree** - Visualize command hierarchy

## Basic Usage

### Using Command IDs in `act()`

```python
from vision_bot import BrowserVisionBot

bot = BrowserVisionBot()
bot.start()

# Option 1: Auto-generated ID
bot.act("click: the search button")
# Generates ID like: cmd_123456_1

# Option 2: Custom ID
bot.act("click: the login button", command_id="login-btn-click")

# Option 3: Use ID in nested commands
bot.act("for 3 times: click: next button", command_id="pagination-loop")
# Child commands auto-track parent_id
```

### Using Command IDs in `register_prompts()`

```python
bot.register_prompts([
    "click: job listing",
    "click: apply button",
    "click: submit"
], "job-app-flow", command_id="job-application")
```

## PostActionContext Integration

Callbacks now receive command ID information:

```python
from action_executor import PostActionContext

def my_callback(ctx: PostActionContext):
    print(f"Command ID: {ctx.command_id}")
    print(f"Command Lineage: {ctx.command_lineage}")
    
    # Example: Different behavior for different command sources
    if ctx.command_id and "login" in ctx.command_id:
        print("This action is part of the login flow")

bot.action_executor.register_post_action_callback(my_callback)
```

## Accessing the Ledger

The ledger is accessible via `bot.command_ledger`:

### Get Stats

```python
stats = bot.command_ledger.get_stats()
print(f"Total commands: {stats['total_commands']}")
print(f"By status: {stats['by_status']}")
print(f"Average duration: {stats['average_duration']:.3f}s")
```

### Query Commands

```python
from command_ledger import CommandStatus

# Get all completed commands
completed = bot.command_ledger.filter_records(status=CommandStatus.COMPLETED)

# Get all commands from a specific source
from_act = bot.command_ledger.filter_records(
    metadata_filter={"source": "act"}
)

# Get children of a command
children = bot.command_ledger.get_children("cmd_123456_1", recursive=True)
```

### Get Execution Tree

```python
tree = bot.command_ledger.get_execution_tree()
# Returns hierarchical structure of all commands
```

## Command Record Structure

Each command record contains:

```python
@dataclass
class CommandRecord:
    id: str                          # Unique command ID
    command: str                      # Command text/description  
    parent_id: Optional[str]         # Parent command ID (for nested commands)
    status: CommandStatus            # PENDING, RUNNING, COMPLETED, FAILED, SKIPPED
    started_at: Optional[float]      # Start timestamp
    completed_at: Optional[float]    # Completion timestamp
    error_message: Optional[str]     # Error if failed
    metadata: Dict[str, Any]         # Custom metadata
    child_ids: List[str]             # List of child command IDs
    
    @property
    def duration(self) -> Optional[float]:  # Duration in seconds
    
    @property
    def lineage(self) -> List[str]:  # Full lineage from root to this command
```

## Command Status Values

```python
class CommandStatus(str, Enum):
    PENDING = "pending"      # Registered but not started
    RUNNING = "running"      # Currently executing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"        # Failed to complete
    SKIPPED = "skipped"      # Skipped (e.g., conditional branch not taken)
```

## Example: Tracking Command Hierarchy

```python
bot = BrowserVisionBot()
bot.start()

# Root command
bot.act("for 3 times: click: next button", command_id="root")

# This creates a hierarchy like:
# root (for loop)
#  └─ root_iter_1 (click: next button)
#  └─ root_iter_2 (click: next button)  
#  └─ root_iter_3 (click: next button)

# Get the lineage
record = bot.command_ledger.get_record("root_iter_2")
lineage = bot.command_ledger.get_lineage(record.id)
# Returns: ["root", "root_iter_2"]
```

## Example: Filter and Analyze

```python
# Get all failed commands
failed = bot.command_ledger.filter_records(status=CommandStatus.FAILED)
for cmd in failed:
    print(f"Failed: {cmd.id} - {cmd.error_message}")

# Get commands from a specific parent
children = bot.command_ledger.filter_records(parent_id="root")

# Get commands with specific metadata
clicks = bot.command_ledger.filter_records(
    metadata_filter={"action_type": "click"}
)
```

## Example: Execution Tree Visualization

```python
import json

tree = bot.command_ledger.get_execution_tree()
print(json.dumps(tree, indent=2))

# Output:
# {
#   "roots": [
#     {
#       "id": "cmd_123_1",
#       "command": "for 3 times: click: next button",
#       "status": "completed",
#       "duration": 5.2,
#       "children": [
#         {
#           "id": "cmd_123_2",
#           "command": "click: next button",
#           "status": "completed",
#           "duration": 1.5,
#           "children": []
#         },
#         ...
#       ]
#     }
#   ]
# }
```

## Advanced: Custom Metadata

You can attach custom metadata to commands:

```python
# The metadata is automatically populated, but you can add more via the ledger API
cmd_id = bot.command_ledger.register_command(
    command="custom action",
    metadata={
        "source": "api",
        "user_id": "123",
        "session_id": "abc",
        "priority": "high"
    }
)
```

## Ledger Methods Reference

| Method | Description |
|--------|-------------|
| `register_command()` | Register a new command |
| `start_command()` | Mark command as started |
| `complete_command()` | Mark command as completed/failed |
| `get_record()` | Get a command record by ID |
| `get_current_command_id()` | Get currently executing command |
| `get_lineage()` | Get full lineage of command IDs |
| `get_lineage_commands()` | Get full lineage as CommandRecord objects |
| `get_children()` | Get child commands (recursive option) |
| `filter_records()` | Filter commands by criteria |
| `get_execution_tree()` | Get hierarchical tree structure |
| `get_stats()` | Get statistics about execution |
| `clear()` | Clear all records |

## Use Cases

### 1. Debugging Failed Commands

```python
failed = bot.command_ledger.filter_records(status=CommandStatus.FAILED)
for cmd in failed:
    lineage = bot.command_ledger.get_lineage_commands(cmd.id)
    print(f"\nFailed Command: {cmd.id}")
    print(f"Error: {cmd.error_message}")
    print("Lineage:")
    for ancestor in lineage:
        print(f"  {ancestor.id}: {ancestor.command}")
```

### 2. Performance Analysis

```python
stats = bot.command_ledger.get_stats()
print(f"Total execution time: {stats['total_duration']:.2f}s")
print(f"Average command duration: {stats['average_duration']:.3f}s")

# Find slow commands
all_commands = bot.command_ledger.filter_records(status=CommandStatus.COMPLETED)
slow_commands = [c for c in all_commands if c.duration and c.duration > 5.0]
for cmd in slow_commands:
    print(f"Slow: {cmd.id} took {cmd.duration:.2f}s - {cmd.command}")
```

### 3. Command Analytics

```python
# Count commands by source
from collections import Counter
all_commands = list(bot.command_ledger.records.values())
by_source = Counter(cmd.metadata.get("source") for cmd in all_commands)
print(f"Commands by source: {by_source}")

# Success rate
total = len(all_commands)
completed = len([c for c in all_commands if c.status == CommandStatus.COMPLETED])
success_rate = (completed / total * 100) if total > 0 else 0
print(f"Success rate: {success_rate:.1f}%")
```

## See Also

- `command_ledger.py` - Implementation
- `demo_command_ledger.py` - Working demo
- `POST_ACTION_HOOKS.md` - Post-action callbacks with command IDs

