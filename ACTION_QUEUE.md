# Action Queue System

The Action Queue system allows you to defer actions from callbacks and execute them safely after the current action completes.

## Features

- **Thread-safe**: Multiple threads can safely queue actions
- **Priority-based**: Higher priority actions execute first
- **Automatic processing**: Queue is automatically processed after each `act()` call
- **Command tracking**: All queued actions get proper command IDs
- **Circular dependency protection**: Prevents infinite loops
- **Error handling**: Failed actions don't stop the queue processing

## Basic Usage

### Queue an Action

```python
def post_action_callback(ctx: PostActionContext):
    if ctx.command_id == "my_command":
        # Queue an action instead of calling act() directly
        bot.queue_action(
            "click: next button",
            command_id="queued_next_click",
            priority=1,
            metadata={"triggered_by": ctx.command_id}
        )
```

### Queue Multiple Actions

```python
def post_action_callback(ctx: PostActionContext):
    if ctx.command_id == "my_command":
        # Queue multiple actions with different priorities
        bot.queue_action("click: close modal", priority=10)  # High priority
        bot.queue_action("click: next item", priority=1)     # Normal priority
        bot.queue_action("scroll: down", priority=0)         # Low priority
```

## Queue Management

### Manual Processing

```python
# Process queue manually (auto-processing is enabled by default)
bot.process_queue()

# Check queue status
print(f"Queue size: {bot.queue_size()}")
print(f"Queued actions: {bot.inspect_queue()}")

# Clear queue
bot.clear_queue()
```

### Disable Auto-Processing

```python
# Disable automatic queue processing
bot._auto_process_queue = False

# Process manually when needed
bot.process_queue()
```

## Command ID Hierarchy

Queued actions are properly tracked in the command ledger:

```
main_command (parent)
├─ main_command_cmd1 (normal execution)
├─ main_command_cmd2 (normal execution)
└─ queued_next_click (queued action) ← Properly linked
```

## Priority System

Actions are executed in priority order (higher priority first):

```python
bot.queue_action("urgent: close modal", priority=10)    # Executes first
bot.queue_action("normal: click button", priority=1)    # Executes second  
bot.queue_action("low: scroll down", priority=0)        # Executes last
```

## Error Handling

The queue system handles errors gracefully:

- Failed actions are logged but don't stop queue processing
- Each action is executed independently
- Error count is reported after processing

## Safety Features

### Circular Dependency Protection

```python
# This will raise an error
bot.queue_action("action1", command_id="same_id")
bot.queue_action("action2", command_id="same_id")  # ValueError: Circular dependency
```

### Max Queue Size

```python
# Queue has a default max size of 100 actions
# This prevents memory issues from runaway queuing
```

## Example: Post-Action Callback

```python
def post_action_callback(ctx: PostActionContext):
    if ctx.command_id == "click_ios_job_action_cmd6":
        print("About to click on the next job listing")
        # Queue the action instead of calling act() directly
        bot.queue_action(
            "click: next job listing",
            command_id="queued_next_job",
            priority=1,
            metadata={"triggered_by": ctx.command_id, "reason": "post_escape"}
        )

bot.action_executor.register_post_action_callback(post_action_callback)
```

## Benefits

1. **Non-blocking**: Callbacks don't interrupt current execution
2. **Safe**: No risk of execution flow disruption
3. **Flexible**: Multiple actions with different priorities
4. **Traceable**: All actions get proper command IDs
5. **Controllable**: Can disable auto-processing if needed

## When to Use

- **Post-action callbacks**: Queue actions triggered by other actions
- **Conditional actions**: Queue actions based on page state
- **Batch operations**: Queue multiple related actions
- **Priority actions**: Queue urgent actions that need to execute first

## When NOT to Use

- **Simple direct actions**: Just use `act()` directly
- **Immediate actions**: If you need the action to execute right now
- **Complex workflows**: Consider using ref commands instead
