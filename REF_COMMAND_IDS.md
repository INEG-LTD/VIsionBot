# Ref Command ID Hierarchy

## How IDs Work with Refs

When you use refs, commands are organized in a **parent-child hierarchy**.

### Example

```python
# 1. Register prompts with a ref
bot.register_prompts([
    "click: job listing",
    "click: apply button",  
    "click: submit"
], "job-app-flow", command_id="job-app-ref")
```

**What gets created:**
- One command record: `job-app-ref` for the registration

```python
# 2. Execute the ref
bot.act("ref: job-app-flow", command_id="execute-job-app")
```

**What happens:**

```
execute-job-app (parent)
├─ execute-job-app_cmd1 → "click: job listing"
├─ execute-job-app_cmd2 → "click: apply button"
└─ execute-job-app_cmd3 → "click: submit"
```

Each command within the ref:
- Gets its own unique ID: `{parent_id}_cmd{N}`
- Has `parent_id` set to the ref command ID
- Can be queried, tracked, and analyzed independently
- Maintains lineage back to parent

## Benefits

### 1. Track Individual Commands

```python
# Find which specific command in the ref failed
ref_cmd = bot.command_ledger.get_record("execute-job-app")
for child_id in ref_cmd.child_ids:
    child = bot.command_ledger.get_record(child_id)
    if child.status == CommandStatus.FAILED:
        print(f"Failed: {child.command} - {child.error_message}")
```

### 2. Measure Performance Per Command

```python
# See which command in the ref is slowest
children = bot.command_ledger.get_children("execute-job-app")
slowest = max(children, key=lambda c: c.duration or 0)
print(f"Slowest: {slowest.command} ({slowest.duration:.2f}s)")
```

### 3. Understand Command Lineage

```python
# In post-action callbacks
def my_callback(ctx: PostActionContext):
    if ctx.command_lineage:
        print(f"Lineage: {' → '.join(ctx.command_lineage)}")
        # Example output:
        # Lineage: execute-job-app → execute-job-app_cmd2
```

### 4. Query by Ref

```python
# Get all commands that were part of a specific ref execution
ref_children = bot.command_ledger.get_children("execute-job-app", recursive=False)
print(f"Ref executed {len(ref_children)} commands")
```

## Visualization Example

Let's say you run this automation:

```python
bot.register_prompts([
    "click: search button",
    "type: 'python' into: search field"
], "search-flow", command_id="search-ref")

bot.act("for 3 times: ref: search-flow", command_id="search-loop")
```

**Resulting hierarchy:**

```
search-loop (for loop)
├─ search-loop_iter1 (ref: search-flow)
│  ├─ search-loop_iter1_cmd1 (click: search button)
│  └─ search-loop_iter1_cmd2 (type: 'python' into: search field)
├─ search-loop_iter2 (ref: search-flow)
│  ├─ search-loop_iter2_cmd1 (click: search button)
│  └─ search-loop_iter2_cmd2 (type: 'python' into: search field)
└─ search-loop_iter3 (ref: search-flow)
   ├─ search-loop_iter3_cmd1 (click: search button)
   └─ search-loop_iter3_cmd2 (type: 'python' into: search field)
```

You can query this:

```python
# Get all iterations
iterations = bot.command_ledger.get_children("search-loop")
print(f"Loop ran {len(iterations)} times")

# Get commands in first iteration
first_iter = bot.command_ledger.get_record("search-loop_iter1")
first_iter_cmds = bot.command_ledger.get_children(first_iter.id)
print(f"First iteration ran {len(first_iter_cmds)} commands")
```

## Comparison: With vs Without IDs

### Without Parent-Child Linking (Old Behavior)
```
cmd_123_1: ref: job-app-flow
cmd_123_2: click: job listing      ❌ No parent
cmd_123_3: click: apply button     ❌ No parent
cmd_123_4: click: submit            ❌ No parent
```
**Problem:** Can't tell which commands came from which ref

### With Parent-Child Linking (New Behavior)
```
execute-job-app: ref: job-app-flow
├─ execute-job-app_cmd1: click: job listing      ✅ parent_id = execute-job-app
├─ execute-job-app_cmd2: click: apply button     ✅ parent_id = execute-job-app
└─ execute-job-app_cmd3: click: submit            ✅ parent_id = execute-job-app
```
**Benefit:** Full hierarchy and traceability

## Real-World Example

```python
from vision_bot import BrowserVisionBot
from command_ledger import CommandStatus

bot = BrowserVisionBot()
bot.start()
bot.command_ledger.enable_logger_integration(bot.logger)

# Register job application flow
bot.register_prompts([
    "click: job listing",
    "click: apply button",
    "click: submit",
    "press: escape"
], "apply-job", command_id="apply-job-ref")

# Apply to multiple jobs
bot.act("for 5 times: ref: apply-job", command_id="apply-5-jobs")

# Analyze results
loop_cmd = bot.command_ledger.get_record("apply-5-jobs")
if loop_cmd:
    iterations = bot.command_ledger.get_children(loop_cmd.id)
    
    print(f"\n📊 Applied to {len(iterations)} jobs")
    
    for i, iteration in enumerate(iterations, 1):
        cmds = bot.command_ledger.get_children(iteration.id)
        failed = [c for c in cmds if c.status == CommandStatus.FAILED]
        
        if failed:
            print(f"\n❌ Job {i} failed:")
            for cmd in failed:
                print(f"   - {cmd.command}: {cmd.error_message}")
        else:
            print(f"✅ Job {i} succeeded ({iteration.duration:.2f}s)")
```

## Key Takeaways

1. **Each command gets its own ID** - Even within a ref
2. **IDs are hierarchical** - Parent-child relationships preserved
3. **IDs are predictable** - `{parent_id}_cmd{N}` format
4. **Full traceability** - Can trace any action back to its root command
5. **Powerful queries** - Filter, analyze, and debug at any level

## Testing

Run `test_ref_hierarchy.py` to see the hierarchy in action:

```bash
python test_ref_hierarchy.py
```

This will show you:
- The ref command and its children
- Full execution tree
- Saved ledger for inspection

