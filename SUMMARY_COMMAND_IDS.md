# Command ID System - Complete Summary

All command execution points now support command IDs with full hierarchy tracking.

## ✅ What's Been Implemented

### 1. **`act()` Method**
```python
bot.act("click: button", command_id="my-click")
# Creates: my-click
```

### 2. **`register_prompts()` Method**
```python
bot.register_prompts([
    "click: button",
    "type: text"
], "flow", command_id="my-flow")
# Creates: my-flow (stored for later)
```

### 3. **`ref:` Commands (NEW)**
```python
bot.act("ref: flow", command_id="execute-flow")
# Creates hierarchy:
# execute-flow (parent)
# ├─ execute-flow_cmd1 → "click: button"
# └─ execute-flow_cmd2 → "type: text"
```

### 4. **`on_new_page_load()` (NEW)**
```python
bot.on_new_page_load([
    "if: banner then: click: accept"
], command_id="page-cleanup")
# Creates hierarchy on each page load:
# page-cleanup (parent)
# └─ page-cleanup_action1 → "if: banner then..."
```

## Command Hierarchy Examples

### Simple Commands
```python
bot.act("click: button", command_id="btn-click")
```
**Result:**
```
btn-click
```

### Refs
```python
bot.register_prompts(["cmd1", "cmd2", "cmd3"], "flow")
bot.act("ref: flow", command_id="run-flow")
```
**Result:**
```
run-flow
├─ run-flow_cmd1
├─ run-flow_cmd2
└─ run-flow_cmd3
```

### For Loops with Refs
```python
bot.act("for 3 times: ref: flow", command_id="loop")
```
**Result:**
```
loop
├─ loop_iter1
│  ├─ loop_iter1_cmd1
│  ├─ loop_iter1_cmd2
│  └─ loop_iter1_cmd3
├─ loop_iter2
│  ├─ loop_iter2_cmd1
│  ├─ loop_iter2_cmd2
│  └─ loop_iter2_cmd3
└─ loop_iter3
   ├─ loop_iter3_cmd1
   ├─ loop_iter3_cmd2
   └─ loop_iter3_cmd3
```

### Page Load Actions
```python
bot.on_new_page_load(["action1", "action2"], command_id="cleanup")
bot.goto("https://site1.com")
bot.goto("https://site2.com")
```
**Result:**
```
cleanup (on site1.com)
├─ cleanup_action1
└─ cleanup_action2

cleanup (on site2.com)
├─ cleanup_action1
└─ cleanup_action2
```

## PostActionContext

All actions now include command context:

```python
from action_executor import PostActionContext

def my_callback(ctx: PostActionContext):
    print(f"Command ID: {ctx.command_id}")
    print(f"Lineage: {ctx.command_lineage}")
    
    # Example: Track which flow this is part of
    if ctx.command_lineage and "login-flow" in str(ctx.command_lineage):
        analytics.track("login_action", {
            "action": ctx.action_type.value,
            "success": ctx.success
        })

bot.action_executor.register_post_action_callback(my_callback)
```

## Querying Commands

### By Source
```python
# Get all page load actions
page_loads = bot.command_ledger.filter_records(
    metadata_filter={"source": "on_new_page_load"}
)

# Get all ref executions
refs = bot.command_ledger.filter_records(
    metadata_filter={"source": "act"}
)
```

### By Status
```python
from command_ledger import CommandStatus

failed = bot.command_ledger.filter_records(
    status=CommandStatus.FAILED
)
```

### By Hierarchy
```python
# Get all children of a command
parent = bot.command_ledger.get_record("my-flow")
children = bot.command_ledger.get_children(parent.id)

# Get full lineage
lineage = bot.command_ledger.get_lineage("my-flow_cmd2")
# Returns: ["my-flow", "my-flow_cmd2"]
```

## Complete Example

```python
from vision_bot import BrowserVisionBot
from command_ledger import CommandStatus

bot = BrowserVisionBot()
bot.start()
bot.command_ledger.enable_logger_integration(bot.logger)

# 1. Page load actions
bot.on_new_page_load([
    "if: cookie banner then: click: accept"
], command_id="cookies")

# 2. Register flows
bot.register_prompts([
    "click: listing",
    "click: apply",
    "click: submit"
], "job-flow", command_id="job-flow-ref")

# 3. Execute with hierarchy
bot.goto("https://jobs.com")  # Triggers "cookies"
bot.act("for 5 times: ref: job-flow", command_id="apply-jobs")

# 4. Analyze
loop = bot.command_ledger.get_record("apply-jobs")
iterations = bot.command_ledger.get_children(loop.id)

print(f"Applied to {len(iterations)} jobs")

for i, iteration in enumerate(iterations, 1):
    actions = bot.command_ledger.get_children(iteration.id)
    failed = [a for a in actions if a.status == CommandStatus.FAILED]
    
    if failed:
        print(f"Job {i} failed:")
        for action in failed:
            print(f"  - {action.command}: {action.error_message}")
    else:
        print(f"Job {i} succeeded ({iteration.duration:.2f}s)")

# 5. Save and compare
bot.command_ledger.save_to_file("ledgers/job_run_1.json")
bot.end()
```

## Documentation Files

- **`COMMAND_LEDGER.md`** - Basic ledger usage
- **`REF_COMMAND_IDS.md`** - How refs create hierarchies
- **`ON_NEW_PAGE_LOAD_IDS.md`** - Page load action tracking
- **`ADVANCED_LEDGER_FEATURES.md`** - Persistence, comparison, logger integration
- **`POST_ACTION_HOOKS.md`** - Using command IDs in callbacks

## Key Benefits

1. ✅ **Full Traceability** - Every action traces back to its command
2. ✅ **Hierarchy Tracking** - Understand parent-child relationships
3. ✅ **Performance Analysis** - Measure duration at any level
4. ✅ **Failure Debugging** - Pinpoint exactly what failed
5. ✅ **Analytics** - Track command execution across sessions
6. ✅ **Comparison** - Compare runs to find regressions
7. ✅ **Logger Integration** - Unified logging of all commands

