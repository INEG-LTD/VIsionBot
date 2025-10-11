# Advanced Command Ledger Features

The Command Ledger now includes three powerful features: **Persistence**, **Comparison**, and **Logger Integration**.

---

## 1. 💾 Persistence

Save and load command ledgers to analyze automation runs later.

### Save to File

```python
from vision_bot import BrowserVisionBot

bot = BrowserVisionBot()
bot.start()

# Run your automation
bot.act("click: button")
bot.act("type: 'hello' into: input")

# Save the ledger
bot.command_ledger.save_to_file("ledgers/session_2024_01_15.json")
```

### Load from File

```python
from command_ledger import CommandLedger

# Load a previous session
ledger = CommandLedger()
ledger.load_from_file("ledgers/session_2024_01_15.json")

# Analyze it
stats = ledger.get_stats()
print(f"Session had {stats['total_commands']} commands")
print(f"Total duration: {stats['total_duration']:.2f}s")

# Find failed commands
from command_ledger import CommandStatus
failed = ledger.filter_records(status=CommandStatus.FAILED)
for cmd in failed:
    print(f"Failed: {cmd.command} - {cmd.error_message}")
```

### Export Human-Readable Summary

```python
# Create a text summary of the entire session
bot.command_ledger.export_summary("ledgers/session_summary.txt")
```

**Output Example:**
```
================================================================================
COMMAND LEDGER SUMMARY
================================================================================

Total Commands: 15
Total Duration: 45.23s
Average Duration: 3.015s

Status Breakdown:
  completed: 12
  failed: 2
  skipped: 1

================================================================================
COMMAND DETAILS
================================================================================
✅ [cmd_123_1] for 3 times: click: next button (12.50s)
  ✅ [cmd_123_2] click: next button (3.20s)
  ✅ [cmd_123_3] click: next button (4.10s)
  ❌ [cmd_123_4] click: next button (5.20s)
     Error: Element not found

✅ [cmd_456_5] type: 'python developer' into: search (2.30s)
```

---

## 2. 🔍 Comparison

Compare two automation runs to identify differences, performance changes, and new failures.

### Compare Two Ledgers

```python
from command_ledger import CommandLedger

# Option 1: Load and compare from files
comparison = CommandLedger.load_and_compare(
    "ledgers/run1.json",
    "ledgers/run2.json"
)

# Option 2: Compare in-memory ledgers
comparison = CommandLedger.compare(ledger1, ledger2)

# Print summary
comparison.print_summary()
```

### Comparison Output

```
================================================================================
LEDGER COMPARISON SUMMARY
================================================================================

➕ Added Commands (2):
   + click: filters button
   + click: sort by relevance

➖ Removed Commands (1):
   - click: old button

🔄 Status Changes (3):
   completed → failed: click: login button
   failed → completed: type: password
   pending → skipped: click: optional step

🐌 Slower Commands (2):
   +45.2% (2.00s → 2.90s): click: search button
   +32.1% (1.50s → 1.98s): type: 'python developer' into: search

⚡ Faster Commands (1):
   -25.3% (4.00s → 2.99s): click: apply button

⚠️ Error Changes (1):
   click: submit
     Old: None
     New: Timeout waiting for element

📊 Overall Stats:
   Total Commands: +2
   Total Duration: +5.40s
   Avg Duration: +0.250s
================================================================================
```

### Use Cases for Comparison

**Regression Testing:**
```python
# Compare current run against baseline
comparison = CommandLedger.load_and_compare("baseline.json", "current.json")

if comparison.error_changes or comparison.status_changes:
    print("⚠️ Regression detected!")
    for change in comparison.status_changes:
        if change['new_status'] == 'failed':
            print(f"  New failure: {change['command']}")
```

**Performance Monitoring:**
```python
if comparison.slower_commands:
    print("⚠️ Performance degradation:")
    for cmd in comparison.slower_commands:
        if cmd['diff_percent'] > 50:  # More than 50% slower
            print(f"  {cmd['command']}: {cmd['diff_percent']:.1f}% slower")
```

**Workflow Evolution:**
```python
print(f"Workflow changes:")
print(f"  Added: {len(comparison.added_commands)} commands")
print(f"  Removed: {len(comparison.removed_commands)} commands")
print(f"  Modified: {len(comparison.status_changes)} commands")
```

---

## 3. 📝 Logger Integration

Automatically log all command events to the bot's logger for unified tracking.

### Enable Integration

```python
from vision_bot import BrowserVisionBot

bot = BrowserVisionBot()
bot.start()

# Enable automatic logging of command events
bot.command_ledger.enable_logger_integration(bot.logger)

# Now all command events are logged
bot.act("click: button")  # Logged: "Command started: click: button"
                          # Logged: "Command completed: click: button"
```

### Disable Integration

```python
bot.command_ledger.disable_logger_integration()
```

### What Gets Logged

With logger integration enabled, the following events are automatically logged:

1. **Command Registration**
   ```
   [INFO] [COMMAND] Command registered: click: the search button
   ```

2. **Command Start**
   ```
   [INFO] [COMMAND] Command started: click: the search button
   ```

3. **Command Completion**
   ```
   [INFO] [COMMAND] Command completed: click: the search button (duration: 2.5s)
   ```

4. **Command Failure**
   ```
   [ERROR] [COMMAND] Command failed: click: the search button (error: Element not found)
   ```

### Benefits

- **Unified Logging**: All bot activity (commands, goals, actions) in one place
- **Automatic**: No manual logging code needed
- **Session Logs**: Command tracking included in session log exports
- **Debugging**: Easier to correlate commands with actions and errors

### Access Session Logs

```python
# Write session log with command tracking
bot.write_session_log()

# Or access logger directly
print(bot.logger.get_logs())
```

---

## 🎯 Real-World Example: Job Application Tracking

```python
from vision_bot import BrowserVisionBot
from command_ledger import CommandLedger, CommandStatus
import datetime

# Session 1: Monday
bot = BrowserVisionBot()
bot.start()
bot.command_ledger.enable_logger_integration(bot.logger)

bot.goto("https://www.reed.co.uk/")
bot.act("for 10 times: click: job listing", command_id="apply-jobs-mon")
bot.command_ledger.save_to_file(f"ledgers/jobs_monday.json")
bot.end()

# Session 2: Tuesday
bot = BrowserVisionBot()
bot.start()
bot.command_ledger.enable_logger_integration(bot.logger)

bot.goto("https://www.reed.co.uk/")
bot.act("for 10 times: click: job listing", command_id="apply-jobs-tue")
bot.command_ledger.save_to_file(f"ledgers/jobs_tuesday.json")
bot.end()

# Analyze differences
comparison = CommandLedger.load_and_compare(
    "ledgers/jobs_monday.json",
    "ledgers/jobs_tuesday.json"
)

print("\n📊 Job Application Analysis:")
comparison.print_summary()

# Check if new errors appeared
if comparison.error_changes:
    print("\n⚠️ New issues detected:")
    for change in comparison.error_changes:
        if change['new_error']:
            print(f"  - {change['command']}: {change['new_error']}")

# Check performance
if comparison.slower_commands:
    print("\n🐌 Website slower today:")
    for cmd in comparison.slower_commands[:3]:
        print(f"  - {cmd['command'][:50]}: +{cmd['diff_percent']:.1f}%")
```

---

## 📁 File Structure

Recommended directory structure for ledger files:

```
project/
├── ledgers/
│   ├── 2024-01-15_session1.json
│   ├── 2024-01-15_session1_summary.txt
│   ├── 2024-01-15_session2.json
│   ├── 2024-01-15_session2_summary.txt
│   └── baseline.json
├── comparisons/
│   └── session1_vs_session2.txt
└── your_automation.py
```

### Naming Conventions

- **Session files**: `YYYY-MM-DD_description.json`
- **Summaries**: `YYYY-MM-DD_description_summary.txt`
- **Baselines**: `baseline_<feature>.json`

---

## 🔧 API Reference

### Persistence Methods

| Method | Description |
|--------|-------------|
| `save_to_file(filepath)` | Save ledger to JSON file |
| `load_from_file(filepath)` | Load ledger from JSON file |
| `export_summary(filepath)` | Export human-readable summary |

### Comparison Methods

| Method | Description |
|--------|-------------|
| `CommandLedger.compare(l1, l2)` | Compare two ledgers |
| `CommandLedger.load_and_compare(f1, f2)` | Load and compare files |
| `comparison.print_summary()` | Print comparison summary |

### Logger Integration Methods

| Method | Description |
|--------|-------------|
| `enable_logger_integration(logger)` | Enable auto-logging |
| `disable_logger_integration()` | Disable auto-logging |

---

## 🚀 Best Practices

1. **Save After Every Session**
   ```python
   bot.command_ledger.save_to_file(f"ledgers/session_{datetime.now()}.json")
   ```

2. **Create Baselines**
   ```python
   # After successful automation
   bot.command_ledger.save_to_file("ledgers/baseline_working.json")
   ```

3. **Regular Comparisons**
   ```python
   # Compare against baseline
   comp = CommandLedger.load_and_compare("baseline.json", "current.json")
   ```

4. **Enable Logger Integration**
   ```python
   # Always enable for production
   bot.command_ledger.enable_logger_integration(bot.logger)
   ```

5. **Clean Up Old Files**
   ```python
   # Keep only last 30 days
   import os
   from pathlib import Path
   from datetime import datetime, timedelta
   
   cutoff = datetime.now() - timedelta(days=30)
   for f in Path("ledgers").glob("*.json"):
        if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
            f.unlink()
   ```

---

## See Also

- `command_ledger.py` - Core implementation
- `COMMAND_LEDGER.md` - Basic ledger documentation
- `demo_ledger_features.py` - Working examples

