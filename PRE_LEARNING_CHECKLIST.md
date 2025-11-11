# Pre-Learning Checklist - Complete Analysis

**Date:** November 10, 2025
**Status:** ⚠️ **NOT READY FOR OVERNIGHT LEARNING**

---

## Executive Summary

I've thoroughly analyzed your changes and tested the system. **The code changes are good and compile correctly**, but there are **critical data issues** that will prevent meaningful learning.

### ✅ What's Working
1. ✅ Code compiles successfully
2. ✅ Anti-oscillation logic implemented
3. ✅ Time budget reduced (0.12-0.40s, much faster)
4. ✅ Teacher pipeline fixed and working
5. ✅ Agent runs and makes moves

### ❌ Critical Blockers for Learning

1. **❌ 100% Turn-0 Data** - All training data is from opening positions only
2. **❌ Zero Target Variance** - All MCTS evaluations return 0.0 (balanced start positions)
3. **❌ Zero-Weight Model** - Trained model has all weights = 0.0 (nothing to learn from)
4. **❌ Illegal Move Generation** - Agent generates "rivers cannot push rivers" errors

---

## Detailed Analysis

### 1. Code Changes Review ✅

**Time Budget (student_agent.cpp:926-939)**
```cpp
double base_budget = (current_turn_index_ <= 3) ? 0.12 : 0.20;
double time_limit = base_budget + 0.20 * scale;
double hard_cap = 0.40;  // Max 0.4s per move
```
- ✅ Correctly implemented
- ✅ Much faster than before (was 0.3-0.7s)
- ✅ Configurable via RS_TIME_CAP environment variable

**Anti-Oscillation (student_agent.cpp:1058-1099)**
```cpp
// Checks last 3 moves for reverse patterns
// Requires gain >= 0.15 to allow reverse moves
```
- ✅ Logic is sound
- ✅ Tracks recent_moves window correctly
- ✅ Seeks alternatives with best progress

**Anti-Repetition (student_agent.cpp:1035-1056)**
```cpp
// Blocks 3+ consecutive identical moves
consecutive_repeat_count tracking
```
- ✅ Correctly implemented
- ✅ Will prevent flip spam

**Flip Cooldown (student_agent.cpp:1004-1033)**
```cpp
// Prevents flipping same piece within 2 turns
// Unless gain >= 0.4
```
- ✅ Working as designed

### 2. Compilation ✅

```bash
make cmake-rebuild
```
**Result:** ✅ Success (no errors, no warnings)

### 3. Runtime Testing ⚠️

**Basic Functionality:** ✅ Pass
- Agent creates successfully
- Makes moves in 0.18-0.40s (good!)
- No crashes

**Game Behavior:** ⚠️ Issues Found
- ERROR: "rivers cannot push rivers" (multiple times)
- This is a move validation bug in the C++ move generator
- The MCTS is generating illegal river-push-river moves

**Observed Issues in 60-turn test game:**
- ~5-10 flip-backs still occurring (less than before, but not eliminated)
- Some pieces flip 3-4 times in 10 turns
- Agent still making questionable tactical choices

### 4. Training Data Quality ❌ CRITICAL

**Current State (logs/value_all.jsonl):**
```
Total positions: 2,535
Turn 0 positions: 100% (2,535/2,535)
Mid-game (10-30): 0%
Late-game (>30): 0%
Score diversity: 3 values (39.48, 39.63, 39.75)
```

**❌ PROBLEM:** All data is from turn 0 (starting positions)

**Teacher Targets (logs/value_teacher.jsonl):**
```
Total targets: 500
Unique values: 1
All targets: 0.0
```

**❌ PROBLEM:** Zero variance = impossible to learn

**Trained Model (models/value_v1.txt):**
```
All 16 weights: 0.0 ± 0.0
```

**❌ PROBLEM:** Model learned nothing (as expected with zero-variance targets)

---

## Why Learning Will Fail

### The Data Problem

When you train on only turn-0 positions:

1. **All positions are identical** (starting board state)
2. **All evaluations are identical** (MCTS correctly evaluates start as ~0.0)
3. **Ridge regression gets y = [0, 0, 0, ..., 0]**
4. **Solution: w = [0, 0, 0, ..., 0]** (mathematically correct!)

**This is not a bug - it's mathematically correct behavior given the data.**

### What You Need

Training data with **diversity**:

```
Turn distribution:
  Turn 0-5:    300 positions (15%)
  Turn 6-15:   600 positions (30%)
  Turn 16-30:  600 positions (30%)
  Turn 31+:    500 positions (25%)

Value distribution:
  Negative (<-0.5):  400 positions (20%)
  Near-zero (-0.5 to 0.5): 800 positions (40%)
  Positive (>0.5):   400 positions (20%)
  Strong (>2.0):     400 positions (20%)
```

---

## Action Plan Before Learning

### STEP 1: Generate Diverse Game Data ⚠️ REQUIRED

Option A: Run Many Self-Play Games (Recommended)
```bash
# Generate 50 games with varied opponents
python3 scripts/arena.py \
  --games-per-size 10 \
  --sizes small medium \
  --matchups self greedy random \
  --log \
  --rotate-port

# Check diversity
python3 - << 'PY'
import json
from collections import Counter
c = Counter()
for i, line in enumerate(open('logs/value_data.jsonl')):
    if i > 5000: break
    try:
        c[json.loads(line).get('meta', {}).get('turn', -1)] += 1
    except: pass
print(f"Turn 0: {c[0]}, Mid-game (10-30): {sum(c[t] for t in range(10, 31))}")
PY
```

Option B: Use Existing Game Replays
```bash
# If you have game logs from actual matches
ls logs/games/*.jsonl | head -20
# Extract positions from those
```

Option C: Generate Random Mid-Game Positions
```python
# Create script to play N random moves, then log
# This is least realistic but better than turn-0 only
```

### STEP 2: Regenerate Teacher Targets

```bash
# After you have diverse data
python3 scripts/teacher_value_fixed.py \
  --input logs/value_data.jsonl \
  --output logs/value_teacher_v2.jsonl \
  --seconds 0.3 \
  --max 5000

# Verify diversity
python3 - << 'PY'
import json
vals = set()
for i, l in enumerate(open('logs/value_teacher_v2.jsonl')):
    if i > 1000: break
    vals.add(round(json.loads(l).get('target', 0.0), 3))
print(f'Unique targets: {len(vals)}')
print(f'Sample: {sorted(list(vals))[:20]}')
PY

# Should see 50+ unique values, not all 0.0!
```

### STEP 3: Retrain Value Model

```bash
python3 ml/train_value.py \
  --input logs/value_teacher_v2.jsonl \
  --out models/value_v1.txt \
  --l2 1e-3

# Should see:
# - MAE > 0.0 (not 0.0000)
# - Pearson r = 0.3-0.7 (not nan)
# - Non-zero weights
```

### STEP 4: Fix Illegal Move Bug (Optional but Recommended)

The "rivers cannot push rivers" error indicates a bug in `generate_all_moves` or MCTS expansion.

**Quick Fix:**
```cpp
// In update_board() or push validation:
// Add check before generating push moves
if (target_cell.side == SIDE_RIVER && pushing_cell.side == SIDE_RIVER) {
    continue;  // Skip this push
}
```

### STEP 5: Validation Test

```bash
# Run 5 quick games to verify
python3 scripts/arena.py \
  --games-per-size 2 \
  --sizes small \
  --matchups self \
  --log

# Check for:
# 1. No "rivers cannot push rivers" errors
# 2. Reasonable move times (0.12-0.40s)
# 3. No excessive flip-backs
# 4. Game completes successfully
```

---

## Recommendations

### For Tonight/Tomorrow

**DO NOT** start overnight learning yet. You will waste compute time and get zero improvement.

**INSTEAD:**

1. **Generate 20-50 diverse games** (1-2 hours)
   ```bash
   nohup python3 scripts/arena.py \
     --games-per-size 15 \
     --sizes small medium \
     --matchups self greedy \
     --log > logs/generation.log 2>&1 &
   ```

2. **Run teacher pass** (2-3 hours for 5000 positions)
   ```bash
   nohup python3 scripts/teacher_value_fixed.py \
     --input logs/value_data.jsonl \
     --output logs/value_teacher_v2.jsonl \
     --seconds 0.3 \
     --max 5000 > logs/teacher.log 2>&1 &
   ```

3. **Train and validate** (5 minutes)
   ```bash
   python3 ml/train_value.py \
     --input logs/value_teacher_v2.jsonl \
     --out models/value_v1.txt

   # Verify non-zero weights!
   head models/value_v1.txt
   ```

4. **Then** start overnight learning loop

### For Overnight Learning (After Above)

Once you have a good model:

```bash
# Iterative improvement loop
for i in {1..10}; do
  echo "=== Iteration $i ==="

  # Generate more games with current model
  python3 scripts/arena.py \
    --games-per-size 10 \
    --sizes medium \
    --matchups self \
    --log

  # Append new data
  cat logs/value_data.jsonl >> logs/value_all_combined.jsonl

  # Teacher pass on combined data
  python3 scripts/teacher_value_fixed.py \
    --input logs/value_all_combined.jsonl \
    --output logs/value_teacher_iter${i}.jsonl \
    --seconds 0.3 \
    --max 10000

  # Retrain
  python3 ml/train_value.py \
    --input logs/value_teacher_iter${i}.jsonl \
    --out models/value_v${i}.txt

  # Update symlink
  ln -sf value_v${i}.txt models/value_v1.txt
done
```

---

## Summary Checklist

Before starting overnight learning:

- [ ] Generate diverse game data (not all turn-0)
- [ ] Verify turn distribution (check mid/late game positions exist)
- [ ] Run teacher pass on diverse data
- [ ] Verify target diversity (not all 0.0)
- [ ] Train value model
- [ ] Verify non-zero weights
- [ ] Fix "rivers cannot push rivers" bug
- [ ] Run validation games (no errors, reasonable behavior)
- [ ] Monitor first 2-3 iterations before leaving overnight

**Current Status:** ❌ Only item 7 partially done

**Estimated Time to Ready:** 4-6 hours

---

## Files to Monitor

During learning, watch these:

```bash
# Data generation
tail -f logs/generation.log

# Teacher progress
tail -f logs/teacher.log
wc -l logs/value_teacher_v2.jsonl

# Training output
tail -f logs/training.log

# Validation
grep "ERROR\|MAE\|Pearson" logs/*.log
```

---

## Contact Points

If issues arise:

1. **Illegal moves**: Check `student_agent.cpp` move generation (lines 200-600)
2. **Slow performance**: Adjust RS_TIME_CAP=0.25 for faster games
3. **Zero learning**: Verify data diversity with scripts above
4. **Crashes**: Check `logs/value_teacher.jsonl` for incomplete writes

---

**Ready for learning:** ❌ NO
**Estimated time to ready:** 4-6 hours
**Blocking issues:** Data diversity (critical)

**Next immediate step:** Generate diverse game data with arena.py
