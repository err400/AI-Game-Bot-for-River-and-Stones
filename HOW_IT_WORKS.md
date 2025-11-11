# How the Prep Script Works - Detailed Breakdown

## Your Question: "How does it handle different board sizes and times?"

**Answer: YES! It automatically handles everything.** Here's exactly how:

---

## 🎮 Board Sizes & Time Allocation

The script runs games on **2 different board sizes** with **automatic time scaling**:

### Board Configurations

| Size | Rows x Cols | Time per Player | Score Columns | Win Count |
|------|-------------|-----------------|---------------|-----------|
| **small** | 13 x 12 | 2 minutes (120s) | 4 middle cols | 4 pieces |
| **medium** | 15 x 14 | 3 minutes (180s) | 5 middle cols | 5 pieces |
| **large** | 17 x 16 | 4 minutes (240s) | 6 middle cols | 6 pieces |

**Your prep script uses:** `small` + `medium` (configurable via `SIZES` env var)

---

## 🔄 How It Generates Diverse Games

### Step-by-Step Process

#### 1. Script Parameters (Line 32-34)
```bash
GAMES_PER_SIZE=10        # 10 games per size
SIZES="small medium"     # 2 sizes
MATCHUPS="self greedy random"  # 3 matchup types
```

**Total games = 10 × 2 × 3 = 60 games!** (Not 30 as I said earlier)

#### 2. Arena Loop (arena.py:316-320)
```python
for size in args.sizes:              # small, medium
    for label, ccmd, scmd in scenarios:  # self, greedy, random
        for gi in range(games_per_size):  # 1..10
            create_game(size)         # Creates game with proper dimensions
```

#### 3. Game Creation (web_server.py:88-108)
```python
def create_game(board_size="small"):
    if board_size == "small":
        rows, cols = 13, 12
        time_left = 120.0  # 2 minutes
    elif board_size == "medium":
        rows, cols = 15, 14
        time_left = 180.0  # 3 minutes

    board = default_start_board(rows, cols)
    score_cols = score_cols_for(cols)
```

**Each player gets the appropriate time budget automatically!**

#### 4. Agent Time Management (student_agent.cpp:926-939)
```cpp
// Your C++ agent adjusts thinking time based on:
double base_budget = (current_turn <= 3) ? 0.12 : 0.20;
double scale = min(1.0, current_player_time / 30.0);
double time_limit = base_budget + 0.20 * scale;
double hard_cap = 0.40;  // Override with RS_TIME_CAP

// Examples:
// Turn 1, 120s left: 0.12s per move (fast early game)
// Turn 20, 80s left: 0.20 + 0.20*(80/30) = 0.73s → capped to 0.40s
// Turn 50, 30s left: 0.20 + 0.20*(30/30) = 0.40s
```

**Agent automatically speeds up when low on time!**

---

## 📊 Data Diversity Creation

### How Different Board Sizes Create Diversity

#### Small Board (13x12)
- **Faster games** → reach turn 20-40 quickly
- **More tactical** → less space, more direct battles
- **Good for:** Learning short-term tactics

#### Medium Board (15x14)
- **Longer games** → reach turn 30-60
- **More strategic** → more space for maneuvers
- **Good for:** Learning mid-game planning

### How Different Matchups Create Diversity

#### 1. Self-Play (student_cpp vs student_cpp)
```
Both agents: Your MCTS C++ agent
Positions: Turn 0 → Turn 60+
Why diverse: Both thinking hard, games last long
Value range: -2.0 to +2.0 (competitive)
```

#### 2. vs Greedy (student_cpp vs greedy_river)
```
Your agent vs Simple baseline
Positions: Turn 0 → Turn 30-50
Why diverse: You usually win, see winning positions
Value range: +1.0 to +5.0 (you dominating)
```

#### 3. vs Random (student_cpp vs random)
```
Your agent vs Random moves
Positions: Turn 0 → Turn 20-40
Why diverse: Quick wins, see various random states
Value range: +3.0 to +8.0 (you crushing it)
```

---

## 🎯 What Gets Logged

### Position Logging (bot_client.py with RS_LOG_VALUE=1)

Each move logs:
```json
{
  "features": [16 board evaluation features],
  "rows": 13,
  "cols": 12,
  "score_cols": [4, 5, 6, 7],
  "side": "circle",
  "board": [[cell data...]],
  "meta": {
    "game_id": "game_1699123456",
    "turn": 15,        // <-- THIS VARIES!
    "strategy": "student_cpp"
  },
  "final_score_from_side": 25.5
}
```

**Key point: `turn` varies from 0 to 60+**

### Turn Distribution (Expected)

From 60 games across small+medium, self+greedy+random:

```
Turn 0-5:     ~300 positions (15%)
Turn 6-10:    ~400 positions (20%)
Turn 11-20:   ~600 positions (30%)
Turn 21-40:   ~500 positions (25%)
Turn 41+:     ~200 positions (10%)
```

**This is DIVERSE data!** Not 100% turn-0 like before.

---

## ⏱️ Time Budget Flow

### Example: Small Board Game

```
Game Start:
├─ circle_time_left = 120.0s
├─ square_time_left = 120.0s
└─ Agent RS_TIME_CAP = 0.30s (from prep script)

Turn 1 (circle):
├─ current_turn_index_ = 0
├─ base_budget = 0.12s (early turn boost)
├─ Agent thinks: ~0.12s
└─ circle_time_left = 119.88s

Turn 5 (circle):
├─ current_turn_index_ = 2
├─ base_budget = 0.12s
├─ Agent thinks: ~0.12s
└─ circle_time_left = 119.4s

Turn 20 (circle):
├─ current_turn_index_ = 10
├─ base_budget = 0.20s
├─ scale = min(1.0, 115/30) = 1.0
├─ time_limit = 0.20 + 0.20*1.0 = 0.40s
├─ Capped by RS_TIME_CAP = 0.30s
├─ Agent thinks: ~0.30s
└─ circle_time_left = 112.0s

Turn 60 (circle):
├─ current_turn_index_ = 30
├─ circle_time_left = 90.0s
├─ base_budget = 0.20s
├─ scale = min(1.0, 90/30) = 1.0
├─ time_limit = 0.40s → capped to 0.30s
├─ Agent thinks: ~0.30s
└─ circle_time_left = 89.7s
```

**Agent never runs out of time!**

---

## 🔍 Verification Checks

### Step 1b: Turn Diversity (Line 78-94)
```bash
python3 - "$DATA_FILE" << 'PY'
import json
from collections import Counter
c = Counter()
for line in open(path):
    turn = json.loads(line)['meta']['turn']
    c[turn] += 1
print('Turn counts:', dict(c))
mid = sum(c[t] for t in range(10,31))
print('Mid-game (10..30) count:', mid)
PY
```

**Checks:** Are we getting mid/late game positions?

**Expected output:**
```
Turn counts: {0: 60, 1: 58, 5: 52, 10: 48, 15: 42, 20: 38, ...}
Mid-game (10..30) count: 450
```

### Step 2b: Target Diversity (Line 110-124)
```python
vals = set()
for line in open(teacher_file):
    target = json.loads(line)['target']
    vals.add(round(target, 3))
print('Unique targets:', len(vals))
print('Sample values:', sorted(list(vals))[:20])
```

**Checks:** Are MCTS evaluations diverse?

**Expected output:**
```
Unique targets: 127
Sample values: [-2.5, -1.8, -0.9, -0.3, 0.0, 0.3, 0.9, 1.2, 1.8, 2.5, ...]
```

**NOT:** `Unique targets: 1, Sample values: [0.0]` ← That's bad!

### Step 3b: Non-Zero Weights (Line 138-145)
```bash
nonzero=$(awk '{if ($1+0 != 0) c++} END{print c+0}' "$MODEL_FILE")
total=$(wc -l < "$MODEL_FILE")
echo "Weights non-zero: $nonzero / $total"
```

**Checks:** Did the model actually learn?

**Expected output:**
```
Weights non-zero: 16 / 16
```

**NOT:** `Weights non-zero: 0 / 16` ← That means all zeros!

---

## 🚦 What If It Fails?

### Failure Mode 1: Still 100% Turn-0
```
Turn counts: {0: 2535}
Mid-game count: 0
```

**Cause:** Games finishing too quickly
**Fix:** Increase `GAMES_PER_SIZE=20` or add more matchups

### Failure Mode 2: All Targets Still 0.0
```
Unique targets: 1
Sample values: [0.0]
```

**Cause:** Positions too similar (all balanced)
**Fix:** Add more diverse matchups, increase game length

### Failure Mode 3: Model Weights All Zero
```
Weights non-zero: 0 / 16
```

**Cause:** Targets had zero variance
**Fix:** Go back to Step 1, generate more diverse data

---

## 🎓 Summary

**Q: How does it handle different board sizes?**
✅ Arena passes `board_size="small"/"medium"` → Web server creates board with correct dimensions (13x12 or 15x14)

**Q: How does it handle different times?**
✅ Web server sets time per player based on size (2min or 3min)
✅ Agent automatically adjusts thinking time based on time remaining
✅ RS_TIME_CAP=0.30 from prep script caps per-move thinking

**Q: Does it create diverse data?**
✅ 60 total games (10 per size × 2 sizes × 3 matchups)
✅ Games reach turn 20-60+ depending on matchup
✅ Different board sizes → different strategic considerations
✅ Different opponents → different value ranges

**Q: Is everything automatic?**
✅ YES! Just run `./scripts/prepare_for_learning.sh`
✅ It builds, generates, evaluates, trains, verifies
✅ Exits with error if something wrong
✅ Takes 4-6 hours, fully automated

---

**Sab automatic hai! Board size, time, diversity - sab script sambhal legi!** 🚀
