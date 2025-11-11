# Training Pipeline Diagnosis & Fix

## Problem Summary

Your `prepare_for_learning.sh` script failed with:
1. **All games timing out** without progressing past turn 0
2. **Only initial positions logged** (2,877 lines, all with `turn: 0`)
3. **Teacher process killed** (likely timeout/memory)
4. **Homogeneous targets** (all 0.0)
5. **Zero model weights** (0/15 non-zero)

## Root Causes

### 1. Game Timeouts
- **Symptom**: `[warn] game timeout; terminating bots …` for all 100 games
- **Analysis**: Games were timing out (300s limit) because:
  - Self-play with TIME_CAP=0.3s per move is slow (requires MCTS for both sides)
  - Some games encountered C++ exceptions: `ERROR: Unexpected error in game loop: vector`
  - Bots were crashing or hanging during MCTS search

### 2. Only Turn-0 Data
- **Consequence**: With only initial positions:
  - No game outcomes to learn from
  - Teacher produces `target: 0.0` for all positions (no terminal states)
  - Training data is completely homogeneous

### 3. Teacher Process Killed
- **Symptom**: `./scripts/prepare_for_learning.sh: line 101: 17010 Killed: 9`
- **Cause**: macOS killing process (likely memory or timeout)
- **Result**: Even if teacher ran, homogeneous input → homogeneous output

### 4. Zero Weights
- **Root**: Homogeneous targets (all 0.0) → gradient descent has nothing to learn
- **Result**: Model converges to all-zero weights

## Quick Fix: Use Faster Opponents

The fastest solution is to:
1. **Avoid self-play** (two MCTS agents is slow)
2. **Use greedy/random opponents** (near-instant moves)
3. **Reduce games/boards** to get faster results

I've created `scripts/quick_prep.sh` with these optimizations:

```bash
GAMES_PER_SIZE=5              # Fewer games
SIZES="small"                 # Small boards only
MATCHUPS="greedy random"      # Fast opponents (no self-play)
TIME_CAP="0.10"              # Very fast moves (100ms)
TIMEOUT="600"                # 10-minute timeout per game
TEACHER_SECONDS="0.5"        # Fast teacher evaluation
TEACHER_MAX="2000"           # Limit teacher rows
```

## How to Run

```bash
# Clean start
cd /Users/yuvikachaudhary/Desktop/Semester\ 5/COL333/COL333_A5

# Run the quick preparation (should take 5-15 minutes)
./scripts/quick_prep.sh

# If successful, you'll see:
# === SUCCESS ===
# Model: models/value_v1.txt
# Weights non-zero: X / 15
```

## What Changed

| Original | Quick Fix | Why |
|----------|-----------|-----|
| Self-play (MCTS vs MCTS) | Greedy/Random opponents | 100x faster |
| TIME_CAP=0.30s | TIME_CAP=0.10s | Faster data gen |
| small + medium boards | small only | Fewer positions |
| 10 games/matchup | 5 games/matchup | Faster completion |
| 300s timeout | 600s timeout | More tolerance |
| Teacher: 5000 rows | Teacher: 2000 rows | Faster eval |

## Expected Results

After running `quick_prep.sh`, you should see:
- **100-500 logged positions** with diverse turns (0-50+)
- **Diverse targets** from teacher (not all 0.0)
- **Non-zero weights** in trained model
- **Success message** at the end

## If It Still Fails

If games still timeout, the issue is likely a C++ crash in your MCTS implementation. To debug:

```bash
# Run a single test game manually
python3 client_server/web_server.py 9999 &
SERVER_PID=$!
sleep 2

RS_TIME_CAP=0.1 RS_LOG_VALUE=1 \
  python3 client_server/bot_client.py circle 9999 --strategy student_cpp &
BOT1_PID=$!

python3 client_server/bot_client.py square 9999 --strategy random &
BOT2_PID=$!

# Wait 30 seconds
sleep 30

# Check if bots are still running
ps -p $BOT1_PID $BOT2_PID $SERVER_PID

# Kill all
kill $SERVER_PID $BOT1_PID $BOT2_PID 2>/dev/null
```

If the student_cpp bot crashes, the issue is in your C++ MCTS code (likely a vector out-of-bounds or similar).

## Next Steps

After successful training:
1. **Verify model**: Check `models/value_v1.txt` has non-zero weights
2. **Test evaluation**: Run a quick test to see if value function works
3. **Generate more data**: If the quick version works, scale up:
   - Add `medium` boards
   - Increase `GAMES_PER_SIZE` to 10-20
   - Add self-play after verifying stability
4. **Iterate**: Run arena evaluation to see improvement

## Long-Term Solution

For production-quality training:
1. **Fix C++ crashes**: Debug the "vector" exception
2. **Add timeout handling**: Make MCTS interruptible
3. **Scale gradually**: Start with fast opponents, add complexity
4. **Monitor progress**: Track turn diversity, target variance, model metrics
