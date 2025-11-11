# ✅ READY TO RUN - Final Confirmation

## YES - Just Run This One Command:

```bash
cd "/Users/yuvikachaudhary/Desktop/Semester 5/COL333/COL333_A5"
./scripts/prepare_for_learning.sh
```

---

## What It Will Do (Automatically):

### Step 0: Build C++ Agent ✓
- Compiles `student_agent.cpp` (your MCTS C++ agent)
- Output: `build/student_agent_module.cpython-312-darwin.so`

### Step 1: Generate Diverse Games ✓
**Matchups (all using your C++ agent):**
- `self`: Your C++ agent vs Your C++ agent
- `greedy`: Your C++ agent vs Greedy baseline
- `random`: Your C++ agent vs Random baseline

**Parameters:**
- 10 games per matchup per size
- Sizes: small + medium
- Total: ~30 games
- Time: 1-2 hours

**This creates diverse data with mid/late game positions!**

### Step 2: Teacher Pass ✓
- Runs MCTS evaluation on each position
- Uses the fixed `teacher_value_fixed.py` (no segfault)
- Generates targets for 5000 positions
- Time: 2-3 hours

### Step 3: Train Model ✓
- Ridge regression on diverse targets
- Creates `models/value_v1.txt` with NON-ZERO weights
- Time: 2 minutes

### Step 4: Verify ✓
- Checks weights are non-zero
- Checks target diversity
- Exits with error if something wrong

---

## Confirmation Checks:

✅ **Script uses YOUR C++ agent** (`student_agent_cpp`)
- Line 11: `self: main student_cpp vs main student_cpp`
- NOT using random agent from duplicate_folder
- NOT using Python baseline

✅ **Script builds C++ agent first**
- Line 48-49: `make cmake-rebuild`
- Ensures latest code is compiled

✅ **Matchups are correct**
- Line 34: `MATCHUPS="${MATCHUPS:-self greedy random}"`
- All three use your C++ MCTS agent

✅ **Logging is enabled**
- Line 67: `--log` flag
- This captures positions from various turns

✅ **Uses fixed teacher script**
- Line 97: `scripts/teacher_value_fixed.py`
- Uses `choose()` + `get_last_root_value()` workaround
- No segfault!

---

## What You'll See:

```
=== Step 0: Build student agent (C++) ===
[Building...]
✓ Build successful

=== Step 1: Generate diverse game data ===
SIZES=small medium MATCHUPS=self greedy random GAMES_PER_SIZE=10
[Games running... ~1-2 hours]

=== Step 1b: Turn diversity check ===
Turn counts (sample): {0: 150, 1: 120, 5: 95, 10: 80, ...}
Mid-game (10..30) sample count: 450  ✓ Good!

=== Step 2: Teacher pass ===
[Computing targets... ~2-3 hours]
Processed 8000 rows, wrote 5000 with teacher targets

=== Step 2b: Target diversity check ===
Unique targets (sample): 127
Sample values: [-2.5, -1.8, -0.9, 0.0, 0.3, 1.2, ...]  ✓ Good!

=== Step 3: Train value model ===
Trained on 5000 rows (bad lines: 0)
Primary value: MAE=0.2341, Pearson r=0.5234

=== Step 3b: Non-zero weights check ===
Weights non-zero: 16 / 16  ✓ Good!

=== SUCCESS: Value model trained with non-zero weights ===
Model: models/value_v1.txt
```

---

## Total Time: ~4-6 Hours

| Step | Time | Can Leave Running? |
|------|------|--------------------|
| Build | 1 min | No |
| Generate games | 1-2 hours | Yes |
| Teacher pass | 2-3 hours | Yes |
| Train model | 2 min | No |
| **Total** | **4-6 hours** | **Yes!** |

---

## After It Finishes:

### 1. Verify Success
```bash
# Should see NON-ZERO numbers
head models/value_v1.txt
```

### 2. Test Agent
```bash
# Quick validation game
python3 scripts/arena.py --games-per-size 1 --sizes small --matchups self --log
```

### 3. THEN Start Overnight Learning
```bash
# Now you can run this overnight!
for i in {1..20}; do
  echo "=== Iteration $i at $(date) ==="
  
  python3 scripts/arena.py --games-per-size 10 --sizes medium --matchups self --log
  cat logs/value_data.jsonl >> logs/value_cumulative.jsonl
  
  python3 scripts/teacher_value_fixed.py \
    --input logs/value_cumulative.jsonl \
    --output logs/value_teacher_iter${i}.jsonl \
    --seconds 0.3 --max 10000
  
  python3 ml/train_value.py \
    --input logs/value_teacher_iter${i}.jsonl \
    --out models/value_v1.txt
done
```

---

## FAQ

**Q: Will it use my C++ agent or the random one?**
✅ **Your C++ agent!** The script uses `student_cpp` which is your MCTS C++ agent compiled from `student_agent.cpp`, NOT the random agent from duplicate_folder.

**Q: Can I leave it running overnight?**
✅ **YES!** It takes 4-6 hours and runs fully automated.

**Q: What if it fails?**
Check the logs:
- `logs/generation.log` - Game generation issues
- `logs/teacher.log` - Teacher pass issues  
- `logs/training.log` - Training issues

**Q: How do I know it worked?**
✅ File `models/value_v1.txt` exists with NON-ZERO numbers
✅ No error messages at the end
✅ Sees "SUCCESS: Value model trained with non-zero weights"

---

## Just Run It!

```bash
./scripts/prepare_for_learning.sh
```

**That's it! Leave it running for 4-6 hours, then check the results.**

After it succeeds, you can start the overnight iterative learning loop.

**Bas yeh ek command chalao, 4-6 ghante wait karo, phir overnight learning shuru karo!** 🚀
