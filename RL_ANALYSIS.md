# Reinforcement Learning Implementation Analysis: Stones & Rivers Agent

## Executive Summary
This codebase implements a **hybrid Monte Carlo Tree Search (MCTS) with learned value and policy networks** for the Stones & Rivers board game. The system combines:
- **Value-based learning**: Ridge regression on 16-dimensional board state features
- **Policy-based learning**: Softmax-based linear policy over 10-dimensional move features  
- **MCTS backbone**: PUCT (Parallel UCT) algorithm with Zobrist hashing and transposition tables
- **Training pipeline**: Self-play game generation → feature extraction → supervised learning

---

## 1. RL Algorithm Type

### **Hybrid Value + Policy Approach**

**NOT pure actor-critic or pure policy gradient.** Instead:

1. **Value Network** (Ridge Regression)
   - Linear model: `v(s) = dot(weights, features)`
   - 16 state features → single scalar value
   - Training: MSE on terminal positions and teacher MCTS evaluations
   - Used as leaf evaluation in MCTS

2. **Policy Network** (Linear Softmax)
   - Linear model: `logits = dot(move_weights, move_features) + bias`
   - Applied softmax over available moves
   - 10 move features → action probability distribution
   - Used to set MCTS root prior (influences exploration)

3. **MCTS Engine** (PUCT)
   - UCB-based selection: `Q(a) + c_puct * prior(a) * sqrt(N) / (1 + N(a))`
   - Progressive widening: expands K = K0 + K1*sqrt(N) children per iteration
   - Values stored from root player's perspective (no negamax)
   - Evaluation: terminal detection → learned value model → fallback heuristic

**Classification**: This is a **value-augmented policy search** system, similar to AlphaGo's approach but with linear models instead of neural networks.

---

## 2. Training Pipeline & Data Flow

### **Complete Training Pipeline**

1. **Game Generation** (`scripts/arena.py`)
   - Builds C++ agent and baseline opponents
   - Spawns web server and bot clients
   - Plays multiple games with logging enabled (RS_LOG_VALUE=1)

2. **Feature Extraction** (`student_agent.cpp`)
   - On each MCTS leaf evaluation, logs board state
   - Extracts 16-dim value features and 10-dim move features
   - Outputs to `logs/value_data.jsonl` and `logs/policy_data.jsonl`

3. **Teacher Pass** (`scripts/teacher_value.py`)
   - Runs deeper MCTS (0.3s) on each logged position
   - Generates "target" values for supervised learning
   - Output: `logs/value_teacher.jsonl`

4. **Supervised Learning**
   - **Value**: Ridge regression (ml/train_value.py) → models/value_v1.txt
   - **Policy**: Softmax cross-entropy (ml/train_policy.py) → models/policy_v1.txt

5. **Model Deployment**
   - C++ code loads trained models at initialization
   - Uses them in `leaf_value()` and root prior computation

### **Feature Definitions**

**Value Features** (16 dims):
- `[0]` n_self: Stones in own score area
- `[1]` n_opp: Stones in opponent's score area
- `[2]` m_self: Reachable stones (within 1 move)
- `[3]` m_opp: Opponent's reachable stones
- `[4-5]` min_dist: Minimum distance to score (self/opp)
- `[6-7]` avg_dist: Average distance to score (self/opp)
- `[8-9]` rivers: River pieces owned (self/opp)
- `[10-11]` mobility: Adjacent empty cells (self/opp)
- `[12-13]` center_pressure: Control of center columns (self/opp)
- `[14-15]` threat: One-move scoring threat (self/opp)

**Policy Features** (10 dims per move):
- `[0]` progress_delta: Movement toward goal
- `[1]` manhattan_dist: From position to target
- `[2-4]` action_type: One-hot (push/flip/rotate)
- `[5]` unsafe_flow: Penalty if enables opponent scoring
- `[6-7]` post_move_reach: Reachability after move (self/opp)
- `[8]` landing_score: Is target in own score area?
- `[9]` to_manh: Manhattan distance of move

---

## 3. Model Architecture

### **Value Model**
```
Input: 16-dimensional state features
├─ Linear layer: w_i for each feature
└─ Output: Single scalar v(s) ∈ [-10, 10]

Training: Ridge regression
  w = (X^T X + λI)^{-1} X^T y
  λ = 1e-3
  
Evaluation:
  v(s) = sum_i(w_i * feature_i)
  clamped to [-10, 10]
```

### **Policy Model**
```
Input: 10-dimensional move features (per action)
├─ Linear transformation: logits[a] = w^T features[a] + b
├─ Softmax: π(a|s) = exp(logits[a]) / sum_k(exp(logits[k]))
└─ Output: K-way probability distribution

Training: Softmax cross-entropy
  L = -sum_a(target[a] * log(π(a)))
  SGD with lr=0.05, epochs=200, λ=1e-3
```

### **MCTS Tree Structure**
```cpp
struct MCTSNode {
    int N = 0;                    // Visit count
    double W = 0.0;               // Accumulated value (root perspective)
    double Q = 0.0;               // Q = W / N (average value)
    uint64_t hash;                // Zobrist hash
    string to_move;               // Player to move
    Board state;                  // Compact board representation
    vector<Move> moves;           // Legal moves available
    vector<double> priors;        // Policy priors (from learned model)
    vector<MCTSNode*> children;   // Child nodes
    vector<int> order_by_prior;   // Moves sorted by prior (for progressive widening)
    bool terminal;                // Terminal state?
};
```

### **PUCT Selection Formula**
```
For each action a:
  Q(a) = total_value / visit_count
  U(a) = c_puct * prior(a) * sqrt(parent_visits) / (1 + action_visits)
  PUCT_score(a) = Q(a) + U(a)

Action selection: argmax_a PUCT_score(a)
Progressive widening: Only expand top K0 + K1*sqrt(N) actions
```

---

## 4. Reward Structure & Environment

### **Terminal Rewards**
- Win (≥4 stones in own goal): **+50.0**
- Loss (opponent ≥4 stones): **-50.0**

### **Intermediate Value Estimates**
When a terminal state is reached:
1. Check if actual win/loss → return ±50
2. If learned value model available → return dot(weights, features) ∈ [-10, 10]
3. Otherwise → use heuristic `quick_state_score()` with fallback scaling

### **Heuristic Scoring** (fallback)
```cpp
quick_state_score(board, player) = 
    + 10 * stones_in_goal
    + 3 * reachable_in_one_move
    - distance_penalties
    + reachability_bonuses
```

### **Exploration Mechanism**
- **Base**: PUCT formula with c_puct = 1.2
- **Root noise** (optional): Mix in Dirichlet noise: (1-ε)*prior + ε*noise

---

## 5. Key Hyperparameters

### **MCTS Hyperparameters** (in student_agent.cpp)
| Parameter | Value | Purpose |
|-----------|-------|---------|
| c_puct | 1.2 | UCB exploration weight |
| K0 | 6 | Progressive widening base |
| K1 | 1.5 | Progressive widening coefficient |
| MAX_DEPTH | 100 | Max iterations per search |
| MAX_NODES | 20,000 | Memory safety limit |
| TT_SOFT_LIMIT | 50,000 | GC trigger |
| TT_HARD_LIMIT | 100,000 | Aggressive cleanup |

### **Learning Hyperparameters**
| Parameter | Value | Component |
|-----------|-------|-----------|
| Ridge λ | 1e-3 | Value training |
| Policy λ | 1e-3 | Policy training |
| Policy lr | 0.05 | Policy training |
| Policy epochs | 200 | Policy training |
| Value clamp | [-10, 10] | Leaf evaluation |
| Terminal reward | ±50 | Game end |

### **Runtime Tuning (via environment variables)**
```bash
RS_ROOT_NOISE=0.1         # Root exploration noise
RS_C_PUCT=1.2             # Override PUCT coefficient
RS_K0=6                   # Override widening base
RS_K1=1.5                 # Override widening growth
RS_ADVERSARIAL=1          # Select worst move (debug)
RS_LOG_VALUE=1            # Enable value logging
RS_STUDENT_MODULE_PATH=/path  # Model override
```

---

## 6. Training Logs & Current Status

### **Data Collection** (as of 2025-11-10)

| File | Rows | Status | Issue |
|------|------|--------|-------|
| value_data.jsonl | 7,804 | Generated | No final_score field |
| value_all.jsonl | 2,535 | Generated | **No targets** |
| policy_data.jsonl | 6,346 | Generated | OK |
| value_teacher.jsonl | **0** | **Failed** | **Teacher pass didn't complete** |
| policy_20251110_094053.jsonl | 9,587 | Generated | Snapshot |

### **Training Runs**
- 2025-11-10 09:40: Initial arena run (value + policy logging)
- 2025-11-10 10:05-13:55: Multiple smaller runs
- 2025-11-10 15:57-16:00: Evaluation runs only

### **Current State: BROKEN PIPELINE**
```
✓ Game Generation works
✓ Feature extraction works
✗ Teacher pass failed (value_teacher.jsonl is empty)
✗ Value training failed (no targets in data)
? Policy training untested
✗ Models missing (models/ directory empty except .gitkeep)
? MCTS uses fallback heuristics instead of learned values
```

---

## 7. Critical Issues Preventing Learning

### **CRITICAL ISSUES (Block All Learning)**

#### **Issue #1: Teacher Pass Never Completed**
- **Symptom**: `logs/value_teacher.jsonl` is 0 bytes
- **Impact**: No targets for value regression
- **Root Cause**: Subprocess isolation or MCTS crash during teacher evaluation
- **Fix**: Debug `scripts/teacher_value.py` with smaller dataset

#### **Issue #2: Training Data Format Mismatch**
- **Symptom**: `value_all.jsonl` has no "target" field
- **Impact**: `train_value.py` skips all rows
- **Root Cause**: Raw logs have `final_score_from_side`, trainer expects `target`
- **Fix**: Add normalization step before training

#### **Issue #3: Value Model Never Trained**
- **Symptom**: `models/value_v1.txt` missing
- **Impact**: Agent falls back to heuristic `quick_state_score()`
- **Consequence**: Learned value never used; no value learning loop
- **Fix**: Execute: `python ml/train_value.py --input logs/value_teacher.jsonl --out models/value_v1.txt`

---

### **HIGH-PRIORITY ISSUES (Degrade Performance)**

#### **Issue #4: Transposition Table Reuse Disabled**
- **Location**: Line 1330 in student_agent.cpp
- **Code**: `if(false && jt != tt.end())` — intentionally disabled
- **Impact**: Same positions evaluated multiple times; 2-3x slower search
- **Reason**: "Avoid double-counting bug" (unresolved)
- **Fix**: Implement proper TT handling with virtual loss or separate tree

#### **Issue #5: Potential Q-Value Sign Error**
- **Location**: Line 1450, 1476
- **Code**: `double q_value = (node->to_move == root_side) ? ch->Q : -ch->Q;`
- **Issue**: Unclear semantics—values stored in root perspective, but sign-flipped selectively
- **Impact**: Selection may be biased; sub-optimal move ordering
- **Fix**: Clarify and test sign convention

#### **Issue #6: Value Range Mismatch**
- **Terminal values**: ±50
- **Non-terminal values**: ±8 max (after heuristic scaling)
- **Impact**: 6:1 disparity; exploration biased toward terminals
- **Fix**: Scale non-terminals to comparable range

---

### **MEDIUM-PRIORITY ISSUES (Worsen Learning)**

#### **Issue #7: Policy Network Never Deployed**
- **Status**: Model loading code exists but models never trained
- **Impact**: Hand-crafted move priority dominates; learned policy unused
- **Fix**: Train and deploy policy model

#### **Issue #8: No Online / Incremental Learning**
- **Current**: Full offline ridge regression after each run
- **Problem**: Slow iteration (games → data → train → deploy cycle)
- **Better**: Online SGD or streaming ridge regression

#### **Issue #9: Reachability Heuristic Is Crude**
- **Function**: `estimate_m_reachable_in_one()`
- **Issue**: Only counts adjacent empty cells; ignores river flows
- **Impact**: Heuristic values can be very wrong
- **Used in**: Both fallback scoring and policy features

---

### **LOW-PRIORITY ISSUES (Annoyances)**

#### **Issue #10: Memory Limits Too Strict**
- **First move**: Capped at 15,000 nodes (vs 20,000 normal)
- **Time limit**: 1.5s, but may hit node limit first
- **Impact**: Early moves don't explore enough

#### **Issue #11: GC Can Lose Useful Nodes**
- **Function**: `perform_gc()` (line 1757)
- **Issue**: If `last_root_hash_` goes stale, subtrees deleted
- **Impact**: Second search in game loses prior work

#### **Issue #12: Zobrist Collisions Not Checked**
- **Risk**: Extremely low but silent bugs if occur
- **Recommendation**: Add collision detection in debug mode

---

## 8. Detailed Data Flow Diagram

```
┌──────────────────────────────────┐
│  GAME GENERATION                 │
│  scripts/arena.py                │
│  - Spawn agents (C++ and greedy) │
│  - RS_LOG_VALUE=1 enabled        │
│  - Play games with time limits   │
└──────────┬───────────────────────┘
           │
           v
┌──────────────────────────────────┐
│  FEATURE EXTRACTION              │
│  student_agent.cpp:mcts_decide() │
│  - On each leaf eval             │
│  - Extract 16 state features     │
│  - Extract 10 move features      │
│  → logs/value_data.jsonl         │
│  → logs/policy_data.jsonl        │
└──────────┬───────────────────────┘
           │
           v
┌──────────────────────────────────┐
│  TEACHER PASS ✗ FAILED           │
│  scripts/teacher_value.py        │
│  - Run 0.3s MCTS on each pos     │
│  - Generate target values        │
│  → logs/value_teacher.jsonl (0B) │
└──────────┬───────────────────────┘
           │ BLOCKED HERE
           v
┌──────────────────────────────────┐
│  VALUE TRAINING ✗ FAILED         │
│  ml/train_value.py               │
│  - Ridge regression on targets   │
│  - L2 regularization λ=1e-3      │
│  → models/value_v1.txt (missing) │
└──────────┬───────────────────────┘
           │
           v
┌──────────────────────────────────┐
│  POLICY TRAINING ⚠ INCOMPLETE    │
│  ml/train_policy.py              │
│  - Softmax cross-entropy         │
│  - 200 epochs, lr=0.05           │
│  → models/policy_v1.txt (exists?)│
└──────────┬───────────────────────┘
           │
           v
┌──────────────────────────────────┐
│  MODEL DEPLOYMENT                │
│  student_agent.cpp               │
│  - Load models/value_v1.txt      │
│  - Load models/policy_v1.txt     │
│  - Use in leaf_value() and       │
│    root prior computation        │
└──────────┬───────────────────────┘
           │
           v
┌──────────────────────────────────┐
│  MCTS SEARCH                     │
│  student_agent.cpp:mcts_decide() │
│  - PUCT with learned values      │
│  - Progressive widening          │
│  - Zobrist TT (reuse disabled)   │
└──────────────────────────────────┘
```

---

## 9. Quick Fix Checklist

### **Immediate Actions (Next 30 mins)**
- [ ] Verify `value_all.jsonl` has `final_score_from_side` field
- [ ] Run teacher pass with smaller dataset (--max 100)
- [ ] Check for errors in teacher pass logs
- [ ] Verify models/ directory writable

### **Fix Training Pipeline (Next 1 hour)**
- [ ] Fix data format for train_value.py
- [ ] Complete teacher pass: `python scripts/teacher_value.py --input logs/value_all.jsonl --output logs/value_teacher.jsonl --max 5000`
- [ ] Train value model: `python ml/train_value.py --input logs/value_teacher.jsonl --out models/value_v1.txt --l2 1e-3`
- [ ] Train policy model: `python ml/train_policy.py --input logs/policy_data.jsonl --out models/policy_v1.txt`

### **Verify Deployment (Next 30 mins)**
- [ ] Rebuild C++ agent: `make cmake-rebuild`
- [ ] Run single test game with new models
- [ ] Check logs/arena_loop.log for MCTS using learned values

### **Investigate Issues (Next 2 hours)**
- [ ] Re-enable TT reuse with proper handling
- [ ] Fix Q-value sign convention
- [ ] Scale non-terminal values correctly
- [ ] Increase node limits for better exploration

---

## 10. Implementation Summary

**Algorithm Class**: Value-augmented policy search (MCTS + learned models)
**Network Type**: Linear ridge regression (value) + softmax (policy)
**Training Method**: Supervised learning from self-play + teacher
**Search**: PUCT with progressive widening and Zobrist hashing
**Current Status**: Broken—learned models missing, pipeline incomplete
**Main Blocker**: Teacher pass failed; no supervision for value training

