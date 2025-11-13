#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include<iostream>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <set>
#include <array>
#include <chrono>
#include <algorithm>
#include <chrono> // For timing iterative deepening
#include<queue>
#include <sstream> // for ostringstream used in choose()
using namespace std;

namespace py = pybind11;


/*
=========================================================
 STUDENT AGENT FOR STONES & RIVERS GAME
---------------------------------------------------------
 The Python game engine passes the BOARD state into C++.
 Each board cell is represented as a dictionary in Python:

    {
        "owner": "circle" | "square",          // which player owns this piece
        "side": "stone" | "river",             // piece type
        "orientation": "horizontal" | "vertical"  // only relevant if side == "river"
    }

 In C++ with pybind11, this becomes:

    std::vector<std::vector<std::map<std::string, std::string>>>

 Meaning:
   - board[y][x] gives the cell at (x, y).
   - board[y][x].empty()
   - board[y][x].at("owner")
   - board[y][x].at("side") 
   - board[y][x].at("orientation") 

=========================================================
*/

// Basic enums (for future refactor/migration; not yet wired into logic)
enum Owner { OWN_NONE=0, OWN_CIRCLE=1, OWN_SQUARE=2 };
enum Side  { SIDE_EMPTY=0, SIDE_STONE=1, SIDE_RIVER=2 };
enum Ori   { ORI_NONE=0, ORI_HORIZONTAL=1, ORI_VERTICAL=2 };

// Dynamic board context (set in choose())
static int G_ROWS = 13;
static int G_COLS = 12;
static vector<int> G_SCORE_COLS; // scoring columns
// Global knob to push progress when we are stuck across root turns
static int G_PROGRESS_MODE = 0; // 0=normal, 1=need progress, 2=urgent progress
static int G_ROOT_TURN = 0; // track root turn for constructive flip gating

// Global lane planner state (set per choose() call for quick_order_score)
static int G_ACTIVE_LANE = -1;      // chosen sparse column
static int G_CAP_ROW_SELF = -1;     // row where horizontal cap river should live
static int G_PLAN_PHASE = 0;        // current phase (0=SEED_VERT, 1=SEED_CAP, 2=LADDER_PUSH, 3=CONSOLIDATE)

// Global time control for search (set per choose() call)
static chrono::high_resolution_clock::time_point G_DEADLINE;
static bool G_TIMEOUT_HIT = false;
static inline bool time_up(){
    if(G_TIMEOUT_HIT) return true;
    if(chrono::high_resolution_clock::now() >= G_DEADLINE){ G_TIMEOUT_HIT = true; return true; }
    return false;
}

static inline int top_score_row_dynamic(){ return 2; }
static inline int bottom_score_row_dynamic(){ return G_ROWS - 3; }
static inline bool col_in_score(int c){ return find(G_SCORE_COLS.begin(), G_SCORE_COLS.end(), c) != G_SCORE_COLS.end(); }
static inline bool is_own_score_cell_dyn(int r,int c,const string& owner){
    if(owner=="circle") return (r==top_score_row_dynamic()) && col_in_score(c);
    return (r==bottom_score_row_dynamic()) && col_in_score(c);
}
static inline bool is_opponent_score_cell_dyn(int r,int c,const string& owner){
    if(owner=="circle") return (r==bottom_score_row_dynamic()) && col_in_score(c);
    return (r==top_score_row_dynamic()) && col_in_score(c);
}

///generate all moves ( board, coordinate)-> return a list of all moves 
//board update input->move adn give updated board 
//eval function calculation -> input ->updated board ,side output ->value
// alpha beta pruning and minimax funtcion 

// ---- Move struct ----
struct Move {
    string action;
    vector<int> from;
    vector<int> to;
    vector<int> pushed_to;
    string orientation;
};

vector<Move> generate_all_moves(const vector<vector<map<string, string>>>& board, int row, int col);
vector<vector<map<string, string>>> update_board(const vector<vector<map<string, string>>>& board, Move move);
int evaluate_function(const vector<vector<map<string, string>>>& board,const string& owner);
Move minimax_decision(const vector<vector<map<string, string>>>& board,const string& owner,int max_depth);
int max_value(const vector<vector<map<string, string>>>& board, string owner,int max_depth,int current_depth,int alpha,int beta);
int min_value(const vector<vector<map<string, string>>>& board, string owner,int max_depth,int current_depth,int alpha,int beta);
bool is_terminal(const vector<vector<map<string, string>>>& board);
bool is_in_score_area(int row,int column,string owner);
// Helper returning true if (row,col) is in OWNER's own scoring area
static inline bool is_own_score_area(int row,int col,const string& owner){
    return is_own_score_cell_dyn(row,col,owner);
}

// Forward declarations for heuristics used in quick_order_score
int min_distance_from_scoring_area(const string& owner,int x,int y);
static inline bool is_own_score_cell_fixed(int row, int col, const string& owner);

// Simple full-position hash for repetition/cycle detection (includes side-to-move)
static inline uint64_t simple_fullpos_hash(const vector<vector<map<string,string>>>& board, const string& side_to_move){
    uint64_t h = 1469598103934665603ull; // FNV-1a basis
    auto mix = [&](uint64_t v){ h ^= v; h *= 1099511628211ull; };
    int R = (int)board.size(); if(R==0) return 0;
    int C = (int)board[0].size();
    mix((uint64_t)R); mix((uint64_t)C);
    for(int i=0;i<R;i++){
        for(int j=0;j<C;j++){
            const auto &cell = board[i][j];
            uint64_t code = 1ull; // empty marker
            if(!cell.empty()){
                // owner
                char ow = 'n';
                auto itO = cell.find("owner"); if(itO!=cell.end()){
                    ow = (itO->second=="circle")?'c':((itO->second=="square")?'s':'n');
                }
                // side
                char sd = 'e';
                auto itS = cell.find("side"); if(itS!=cell.end()){
                    sd = (itS->second=="stone")?'t':((itS->second=="river")?'r':'e');
                }
                // orientation
                char orc = 'n';
                auto itR = cell.find("orientation"); if(itR!=cell.end()){
                    orc = (itR->second=="horizontal")?'h':((itR->second=="vertical")?'v':'n');
                }
                code = ((uint64_t)ow) | ((uint64_t)sd<<8) | ((uint64_t)orc<<16);
            }
            mix(code + (uint64_t)(i*257 + j));
        }
    }
    // include side to move
    mix((side_to_move=="circle")?0xC1C1C1ull:0x515151ull);
    return h;
}

// Side-agnostic board hash (ignore side to move) to detect repetition/ping-pong
static inline uint64_t simple_board_hash_no_turn(const vector<vector<map<string,string>>>& board){
    uint64_t h = 1469598103934665603ull; // FNV-1a basis
    auto mix = [&](uint64_t v){ h ^= v; h *= 1099511628211ull; };
    int R = (int)board.size(); if(R==0) return 0;
    int C = (int)board[0].size();
    mix((uint64_t)R); mix((uint64_t)C);
    for(int i=0;i<R;i++){
        for(int j=0;j<C;j++){
            const auto &cell = board[i][j];
            uint64_t code = 1ull; // empty marker
            if(!cell.empty()){
                char ow = 'n';
                auto itO = cell.find("owner"); if(itO!=cell.end()){
                    ow = (itO->second=="circle")?'c':((itO->second=="square")?'s':'n');
                }
                char sd = 'e';
                auto itS = cell.find("side"); if(itS!=cell.end()){
                    sd = (itS->second=="stone")?'t':((itS->second=="river")?'r':'e');
                }
                char orc = 'n';
                auto itR = cell.find("orientation"); if(itR!=cell.end()){
                    orc = (itR->second=="horizontal")?'h':((itR->second=="vertical")?'v':'n');
                }
                code = ((uint64_t)ow) | ((uint64_t)sd<<8) | ((uint64_t)orc<<16);
            }
            mix(code + (uint64_t)(i*257 + j));
        }
    }
    return h;
}

// ==================== HELPER FOR WING/CENTER DETECTION ====================
static inline pair<int,int> center_band_bounds(){
    int mcols = G_COLS;
    int sc_min = 0, sc_max = mcols-1;
    if(!G_SCORE_COLS.empty()){
        sc_min = *min_element(G_SCORE_COLS.begin(), G_SCORE_COLS.end());
        sc_max = *max_element(G_SCORE_COLS.begin(), G_SCORE_COLS.end());
    } else {
        int w = (mcols<=12)?4:((mcols<=14)?5:6);
        int s = max(0, mcols/2 - w/2); sc_min = s; sc_max = min(mcols-1, s+w-1);
    }
    int centerL = max(0, sc_min - 1);
    int centerR = min(mcols-1, sc_max + 1);
    return {centerL, centerR};
}

// Lightweight move-ordering heuristic: higher is better for the mover
static int quick_order_score(const vector<vector<map<string,string>>>& board, const Move& mv){
    if(mv.from.size()<2) return 0;
    int fr = mv.from[0], fc = mv.from[1];
    if(fr<0 || fr>=(int)board.size() || fc<0 || fc>=(int)board[0].size()) return 0;
    if(board[fr][fc].empty()) return 0;
    string mover = board[fr][fc].at("owner");
    string side  = board[fr][fc].at("side");

    auto md = [&](int r,int c){ return min_distance_from_scoring_area(mover, r, c); };
    
    // Wing-first helpers
    auto [centerL,centerR] = center_band_bounds();
    auto is_wing = [&](int cc){ return cc<centerL || cc>centerR; };

    int score = 0;
    int prog_weight = (G_PROGRESS_MODE==0?16:(G_PROGRESS_MODE==1?42:70));
    int no_prog_pen = (G_PROGRESS_MODE==0?10:(G_PROGRESS_MODE==1?56:96));
    auto col_blockers = [&](const string& who, int col){
        if(col<0 || col>=(int)board[0].size()) return 6; // out of bounds treated as heavy
        int start = (who=="circle")? bottom_score_row_dynamic() : top_score_row_dynamic();
        int goal  = (who=="circle")? top_score_row_dynamic()    : bottom_score_row_dynamic();
        int step  = (who=="circle")? -1 : +1;
        int cnt=0;
        for(int r=start; r!=goal+step; r+=step){
            if(r<0 || r>=(int)board.size()) continue;
            if(board[r][col].empty()) continue;
            const auto &cell = board[r][col];
            if(cell.at("owner")!=who && cell.at("side")=="stone") cnt++;
        }
        return cnt;
    };
    if(mv.action=="move"){
        if(mv.to.size()<2) return 0;
        int tr = mv.to[0], tc = mv.to[1];
        int before = md(fr, fc);
        int after  = md(tr, tc);

        // === LANE BLUEPRINT BONUSES FOR MOVES ===
        // Phase 0 (SEED_VERT): favor moving stones TOWARD the active lane
        if(G_PLAN_PHASE == 0 && tc==G_ACTIVE_LANE){
            score += 120;
        }
        // Phase 1 (SEED_CAP): favor positioning stone below cap, ready to push
        if(G_PLAN_PHASE == 1){
            int row_below_cap = G_CAP_ROW_SELF + ((mover=="circle")?-1:+1);
            if(tr==row_below_cap && tc==G_ACTIVE_LANE){
                score += 300;
            }
        }

        score += (before - after) * prog_weight; // stronger progress bonus when stuck
        if(is_own_score_cell_fixed(tr, tc, mover)) score += 2200; // instant score landing slightly increased
        // prefer longer moves slightly
        score += (abs(tr-fr)+abs(tc-fc) > 1 ? 3 : 0);
        // adjacency to friendly river boosts
        auto in = [&](int r,int c){ return r>=0 && r<(int)board.size() && c>=0 && c<(int)board[0].size(); };
        const pair<int,int> dirs[4]={{1,0},{-1,0},{0,1},{0,-1}};
        for(auto [dx,dy]: dirs){
            int rx=tr+dx, ry=tc+dy; if(!in(rx,ry) || board[rx][ry].empty()) continue;
            if(board[rx][ry].at("side")=="river" && board[rx][ry].at("owner")==mover){
                score += (board[rx][ry].at("orientation")=="vertical"? 14:8);
            }
        }
        // strongly penalize sliding moves within own scoring area
        if(board[fr][fc].at("side")=="stone" && is_own_score_cell_fixed(fr,fc,mover) && is_own_score_cell_fixed(tr,tc,mover)){
            score -= 1500;
        }
        
        // Wing-first nudges
        bool toWing = is_wing(tc);
        bool fromCenter = (fc>=centerL && fc<=centerR);
        bool toCenter   = (tc>=centerL && tc<=centerR);
        bool strongProgress = ((before - after) >= 2) || is_own_score_cell_fixed(tr, tc, mover);

        if(toWing)           score += 120;          // leak outward
        if(toCenter && !strongProgress) score -= 260;
        if(fromCenter && toCenter && after >= before) score -= 420;
        // penalize lateral moves that do not reduce distance (avoid cosmetic shuffles)
        if(after >= before) score -= no_prog_pen;
        // prefer lanes with fewer opponent blockers to reach goal
        int oppb_to = col_blockers(mover, tc);
        int oppb_from = col_blockers(mover, fc);
        score -= oppb_to * 35;         // stronger column sparsity preference
        if(oppb_to==0) score += 120;   // very attractive totally free lane
        // reward shifting toward sparser column
        if(oppb_to < oppb_from) score += (oppb_from - oppb_to) * 60;
        // if moving closer to best scoring column by blockers, reward
        int best_sc = tc;
        int best_b = 99;
        if(!G_SCORE_COLS.empty()){
            for(int sc: G_SCORE_COLS){ int b = col_blockers(mover, sc); if(b < best_b){ best_b=b; best_sc=sc; } }
        }
        int dist_from = abs(fc - best_sc), dist_to = abs(tc - best_sc);
        if(dist_to < dist_from) score += 120;
        return score;
    }
    if(mv.action=="push"){
        if(mv.to.size()<2 || mv.pushed_to.size()<2) return 0;
        int target_r = mv.to[0], target_c = mv.to[1]; // pushed stone current
        int pr = mv.pushed_to[0], pc = mv.pushed_to[1];

        // === LANE BLUEPRINT BONUSES FOR PUSHES ===
        // Phase 2 (LADDER_PUSH): strongly favor pushing along active lane toward goal
        // FIXED: allow pusher from outside lane as long as push travels in lane
        if(G_PLAN_PHASE == 2){
            int dir = (mover=="circle") ? -1 : +1;
            bool alongLane = (target_c==G_ACTIVE_LANE && pc==G_ACTIVE_LANE); // pushed stone travels in lane
            bool forward = ((pr-target_r)==dir);
            if(alongLane && forward){
                score += 900; // huge bonus for ladder pushes
            }
        }

        // pushing opponent away from their goal is good
        string pushed_owner = board[target_r][target_c].at("owner");
        auto md_pushed = [&](int r,int c){ return min_distance_from_scoring_area(pushed_owner, r, c); };
        int before = md(fr, fc);
        int after  = md(mv.to[0], mv.to[1]);
        score += (before - after) * (prog_weight/5 + 10);
        int before_p = md_pushed(target_r, target_c);
        int after_p  = md_pushed(pr, pc);
        score += (after_p - before_p) * 8; // hurt opponent more
        if(after >= before) score -= (no_prog_pen/2);
        // prefer pushes that clear our sparser lanes
        int oppb2 = col_blockers(mover, pc);
        score -= oppb2 * 20;
        int oppb_target_col = col_blockers(mover, target_c);
        if(oppb2 < oppb_target_col) score += (oppb_target_col - oppb2) * 50;
        return score;
    }
    if(mv.action=="flip"){
        if(side=="stone"){
            // === LANE BLUEPRINT BONUSES FOR FLIPS ===
            // Phase 0 (SEED_VERT): Strongly favor vertical flips in active lane
            if(G_PLAN_PHASE == 0 && fr>=0 && fc==G_ACTIVE_LANE && mv.orientation=="vertical"){
                score += 800; // huge push to create ladder rungs
                int occ_col=0; for(int rr=0; rr<(int)board.size(); ++rr){ if(!board[rr][fc].empty()) occ_col++; }
                score += 10 * max(0, 4 - occ_col); // prefer emptier lanes
            }
            // discourage horizontal flips in active lane during SEED_VERT
            if(G_PLAN_PHASE == 0 && fr>=0 && fc==G_ACTIVE_LANE && mv.orientation=="horizontal"){
                score -= 250;
            }

            // Phase 1 (SEED_CAP): Strongly favor horizontal cap at cap_row_self in active lane
            if(G_PLAN_PHASE == 1 && fr==G_CAP_ROW_SELF && fc==G_ACTIVE_LANE && mv.orientation=="horizontal"){
                score += 1200; // finish the cap strongly
            }

            // Strongly encourage creating rivers, especially in exterior (wing) columns and open space
            // Base flip bonus and progress mode boost
            score += 120;
            if(G_PROGRESS_MODE>=1) score += (G_PROGRESS_MODE==2 ? 120 : 60);

            // Determine center band (scoring columns +/- 1) to detect wings
            int mcols = (int)board[0].size();
            int sc_min = 0, sc_max = mcols-1;
            if(!G_SCORE_COLS.empty()){
                sc_min = *min_element(G_SCORE_COLS.begin(), G_SCORE_COLS.end());
                sc_max = *max_element(G_SCORE_COLS.begin(), G_SCORE_COLS.end());
            } else {
                int w = (mcols<=12)?4:((mcols<=14)?5:6);
                int s = max(0, mcols/2 - w/2); sc_min = s; sc_max = min(mcols-1, s+w-1);
            }
            int centerL = max(0, sc_min - 1);
            int centerR = min(mcols-1, sc_max + 1);
            bool in_wing = (fc < centerL || fc > centerR);
            bool deep_wing = (fc < max(0, centerL-2) || fc > min(mcols-1, centerR+2));

            // Orientation preference toward goal movement (vertical more valuable)
            score += (mv.orientation=="vertical" ? 160 : 80);

            // Column sparsity: prefer flipping in columns with fewer opponent blockers and lower overall occupancy
            auto col_blockers = [&](const string& who, int col){
                if(col<0 || col>=mcols) return 6; int start=(who=="circle")? bottom_score_row_dynamic(): top_score_row_dynamic();
                int goal=(who=="circle")? top_score_row_dynamic(): bottom_score_row_dynamic(); int step=(who=="circle")?-1:+1; int cnt=0;
                for(int r=start; r!=goal+step; r+=step){ if(r<0||r>=(int)board.size()) continue; if(board[r][col].empty()) continue; auto &cell=board[r][col]; if(cell.at("owner")!=who && cell.at("side")=="stone") cnt++; }
                return cnt; };
            int blk = col_blockers(mover, fc);
            score -= blk * 16; if(blk==0) score += 120;
            int occ_col=0; for(int rr=0; rr<(int)board.size(); ++rr){ if(!board[rr][fc].empty()) occ_col++; }
            score -= occ_col * 10; if(occ_col<=1) score += 100;

            // Wing bonus: seed exterior networks, extra for deep wings
            score += in_wing ? 180 : -40; // mild penalty for staying in center band
            if(deep_wing) score += 120;

            // Explicit center penalty if center already has many of our rivers
            int center_rivers=0, wing_rivers=0;
            for(int r=0;r<(int)board.size();++r){
                for(int c=0;c<mcols;++c){ if(board[r][c].empty()) continue; if(board[r][c].at("side")=="river" && board[r][c].at("owner")==mover){
                    if(c>=centerL && c<=centerR) center_rivers++; else wing_rivers++; }}
            }
            if(fc>=centerL && fc<=centerR && center_rivers > wing_rivers) score -= min(300, 30*(center_rivers-wing_rivers));

            // General center flip penalty (but don't ban - allow search to prove value)
            // FIXED: waive penalty if flip could lead to immediate finish
            bool could_finish_next = false;
            {
                int md_after = min_distance_from_scoring_area(mover, fr, fc);
                if(is_own_score_cell_fixed(fr, fc, mover) || md_after <= 1) could_finish_next = true;
            }
            if(fc>=centerL && fc<=centerR){
                if(!could_finish_next) score -= 220; // skip penalty if flip leads to finish
            }

            // Local crowding: discourage flipping in dense neighborhoods
            int occ=0, empt=0; for(int dr=-1; dr<=1; ++dr) for(int dc=-1; dc<=1; ++dc){ if(!dr && !dc) continue; int r=fr+dr, c=fc+dc; if(r<0||r>=(int)board.size()||c<0||c>=mcols) continue; if(board[r][c].empty()) empt++; else occ++; }
            score += (empt * 24) - (occ * 44);

            // Path potential in the chosen orientation: count contiguous empties/rivers
            auto line_len = [&](int dr,int dc){ int r=fr+dr,c=fc+dc,len=0; while(r>=0&&r<(int)board.size()&&c>=0&&c<mcols){ if(board[r][c].empty() || board[r][c].at("side")=="river") { len++; r+=dr; c+=dc; } else break; } return len; };
            if(mv.orientation=="vertical"){
                int L = (line_len(1,0)+line_len(-1,0)); score += L*26; if(in_wing) score += L*8;
            } else {
                int L = (line_len(0,1)+line_len(0,-1)); score += L*16; if(in_wing) score += L*5;
            }
            // Distance to board edge: prefer flipping nearer outer border
            int d_edge = min(fc, mcols-1-fc);
            score += max(0, 40 - d_edge*10);
            // Slight preference to be away from our stones (to grow outside the cluster)
            int adj_friend=0; const pair<int,int> dirs4[4]={{1,0},{-1,0},{0,1},{0,-1}}; for(auto [dx,dy]: dirs4){ int rx=fr+dx, ry=fc+dy; if(rx<0||rx>=(int)board.size()||ry<0||ry>=mcols) continue; if(!board[rx][ry].empty() && board[rx][ry].at("owner")==mover && board[rx][ry].at("side")=="stone") adj_friend++; }
            score -= adj_friend * 8;
        } else {
            // Generally discourage flipping rivers back to stones (unless search finds it useful)
            score -= 10;
        }
        return score;
    }
    if(mv.action=="rotate"){
        // mild preference
        return 1;
    }
    return 0;
}

// Helper: check if a cell is in the owner's OWN scoring area (dynamic)
static inline bool is_own_score_cell_fixed(int row, int col, const string& owner){
    return is_own_score_cell_dyn(row, col, owner);
}

// Helper functions for path blocking
vector<pair<int,int>> trace_winning_path(const vector<vector<map<string, string>>>& board, const Move& win_move, const string& opponent);
bool simulate_move_path(const vector<vector<map<string, string>>>& board, int current_row, int current_col, int target_row, int target_col, vector<pair<int,int>>& path_stack, set<pair<int,int>>& visited, int dx, int dy);
Move find_path_blocking_moves(const vector<vector<map<string, string>>>& board, const vector<pair<int,int>>& path, const string& owner, const map<pair<int,int>,vector<Move>>& moves);

//this function determines teh minimum distnace from the score area
int min_distance_from_scoring_area(const string& owner,int x,int y){
    int target_row = (owner=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
    int min_dist = 1000000;
    if(G_SCORE_COLS.empty()){
        // Fallback: center band width by board size (4/5/6)
        int m = G_COLS;
        int w = (m<=12)?4:((m<=14)?5:6);
        int s = max(0, m/2 - w/2);
        int e = min(m-1, s+w-1);
        for(int c=s; c<=e; ++c){
            int dist = abs(x - target_row) + abs(y - c);
            min_dist = min(min_dist, dist);
        }
    } else {
        for(int c: G_SCORE_COLS){
            int dist = abs(x - target_row) + abs(y - c);
            min_dist = min(min_dist, dist);
        }
    }
    if(min_dist>=1000000) min_dist=100;
    return min_dist;
}

// Simple position hash for cycle detection: ignore orientation and piece type,
// only encode occupied cells and their owners. This makes repetition detection
// robust to cosmetic flips/rotations.
static inline uint64_t simple_position_hash(const vector<vector<map<string,string>>>& board){
    uint64_t h = 1469598103934665603ull; // FNV-1a basis
    auto mix = [&](uint64_t v){ h ^= v; h *= 1099511628211ull; };
    int R = (int)board.size(); if(R==0) return 0;
    int C = (int)board[0].size();
    mix((uint64_t)R); mix((uint64_t)C);
    for(int i=0;i<R;i++){
        for(int j=0;j<C;j++){
            if(board[i][j].empty()) continue;
            // only owner + position, ignore side/orientation
            char ow = 'n';
            auto itO = board[i][j].find("owner");
            if(itO!=board[i][j].end()) ow = (itO->second=="circle")?'c':((itO->second=="square")?'s':'n');
            uint64_t code = ((uint64_t)ow) | 0x9e3779b97f4a7c15ull;
            mix(code + (uint64_t)(i*257 + j));
        }
    }
    return h;
}

//determines the path through the next river to get the shortest distanve it ca approach to the final score area 
int find_next_river_distance(const std::vector<std::vector<std::map<std::string, std::string>>>& board,const string& owner,int x,int y,const map<pair<int,int>,vector<Move>>& moves){
    vector<Move> current_moves;
    auto it = moves.find(make_pair(x, y));
    if (it != moves.end()) {
        current_moves = it->second;
    }
    int min_dist = 100;
    pair<int,int> min_coords = {-1,-1};
    //checks for every move from that coordinates , and then calculateds the minimum  distance and gets the final minimum of it
    for(const auto& move: current_moves){
        if(move.action == "rotate")continue;
        if(move.action == "flip" && board[x][y].at("side") == "stone")continue;
        int nx,ny,dist;
        if(move.action == "push"){
            if(board[move.to[0]][move.to[1]].at("owner")==owner){
                nx = move.pushed_to[0];
                ny = move.pushed_to[1];
                dist = min_distance_from_scoring_area(owner, nx, ny);
            }
            else{
                continue;
            }
            dist= min(dist,min_distance_from_scoring_area(owner,move.to[0],move.to[1]));
        }
       
        if(move.action == "move"){
            nx = move.to[0];
            ny = move.to[1];
            dist = min_distance_from_scoring_area(owner, nx, ny);
        }
        if(dist < min_dist){
            min_dist = dist;
            min_coords = {nx, ny};
        }
    }
    return min_dist;
}

//picks the move that brings it the closest to the goal area 
static pair<int,int> pick_spearhead_stone(const vector<vector<map<string,string>>>& board,const string& owner){
    int R=board.size(), C=board[0].size();
    int best=100, bx=-1, by=-1;
    for(int i=0;i<R;i++) for(int j=0;j<C;j++){
        if(board[i][j].empty()) continue;
        if(board[i][j].at("owner")!=owner || board[i][j].at("side")!="stone") continue;
        int md = min_distance_from_scoring_area(owner,i,j);
        if (md < best) { best=md; bx=i; by=j; }
    }
    return {bx,by};
}
//function use din evaluation funcion where we see the 
//manhaatan distance each stone or river from its scoring area  
//and if it is less than a  certain amount then decalring it as threat (self and opp)
vector<int> manhattan_distance(const vector<vector<map<string,string>>>& board,const string& owner,const map<pair<int,int>,vector<Move>>& moves){
    const int n = board.size(), m = board[0].size();
    const int circle_target = top_score_row_dynamic(), square_target = bottom_score_row_dynamic();

    auto goal_row = [&](const string& who){ return (who=="circle") ? circle_target : square_target;};
    
    //calculating manhattan distance
    auto MD = [&](const string& who,int i,int j){
        int g = goal_row(who),best = 1000000;
        if(G_SCORE_COLS.empty()){
            int m = G_COLS; int w=(m<=12)?4:((m<=14)?5:6); int s=max(0,m/2-w/2); int e=min(m-1,s+w-1);
            for(int sc=s; sc<=e; ++sc) best = min(best,abs(i-g) + abs(j-sc));
        }else{
            for(int sc: G_SCORE_COLS) best = min(best,abs(i-g) + abs(j-sc));
        }
        return best>=1000000?100:best;
    };
    
    //geeting the nearest distnce by a single movement through river
    auto ONE_MOVE_RIVER = [&](const string& who,int i,int j){
        int d = find_next_river_distance(board,who,i,j,moves);
        return (d >= 100 ? 100 : d);
    };

    int best_self = 100,best_opp = 100;
    int self_threats = 0,opp_threats = 0;

    //picking each block and based on owner calculating minimum of manahaatan and river distance 
    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            if (board[i][j].empty()) continue;
            string who  = board[i][j].at("owner");
            string side = board[i][j].at("side");

            int md = MD(who, i, j);
            int r = ONE_MOVE_RIVER(who, i, j);
            
            int cost;
            //approximations via rive rwith upper bound of 100
            if (side == "stone"){
                int via_river = (r>=100) ? 100 : (1 + r);     
                cost = min(md, via_river);
            }else{ 
                //calculating for river after flipping since we would need to flip it in one move also thsu adding 1 
                int walk_after_flip  = 1 + md;          
                int river_after_flip = (r>=100) ? 100 : (2 + r); 
                cost = min(walk_after_flip, river_after_flip);
            }
            //if distnace less than 3 then threat for me and opponent respectively 
            if(who == owner){
                best_self = min(best_self, cost);
                if(cost <= 3) ++self_threats;
            }else{
                best_opp = min(best_opp, cost);
                if(cost <= 3) ++opp_threats;
            }
        }
    }
    return {best_self, best_opp, self_threats, opp_threats};
}


//check for number of stones in score area 
int count_in_score_area(const vector<vector<map<string, string>>>& board,string owner){
    int r = (owner=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
    int cnt = 0;
    if(G_SCORE_COLS.empty()){
        for(int c: {4,5,6,7}){
            if(r>=0 && r<(int)board.size() && c>=0 && c<(int)board[0].size()){
                if(!board[r][c].empty() && board[r][c].at("side")=="stone" && board[r][c].at("owner")==owner) cnt++;
            }
        }
    } else {
        for(int c: G_SCORE_COLS){
            if(r>=0 && r<(int)board.size() && c>=0 && c<(int)board[0].size()){
                if(!board[r][c].empty() && board[r][c].at("side")=="stone" && board[r][c].at("owner")==owner) cnt++;
            }
        }
    }
    return cnt;
}

//for checking if the stone was going to enter the scorign area of opposition  
bool is_in_score_area(int row,int col,string owner){
    // Returns true if (row,col) is in the OPPONENT's scoring area for the given owner (legacy name kept)
    return is_opponent_score_cell_dyn(row, col, owner);
}


//Explore river paths for MOVE actions
//here is where using recursive movements we tarverse from river to river and get to all the possible move for that particualr stone or river
void explore_river_move(const vector<vector<map<string, string>>>& board,int x, int y, vector<Move>& moves, int rows, int cols,int from_row, int from_col, set<pair<int,int>>& visited, int depth){
    // Depth and safety caps to avoid blow-ups
    if(depth > 30) return;
    if(x < 0 || x >= rows || y < 0 || y >= cols) return;
    if(from_row < 0 || from_row >= rows || from_col < 0 || from_col >= cols) return;
    if (visited.count({x, y})) return;
    visited.insert({x, y});
    if(visited.size() > 50) return;

    // river for new direction movement
    if (board[x][y].at("side") != "river") return;
    string orientation = board[x][y].at("orientation");

    vector<pair<int,int>> directions;
    if (orientation == "horizontal") directions = {{0,1},{0,-1}};
    else directions = {{1,0},{-1,0}};

    string owner = board[from_row][from_col].at("owner");

    for(auto [dx, dy] : directions){
        int nx = x + dx, ny = y + dy;
        int steps = 0;
        while (nx >= 0 && nx < rows && ny >= 0 && ny < cols && steps < 50){
            if (nx == from_row && ny == from_col) { nx += dx; ny += dy; continue;}
            // Do not pass THROUGH opponent scoring area: stop scanning when encountered
            if (is_in_score_area(nx, ny, owner)){
                break;
            }

            if (board[nx][ny].empty()){
                moves.push_back({"move", {from_row, from_col}, {nx, ny}, {}, ""});
            } 
            else if (board[nx][ny].at("side") == "river"){
                //river found now check for new contiuos movement direction and thus recursive call
                explore_river_move(board, nx, ny, moves, rows, cols, from_row, from_col, visited, depth+1);
                break;
            } 
            else break;
            nx += dx; ny += dy; ++steps;
        }
    }
}

//Explore river paths for PUSH actions
//here is where using recursive movements we tarverse from river to river and get to all the possible move for that particualr stone or river
void explore_river_push(const vector<vector<map<string, string>>>& board,int push_row, int push_col, vector<Move>& moves, int rows, int cols,int from_row, int from_col,
                         set<pair<int,int>>& visited,int or_push_row,int or_push_col, int or_from_row, int or_from_col, int depth){

    // Depth and safety caps
    if(depth > 30) return;
    if(push_row < 0 || push_row >= rows || push_col < 0 || push_col >= cols) return;
    if(from_row < 0 || from_row >= rows || from_col < 0 || from_col >= cols) return;
    if(or_push_row < 0 || or_push_row >= rows || or_push_col < 0 || or_push_col >= cols) return;
    if(or_from_row < 0 || or_from_row >= rows || or_from_col < 0 || or_from_col >= cols) return;
    if (visited.count({push_row, push_col})) return;
    visited.insert({push_row, push_col});
    if(visited.size() > 50) return;

    //getting direction
    if (board[from_row][from_col].at("side") != "river") return;
    string orientation = board[from_row][from_col].at("orientation");

    vector<pair<int,int>> directions;
    if (orientation == "horizontal") directions = {{0,1},{0,-1}};
    else directions = {{1,0},{-1,0}};

    
    string owner_to_be_pushed = board[or_push_row][or_push_col].at("owner");
    string pusher_owner = board[or_from_row][or_from_col].at("owner");
    // Do not even explore a push chain that would start by moving our own scored stone out
    if(owner_to_be_pushed == pusher_owner && is_own_score_cell_fixed(or_push_row, or_push_col, owner_to_be_pushed)) return;

    for(auto [dx, dy] : directions){
        int nx = push_row + dx;
        int ny = push_col + dy;
        int steps = 0;
        while (nx >= 0 && nx < rows && ny >= 0 && ny < cols && steps < 50) {
            if(nx == from_row && ny == from_col){ 
                nx += dx; 
                ny += dy; 
                continue; 
            }

            // Do not pass THROUGH the pushed piece's opponent scoring area
            if (is_in_score_area(nx, ny, owner_to_be_pushed)){
                break;
            }

            if(board[nx][ny].empty()){
                // empty thus a possible move position
                moves.push_back({"push",{or_from_row, or_from_col},{or_push_row, or_push_col},{nx, ny},""});
            } 
            else if(board[nx][ny].at("side") == "river"){
                //got river in path now need to tarce path accoirding to that
                explore_river_push(board, nx, ny, moves, rows, cols, nx, ny, visited,or_push_row,or_push_col, or_from_row, or_from_col, depth+1);
                break;
            }
            //stone found so break stop
            else break;

            nx += dx; 
            ny += dy;
            ++steps;
        }
    }
}

// ==================== CONSTRUCTIVE FLIP GATING ====================

static inline int contiguous_run_len(const vector<vector<map<string,string>>>& B,
                                     int r,int c,const string& owner,const string& ori){
    auto inside=[&](int x,int y){ return x>=0 && x<(int)B.size() && y>=0 && y<(int)B[0].size(); };
    int len=1; // including (r,c)
    if(ori=="vertical"){
        for(int x=r-1;x>=0;--x){ if(!inside(x,c))break;
            if(!B[x][c].empty() && B[x][c].at("owner")==owner && B[x][c].at("side")=="river" && B[x][c].at("orientation")=="vertical") len++;
            else if(B[x][c].empty()) len++; else break;
        }
        for(int x=r+1;x<(int)B.size();++x){ if(!inside(x,c))break;
            if(!B[x][c].empty() && B[x][c].at("owner")==owner && B[x][c].at("side")=="river" && B[x][c].at("orientation")=="vertical") len++;
            else if(B[x][c].empty()) len++; else break;
        }
    } else { // horizontal
        for(int y=c-1;y>=0;--y){ if(!inside(r,y))break;
            if(!B[r][y].empty() && B[r][y].at("owner")==owner && B[r][y].at("side")=="river" && B[r][y].at("orientation")=="horizontal") len++;
            else if(B[r][y].empty()) len++; else break;
        }
        for(int y=c+1;y<(int)B[0].size();++y){ if(!inside(r,y))break;
            if(!B[r][y].empty() && B[r][y].at("owner")==owner && B[r][y].at("side")=="river" && B[r][y].at("orientation")=="horizontal") len++;
            else if(B[r][y].empty()) len++; else break;
        }
    }
    return len;
}

// Quick check: does a flip reduce min distance for ANY of our nearest stones by >=2?
static inline bool flip_relieves_distance(const vector<vector<map<string,string>>>& B,
                                          int r,int c,const string& owner,const string& ori){
    // Look at stones within manhattan 2 of the flip cell
    int best_before = 1e9, best_after = 1e9;
    for(int i=max(0,r-2); i<=min((int)B.size()-1, r+2); ++i){
        for(int j=max(0,c-2); j<=min((int)B[0].size()-1, c+2); ++j){
            if(B[i][j].empty()) continue;
            if(B[i][j].at("owner")!=owner || B[i][j].at("side")!="stone") continue;
            int md = min_distance_from_scoring_area(owner,i,j);
            best_before = min(best_before, md);
            // pretend we flip (r,c) to river with ori and see if straight-line stepping opens nearer empties
            int gain = 0;
            if(ori=="vertical"){
                // stepping toward goal row
                int dir = (owner=="circle") ? -1 : +1;
                int x=i+dir, y=j; int steps=0;
                while(x>=0 && x<(int)B.size() && steps<3){
                    if((x==r && y==c) || B[x][y].empty() || ( !B[x][y].empty() && B[x][y].at("side")=="river")) { gain++; x+=dir; steps++; }
                    else break;
                }
            } else {
                // horizontal side-pull
                for(int s=-1; s<=1; s+=2){
                    int x=i, y=j+s; int steps=0;
                    while(y>=0 && y<(int)B[0].size() && steps<3){
                        if((x==r && y==c) || B[x][y].empty() || (!B[x][y].empty() && B[x][y].at("side")=="river")) { gain++; y+=s; steps++; }
                        else break;
                    }
                }
            }
            best_after = min(best_after, max(0, md - gain));
        }
    }
    return (best_before - best_after) >= 2;
}

// Only allow flips that actually help the plan / network
static inline bool flip_is_constructive(const vector<vector<map<string,string>>>& B,
                                        int r,int c,const string& owner,const string& ori){
    auto [centerL,centerR] = center_band_bounds();
    bool in_center = (c>=centerL && c<=centerR);

    // Plan-critical: vertical rung in active lane during SEED_VERT, or horizontal cap during SEED_CAP
    if(c==G_ACTIVE_LANE){
        if(G_PLAN_PHASE==0 && ori=="vertical") return true;
        if(G_PLAN_PHASE==1 && r==G_CAP_ROW_SELF && ori=="horizontal") return true;
    }

    // Network growth: extend contiguous run meaningfully
    int run = contiguous_run_len(B, r, c, owner, ori);
    bool extends_network = (ori=="vertical" ? run>=3 : run>=3);

    // Progress relief: flipping here reduces distance for nearby friendly stones
    bool relieves = flip_relieves_distance(B, r, c, owner, ori);

    // Early game is stricter: before turn ~8, only allow plan-critical OR network extend+relief
    if(G_ROOT_TURN <= 8){
        if(extends_network && relieves) return true;
        // center flips early? only if both:
        if(in_center) return false;
        return (extends_network || relieves);
    }

    // Mid/late game: allow if any constructive criterion holds
    if(extends_network || relieves) return true;

    // Otherwise, not constructive
    return false;
}

//this is the main part where i initially generate all the possible move from the state of the board 
vector<Move>generate_all_moves(const vector<vector<map<string, string>>>& board,int row,int col){

    vector<Move>moves;
    int rows = board.size();
    int cols = board[0].size();

    // Initial check: if cell is empty, return no moves
    if(board[row][col].empty())return moves;

    string owner = board[row][col].at("owner");
    string side = board[row][col].at("side");
    string orientation = "";
    if(side == "river")orientation = board[row][col].at("orientation");

    // Directions: up, down, left, right
    vector<pair<int,int>>dirs={{1,0},{-1,0},{0,1},{0,-1}};

    set<pair<int,int>>visited;
    // --- MOVE ---
    for(auto [dx,dy] : dirs){
        int nx = row + dx, ny = col + dy;
        if (nx < 0 || nx >= rows || ny < 0 || ny >= cols) continue;
        //checking if it is going in opponents score area
        if(is_in_score_area(nx,ny,owner)){
            continue;
        }
        // Prevent leaving own scoring area with a stone unless staying in scoring area
        bool currently_in_score = is_own_score_cell_fixed(row, col, owner);
        bool target_in_score    = is_own_score_cell_fixed(nx, ny, owner);
        // keep scored stones fixed (do not move out)
        if(currently_in_score && !target_in_score && side == "stone"){
            continue;
        }
        // avoid sliding inside own scoring area (wasteful)
        if(currently_in_score && target_in_score && side == "stone"){
            continue;
        }
        if(board[nx][ny].empty()){
            moves.push_back({"move", {row,col}, {nx,ny}, {}, ""});
        }
          else if(!board[nx][ny].empty()&& board[nx][ny].at("side")== "river"){
            //exploring moves via rivers
              explore_river_move(board, nx, ny, moves, rows, cols, row, col, visited, 0);
        }
    }
    

    // --- PUSH ---
    for(auto [dx,dy] : dirs){
        int nx = row + dx;
        int ny = col + dy;
        int nx2 = row + 2*dx;
        int ny2 = col + 2*dy;
        if(nx < 0 || nx >= rows || ny < 0 || ny >= cols) continue;
        // scoring-area barrier checks handled relative to pushed piece below
        if(board[nx][ny].empty()) continue;
        if(board[nx][ny].at("side")!="stone") continue;

        if(!board[nx][ny].empty() && board[nx][ny].at("side")=="stone"){  
            string owner_to_be_pushed = board[nx][ny].at("owner");

            // Skip invalid destination
            if(nx2 < 0 || nx2 >= rows || ny2 < 0 || ny2 >= cols)continue;
            // Do NOT push a piece through or into its opponent scoring area
            if(is_in_score_area(nx2, ny2, owner_to_be_pushed))continue;
            if(!board[nx2][ny2].empty())continue;

            //  stone push
            if(board[row][col].at("side") == "stone"){
                moves.push_back({"push", {row, col}, {nx, ny}, {nx2, ny2}, ""});
            }
            // Prevent pushing OUR scored stones out of scoring area
            if(owner_to_be_pushed == owner && is_own_score_cell_fixed(nx, ny, owner_to_be_pushed)){
                // skip pushes that would move our already scored stone
            }
            // River pushing a stone
            else if(board[row][col].at("side") == "river"){
                set<pair<int,int>> visited_push;
                // Also guard river-push chains from starting if we'd push our own scored stone
                if(!(owner_to_be_pushed == owner && is_own_score_cell_fixed(nx, ny, owner_to_be_pushed))){
                    explore_river_push(board,nx,ny,moves,rows,cols,row,col,visited_push,nx,ny,row,col, 0);
                }
            }
        }
    }


    // --- FLIP ---
    //stone to river and river to stone
    if(side == "stone"){
        // Smart exception: if this stone is already in our scoring row, only allow a
        // horizontal flip when it would directly help push another friendly stone
        // into a scoring column on this row ("horizontal cap").
        auto cap_would_help = [&]()->bool{
            if(!is_own_score_cell_fixed(row,col,owner)) return false;
            int target_row = (owner=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
            if(row != target_row) return false; // cap should sit on the scoring row
            auto inb = [&](int r,int c){ return r>=0 && r<rows && c>=0 && c<cols; };
            auto is_sc_col = [&](int c){ if(!G_SCORE_COLS.empty()) return find(G_SCORE_COLS.begin(),G_SCORE_COLS.end(),c)!=G_SCORE_COLS.end();
                                         int mcols = cols; int w=(mcols<=12)?4:((mcols<=14)?5:6); int s=max(0,mcols/2 - w/2); int e=min(mcols-1,s+w-1); return c>=s && c<=e; };
            // Try left and right neighbors as potential stones to push
            for(int dc_dir : {-1, +1}){
                int sx = row, sy = col + dc_dir; // neighbor position on the scoring row
                if(!inb(sx,sy) || board[sx][sy].empty()) continue;
                if(board[sx][sy].at("owner")!=owner || board[sx][sy].at("side")!="stone") continue;
                // scan along row in both horizontal directions from the stone
                for(int dir : {-1, +1}){
                    int cy = sy + dir; bool blocked=false; int steps=0; const int STEP_CAP=8;
                    while(inb(row,cy) && steps<STEP_CAP){
                        if(!board[row][cy].empty() && !(row==row && cy==col)) { blocked=true; break; }
                        if(is_sc_col(cy)) return true; // empty scoring column reachable along row
                        cy += dir; steps++;
                    }
                    (void)blocked; // not used further
                }
            }
            return false;
        };

        // Don't flip a scored stone unless it's a cap that helps (you already check cap_would_help())
        Move fh = {"flip",{row,col},{row,col},{}, "horizontal"};
        Move fv = {"flip",{row,col},{row,col},{}, "vertical"};

        if(!is_own_score_cell_fixed(row, col, owner)){
            if(flip_is_constructive(board, row, col, owner, "horizontal"))
                moves.push_back(fh);
            if(flip_is_constructive(board, row, col, owner, "vertical"))
                moves.push_back(fv);
        } else if(cap_would_help()){
            if(flip_is_constructive(board, row, col, owner, "horizontal"))
                moves.push_back(fh);
        }
    }
    else{
        // river->stone flip (rare, usually to unblock). Keep as is.
        moves.push_back({"flip",{row,col},{row,col},{},""});
    }

    // --- ROTATE ---
    if(side == "river"){
        if(orientation == "horizontal"){
            moves.push_back({"rotate",{row,col},{row,col},{},"vertical"});
        }else{
            moves.push_back({"rotate",{row,col},{row,col},{},"horizontal"});
        }
    }

    // Deduplicate any rare duplicate moves from river exploration
    sort(moves.begin(), moves.end(), [](const Move& a, const Move& b){
        if(a.action != b.action) return a.action < b.action;
        if(a.from != b.from) return a.from < b.from;
        if(a.to != b.to) return a.to < b.to;
        if(a.pushed_to != b.pushed_to) return a.pushed_to < b.pushed_to;
        return a.orientation < b.orientation;
    });
    moves.erase(unique(moves.begin(), moves.end(), [](const Move& a, const Move& b){
        return a.action==b.action && a.from==b.from && a.to==b.to && a.pushed_to==b.pushed_to && a.orientation==b.orientation;
    }), moves.end());

    // Removed previous relaxation: do not allow scored stones to leave their scoring cells
    return moves;
}


//checking if any one of the palyer has one and thus recognizing the terminal node
bool is_terminal(const vector<vector<map<string, string>>>& board){
    int need = 4;
    if(!G_SCORE_COLS.empty()) need = (int)G_SCORE_COLS.size();
    else {
        // Fallback by board size heuristic
        need = (G_COLS<=12)?4:((G_COLS<=14)?5:6);
    }
    return count_in_score_area(board, "circle")>=need || count_in_score_area(board, "square")>=need;
}

//applying the minimax function
int min_value(const vector<vector<map<string, string>>>& board,string owner,int max_depth,int current_depth,int alpha,int beta){
    // Time or terminal/cutoff checks
    if(time_up()){
        // very fast fallback eval to exit quickly
        string opp = (owner=="circle")?"square":"circle";
        int val = 3000*(count_in_score_area(board, owner) - count_in_score_area(board, opp));
        return val;
    }
    if(is_terminal(board) || current_depth == max_depth){
        if(time_up()){
            string opp = (owner=="circle")?"square":"circle";
            int val = 3000*(count_in_score_area(board, owner) - count_in_score_area(board, opp));
            return val;
        }
        return evaluate_function(board, owner);
    }
    
    int num_rows = board.size();
    int num_cols = board[0].size();
    
    //getting all oponents moves
    vector<pair<Move, vector<vector<map<string, string>>>>> all_moves;
    string opponent = (owner == "circle") ? "square" : "circle";
    
    for(int row = 0; row < num_rows; row++){
        if(time_up()) break;
        for(int col = 0; col < num_cols; col++){
            if(time_up()) break;
            if(!board[row][col].empty() && board[row][col].at("owner") == opponent){
                vector<Move> current_moves = generate_all_moves(board, row, col);
                if(current_moves.empty()) continue;
                
                for(const auto& move : current_moves){
                    if(time_up()) break;
                    vector<vector<map<string, string>>> new_board = update_board(board, move);
                    if(new_board == board) continue;
                    
                    all_moves.push_back({move, new_board});
                }
            }
        }
    }
    
    // If no moves found, return evaluation 
    if(all_moves.empty()){
        return evaluate_function(board, owner);
    }
    
    //getting evaluation for each node and doing node ordeering for alpha beta pruning 
    vector<pair<int, pair<Move, vector<vector<map<string, string>>>>>>ordered_moves;

    //reordering in increasing order since opponent wants to minimize the evaluation fuctnoion 
    for(const auto& [move, next_board] : all_moves){
        // Use quick ordering score (from opponent perspective we want lower evaluate_function, but cheaper heuristic first)
        int h = quick_order_score(board, move); // use current board to avoid eval cost
        ordered_moves.push_back({h,{move, next_board}});
    }
    // Opponent minimizing: sort ascending by heuristic
    sort(ordered_moves.begin(),ordered_moves.end(),[](const auto& a,const auto& b){return a.first < b.first;});
    
    // Standard alpha-beta search with ordered moves
    int value = 1e9;
    for(const auto& [score, move_and_board] : ordered_moves){
        const auto& move = move_and_board.first;
        const auto& next_board = move_and_board.second;
        
        if(time_up()){
            string opp = (owner=="circle")?"square":"circle";
            return 3000*(count_in_score_area(board, owner) - count_in_score_area(board, opp));
        }
        int new_value = max_value(next_board, owner, max_depth, current_depth + 1, alpha, beta);
        if(new_value < value){
            value = new_value;
        }
        
        beta = min(beta, value);
        if(value <= alpha){
            return value; // Alpha-beta pruning
        }
        if(time_up()) break;
    }
    
    return value;
}

int max_value(const vector<vector<map<string, string>>>& board,string owner,int max_depth,int current_depth,int alpha,int beta){
    // Terminal / timeout
    if(time_up()){
        string opp = (owner=="circle")?"square":"circle";
        return 3000*(count_in_score_area(board, owner) - count_in_score_area(board, opp));
    }
    if(is_terminal(board) || current_depth == max_depth){
        if(time_up()){
            string opp = (owner=="circle")?"square":"circle";
            return 3000*(count_in_score_area(board, owner) - count_in_score_area(board, opp));
        }
        return evaluate_function(board, owner);
    }
    
    int num_rows = board.size();
    int num_cols = board[0].size();
    
    // Collect all our moves
    vector<pair<Move, vector<vector<map<string, string>>>>>all_moves;
    
    for(int row = 0; row < num_rows; row++) {
        if(time_up()) break;
        for(int col = 0; col < num_cols; col++) {
            if(time_up()) break;
            if(!board[row][col].empty() && board[row][col].at("owner") == owner) {
                vector<Move> current_moves = generate_all_moves(board, row, col);
                if(current_moves.empty()) continue;
                
                for(const auto& move : current_moves) {
                    if(time_up()) break;
                    vector<vector<map<string, string>>> new_board = update_board(board, move);
                    if(new_board == board) continue;
                    
                    all_moves.push_back({move, new_board});
                }
            }
        }
    }
    
    // If no moves found, return evaluation
    if(all_moves.empty()){
        return evaluate_function(board, owner);
    }
    
    // Simple move ordering by heuristic value - no pruning
    vector<pair<int, pair<Move, vector<vector<map<string, string>>>>>>ordered_moves;

    for(const auto& [move, next_board] : all_moves){
        int h = quick_order_score(board, move);
        ordered_moves.push_back({h,{move, next_board}});
    }
    // Maximizer: sort descending
    sort(ordered_moves.begin(), ordered_moves.end(),[](const auto& a,const auto& b){return a.first > b.first;});
    
    // Standard alpha-beta search with ordered moves
    int value = -1e9;
    for(const auto& [score, move_and_board] : ordered_moves){
        const auto& move = move_and_board.first;
        const auto& next_board = move_and_board.second;
        
        if(time_up()){
            string opp = (owner=="circle")?"square":"circle";
            return 3000*(count_in_score_area(board, owner) - count_in_score_area(board, opp));
        }
        int new_value = min_value(next_board,owner,max_depth,current_depth + 1,alpha,beta);
        if(new_value > value){
            value = new_value;
        }
        
        alpha = max(alpha, value);
        if(value >= beta){
            return value; // Alpha-beta pruning
        }
        if(time_up()) break;
    }
    
    return value;
}

//minimax algorithm upto a fixed depth using alpha beta pruning 
Move minimax_decision(const vector<vector<map<string,string>>>& board,const string& owner,int max_depth){
    int best_val = -1000000000;
    Move best_move = {"move",{0,0},{0,0},{},""};

    int alpha = -1000000000;
    int beta  =  1000000000;

    int num_rows = board.size();
    int num_cols = board[0].size();

    // Collect all possible moves from all pieces (owner)
    vector<pair<Move, vector<vector<map<string,string>>>>>all_moves;
    all_moves.reserve(128);
    for(int row = 0;row < num_rows;++row){
        for(int col = 0;col < num_cols;++col){
            if(board[row][col].empty())continue;
            if(board[row][col].at("owner") != owner)continue;

            vector<Move>current_moves = generate_all_moves(board, row, col);
            if(current_moves.empty())continue;

            for(const auto& mv : current_moves){
                auto new_board = update_board(board,mv);
                if(new_board == board)continue;
                all_moves.push_back({mv,std::move(new_board)});
            }
        }
    }

    if(all_moves.empty())return best_move;

    // Cheap heuristic ordering once (avoid expensive evaluate for ordering at root)
    vector<pair<int,int>> scores; scores.reserve(all_moves.size());
    for(int i=0;i<(int)all_moves.size();++i){
        int s = quick_order_score(board, all_moves[i].first);
        scores.emplace_back(s, i);
    }
    sort(scores.begin(), scores.end(),[](const auto &a,const auto &b){
        return a.first > b.first;});

    for(auto &p : scores){
        const Move &mv = all_moves[p.second].first;
        const auto &next_board = all_moves[p.second].second;

        if(time_up()) break;
        int v = min_value(next_board,owner,max_depth,1,alpha,beta);
        if(v > best_val){
            best_val = v;
            best_move = mv;
        }
        alpha = max(alpha, best_val);
    }
    return best_move;
}

//this fucntion is used to check for instant win , to check if after moving a single move if it could win 
// this is used to help with the blocking move 
Move check_for_instant_win(const vector<vector<map<string, string>>>& board,string owner,map<pair<int,int>,vector<Move>> moves){

    int n=board.size();
    int m=board[0].size();

    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            if(!board[i][j].empty() && board[i][j].at("owner") != owner){
                continue;
            }
            //checking if after a move the number in score arae becomes 4
            vector<Move>current_move = moves[make_pair(i,j)];
            for(auto it: current_move){
                vector<vector<map<string, string>>>updated = update_board(board,it);
                int need = G_SCORE_COLS.empty()?4:(int)G_SCORE_COLS.size();
                if(count_in_score_area(updated,owner)>=need){
                    return it;
                }
            }
        }
    }

    return  {"",{0,0},{0,0},{},""};
}

//finding if by some move i can stop my opponent from winning 

Move find_blocking_move(const vector<vector<map<string,string>>>& board,Move win_move,string owner,const map<pair<int,int>,vector<Move>>& moves){
   //getting the winning path 
    string opponent = (owner == "circle") ? "square" : "circle";
    vector<pair<int,int>> winning_path = trace_winning_path(board, win_move, opponent);
    //if winning apth wa sempty return empty move 
    if(winning_path.empty()){
        return {"",{},{},{},""};
    }
    //getting the move after which we can block the opponents winning move
    Move blocking_move=find_path_blocking_moves(board, winning_path, owner, moves);
    return blocking_move;
}

//tracing the winning path
vector<pair<int,int>> trace_winning_path(const vector<vector<map<string, string>>>& board,const Move& win_move,const string& opponent){
    vector<pair<int,int>>complete_path;
    //tracing if it was a move 
    //getting initial and final coordinates 
    if(win_move.action == "move"){
        int start_coor_row = win_move.from[0];
        int start_coor_col = win_move.from[1];
        int end_coor_row = win_move.to[0];
        int end_coor_col = win_move.to[1];

        vector<pair<int,int>>path_stack;
        set<pair<int,int>>visited;

        //simulating possible moves from that position and if it reached to my final postion taht will be the path wanted 
        if(simulate_move_path(board,start_coor_row,start_coor_col,end_coor_row,end_coor_col,path_stack,visited,-1,-1)){
            complete_path = path_stack;
        }
    }
    else if(win_move.action == "push"){
        int pusher_start_row = win_move.from[0], pusher_start_col = win_move.from[1];
        int push_target_row = win_move.to[0], push_target_col = win_move.to[1];
        int pushed_final_row = win_move.pushed_to[0], pushed_final_col = win_move.pushed_to[1];

        if(board[push_target_row][push_target_col].at("side")=="river"){
            return complete_path;
        }
        
        vector<pair<int,int>>path_stack;
        set<pair<int,int>>visited;

        if(!board[pusher_start_row][pusher_start_col].empty() && board[pusher_start_row][pusher_start_col].at("side")=="river"){
            string orientation = board[pusher_start_row][pusher_start_col].at("orientation");
            vector<pair<int,int>>directions;
            if(orientation == "horizontal"){
                directions = {{0,1}, {0,-1}};
            }
            else{
                directions = {{1,0}, {-1,0}};
            }
            for(auto [dx,dy] : directions){
                //simulating possible moves from that position and if it reached to my final postion taht will be the path wanted 
                if(simulate_move_path(board,push_target_row,push_target_col,pushed_final_row,pushed_final_col,path_stack,visited,dx,dy)){
                   complete_path = path_stack;
                }
            }
        }
        //stone ko stone push kar raha hai 
        else if(!board[pusher_start_row][pusher_start_col].empty() && board[pusher_start_row][pusher_start_col].at("side")=="stone"){
            path_stack.push_back({push_target_row,push_target_col});
            path_stack.push_back({pushed_final_row,pushed_final_col});
            complete_path = path_stack;
        }
    }

    return complete_path; 
}
//this is tarcing all the possible paths for given instruction using stack and returning the final path when found
bool simulate_move_path(const vector<vector<map<string, string>>>& board,int current_row,int current_col,int target_row, int target_col,vector<pair<int,int>>& path_stack,set<pair<int,int>>& visited,int dx1,int dy1){
    int n = board.size();
    int m = board[0].size();

    path_stack.push_back({current_row,current_col});
    if(current_row==target_row && current_col==target_col){
        return true;
    }
    // not revisiting something in the path
    visited.insert({current_row,current_col});

    if(!board[current_row][current_col].empty() && board[current_row][current_col].at("side")=="river"){
        string orientation = board[current_row][current_col].at("orientation");
        vector<pair<int,int>>directions;
        //choosing circetions based on the river if got a river in the path
        if(orientation == "horizontal"){
            directions = {{0,1}, {0,-1}};
        }
        else{
            directions = {{1,0}, {-1,0}};
        }
        // moving in possible directions and recursively calling the fucntion
        for(auto [dx,dy] : directions){
            int nx = current_row + dx;
            int ny = current_col + dy;

            if(nx >= 0 && nx < n && ny >= 0 && ny < m && board[nx][ny].empty()){
                if(!visited.count({nx,ny})){
                    if(simulate_move_path(board,nx,ny,target_row,target_col,path_stack,visited,dx,dy)){
                        //if foung path by this then return true
                        return true;
                    }
                }
            }
            else if(nx >= 0 && nx < n && ny >= 0 && ny < m && board[nx][ny].at("side") == "river"){
                if(!visited.count({nx,ny})){
                    if(simulate_move_path(board,nx,ny,target_row,target_col,path_stack,visited,dx,dy)){
                        return true;
                    }
                }
            }
        }
    }
    else{
        //going towards all directions wherever empty since no river 
        vector<pair<int,int>>directions = {{-1,0},{1,0},{0,-1},{0,1}};
        //it is if i was moving beacuse of river and checking it it is river directed movement , or we are just moving the fisrt move 
        if(dx1 == -1 && dy1 == -1){
            for (auto [dx, dy] : directions) {
                int nx = current_row + dx, ny = current_col + dy;
                
                if (nx >= 0 && nx < n && ny >= 0 && ny < m && !visited.count({nx, ny})) {
                    if(board[nx][ny].empty()){
                        if(nx == target_row && ny == target_col){
                            //inserting the coordinates in the stack
                            path_stack.push_back({nx,ny});
                            return true;
                        }
                    }
                    else if (board[nx][ny].at("side") == "river") {
                        if (simulate_move_path(board, nx, ny, target_row, target_col, path_stack, visited, dx1, dy1)) {
                            return true;
                        }
                    }
                }
            }
        }
        else{
        
            int nx = current_row + dx1;
            int ny = current_col + dy1;

            if(nx >= 0 && nx < n && ny >= 0 && ny < m && !visited.count({nx, ny})){
                if(simulate_move_path(board, nx, ny, target_row, target_col, path_stack, visited, dx1, dy1)){
                    return true;
                }
            }
        }
    }
    //poping for the one not in path 
    path_stack.pop_back();
    visited.erase({current_row, current_col});
    return false;

}

//getting the move after which it will no longer be able to get 4 scroing stoned in the scorign area 
Move find_path_blocking_moves(const vector<vector<map<string,string>>>& board,const vector<pair<int,int>>& path,const string& owner,const map<pair<int,int>,vector<Move>>& moves){
    Move blocking_move;

    for(const auto& pos: path){
        int row = pos.first;
        int col = pos.second;

        if(board[row][col].empty()){
            for(const auto& move_pair : moves){
               if(board[move_pair.first.first][move_pair.first.second].at("owner") != owner){
                continue;
               }
               const vector<Move>& piece_moves = move_pair.second;

               for(const Move& move: piece_moves){
                    //checking if for every move my destination could be palced anywhere in its path which willl stop its movement
                    if(move.action == "move" && move.to.size() >= 2 && move.to[0] == row && move.to[1] == col){
                        return move;
                    }
                    else if(move.action == "push" && !move.pushed_to.empty() && move.pushed_to[0] == row && move.pushed_to[1] == col){
                        return move;
                    }
               }
            }
        }
        //just fliiping if my river was helping
        else if(!board[row][col].empty() && board[row][col].at("side")=="river"){
            if(board[row][col].at("owner")==owner){
                return {"flip", {row, col}, {row, col}, {}, {}};
            }
        }
    }
    return {"", {}, {}, {}, ""};
}
    

// ---- Student Agent ----
class StudentAgent {
public:
    // it is created to check for redundant moves , if i get three consecutive moves taht are same i move a randome move 
    struct SimpleMove{
        string action=""; 
        int fx=-1, fy=-1; 
        int tx=-1, ty=-1; 
        int px=-1, py=-1; 
        string orientation=""; 
        bool valid=false; 
    };

    int root_turn = 0;

    explicit StudentAgent(string side) : side(std::move(side)), gen(rd()){ root_turn = 0; }

    Move choose(const vector<vector<map<string, string>>>& board,int row,int col,const vector<int>& score_cols,float current_player_time,float opponent_time){
        root_turn++;
        G_ROOT_TURN = root_turn; // set global for use in generate_all_moves
        
        // Initialize dynamic globals for board sizing & scoring columns
        G_ROWS = (int)board.size();
        G_COLS = board.empty()?0:(int)board[0].size();
        if(!score_cols.empty()){
            G_SCORE_COLS = score_cols; // provided
        } else {
            // Fallback heuristics by size
            if(G_COLS>=12){
                // assume center band of width 4/5/6 depending on size
                int need = (G_COLS<=12)?4:((G_COLS<=14)?5:6);
                int midL = G_COLS/2 - need/2;
                G_SCORE_COLS.clear();
                for(int k=0;k<need;k++) G_SCORE_COLS.push_back(midL+k);
            } else {
                G_SCORE_COLS = {4,5,6,7};
            }
        }

        // Adaptive time allocation by board size (simple heuristic)
        // FIXED: increased time to allow deeper search and better plan commitment
        double base_iter_time = 0.16; // small board
        if(G_ROWS>=15) base_iter_time = 0.22; // medium
        if(G_ROWS>=17) base_iter_time = 0.28; // large
        double time_limit = base_iter_time;
        if(current_player_time < 10) time_limit *= 0.5;
        else if(current_player_time < 30) time_limit *= 0.75;

        // ===== LANE PLANNER SETUP =====
        // Initialize cap row - FIXED: place cap ON the scoring row, not one before
        cap_row_self = (side=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();

        // Helper: lane scoring (opp blockers, total occupancy)
        auto laneScore = [&](int col)->pair<int,int>{
            int blockers=0, occ=0;
            int start = (side=="circle")? bottom_score_row_dynamic() : top_score_row_dynamic();
            int goal  = (side=="circle")? top_score_row_dynamic()    : bottom_score_row_dynamic();
            int step  = (side=="circle")? -1 : +1;
            for(int r=start; r!=goal+step; r+=step){
                if(r<0 || r>=G_ROWS || col<0 || col>=G_COLS) continue;
                if(board[r][col].empty()) continue;
                occ++;
                if(board[r][col].at("owner")!=side && board[r][col].at("side")=="stone") blockers++;
            }
            return {blockers, occ};
        };

        // Helper: check if column is in wing
        auto inWing = [&](int c){
            int m = G_COLS;
            int w = (m<=12)?3: (m<=14? 3:4);
            return (c < w) || (c >= m - w);
        };

        // Choose best sparse wing lane - FIXED: prefer near-empty columns first
        auto chooseSparseLane = [&](){
            int bestC=-1; int bestB=1e9, bestOcc=1e9;
            // First pass: prefer near-empty columns (occupancy <= 1)
            for(int c=0;c<G_COLS;c++){
                if(!inWing(c)) continue;
                auto [blk,occ]=laneScore(c);
                bool near_empty = (occ <= 1);
                if(near_empty && (blk < bestB || (blk==bestB && occ<bestOcc))){ bestB=blk; bestOcc=occ; bestC=c; }
            }
            if(bestC != -1) return bestC;
            // Fallback: if no near-empty column, use original rule
            for(int c=0;c<G_COLS;c++){
                if(!inWing(c)) continue;
                auto [blk,occ]=laneScore(c);
                if(blk < bestB || (blk==bestB && occ<bestOcc)){ bestB=blk; bestOcc=occ; bestC=c; }
            }
            return bestC;
        };

        // Pick/refresh the sparse lane if needed
        if(active_lane<0) active_lane = chooseSparseLane();
        if(backup_lane_L<0 || backup_lane_R<0){
            for(int d=1; d<G_COLS; ++d){
                int L = active_lane - d, R = active_lane + d;
                if(L>=0 && inWing(L) && backup_lane_L<0) backup_lane_L=L;
                if(R<G_COLS && inWing(R) && backup_lane_R<0) backup_lane_R=R;
                if(backup_lane_L>=0 && backup_lane_R>=0) break;
            }
        }

        // Count vertical river rungs in active lane
        auto laneHasVerticalRungs = [&](int col){
            int rungs=0, gaps=0;
            int start = (side=="circle")? bottom_score_row_dynamic() : top_score_row_dynamic();
            int goal  = (side=="circle")? top_score_row_dynamic()    : bottom_score_row_dynamic();
            int step  = (side=="circle")? -1 : +1;
            for(int r=start; r!=goal+step; r+=step){
                if(r<0 || r>=G_ROWS || col<0 || col>=G_COLS) continue;
                if(!board[r][col].empty() && board[r][col].at("owner")==side
                   && board[r][col].at("side")=="river"
                   && board[r][col].at("orientation")=="vertical") rungs++;
                else if(board[r][col].empty()) gaps++;
            }
            return make_pair(rungs, gaps);
        };

        // Check if horizontal cap exists
        auto capPresent = [&](int col){
            if(cap_row_self<0 || cap_row_self>=G_ROWS || col<0 || col>=G_COLS) return false;
            if(board[cap_row_self][col].empty()) return false;
            auto &cell=board[cap_row_self][col];
            return cell.at("owner")==side && cell.at("side")=="river" && cell.at("orientation")=="horizontal";
        };

        // Update phase based on lane progress
        {
            auto [rungs, gaps] = laneHasVerticalRungs(active_lane);
            bool haveCap = capPresent(active_lane);

            if(phase==SEED_VERT && rungs >= 2) phase = SEED_CAP; // FIXED: 2 rungs is enough on small boards
            if(phase==SEED_CAP && haveCap) phase = LADDER_PUSH;

            // If stuck, try backup lane
            if(no_progress_counter>=3) lane_stuck_turns++;
            if(lane_stuck_turns>=2){
                int tryLane = (backup_lane_L>=0)? backup_lane_L : ((backup_lane_R>=0)? backup_lane_R : -1);
                if(tryLane>=0) { active_lane = tryLane; phase = SEED_VERT; lane_stuck_turns=0; }
            }
        }

        // Set global planner state for quick_order_score
        G_ACTIVE_LANE = active_lane;
        G_CAP_ROW_SELF = cap_row_self;
        G_PLAN_PHASE = (int)phase;
        // ===== END LANE PLANNER SETUP =====

        // Compute current best manhattan distance for our side (stones only)
        int curr_best_md = 100;
        for(int i=0;i<G_ROWS;i++){
            for(int j=0;j<G_COLS;j++){
                if(!board[i][j].empty() && board[i][j].at("owner")==side && board[i][j].at("side")=="stone"){
                    curr_best_md = min(curr_best_md, min_distance_from_scoring_area(side, i, j));
                }
            }
        }
        if(curr_best_md < last_best_md){ last_best_md = curr_best_md; no_progress_counter = 0; }
        else { no_progress_counter++; }
        // Be more aggressive about pushing progress to break stalemates
        G_PROGRESS_MODE = (no_progress_counter>=3?2:(no_progress_counter>=1?1:0));

        // Decay lane & piece cooldowns (per root call) - FIXED: removed duplicate decay
        for(auto it = lane_cooldown.begin(); it!=lane_cooldown.end();){ if(it->second<=1) it = lane_cooldown.erase(it); else { --(it->second); ++it; } }
        for(auto it = piece_cooldown.begin(); it!=piece_cooldown.end();){ if(it->second<=1) it = piece_cooldown.erase(it); else { --(it->second); ++it; } }

    // Repetition / cycle detection: compute simple (owner-only) position hash (before move)
    uint64_t current_simple_hash = simple_position_hash(board);
    position_history.push_back(current_simple_hash);
    if(position_history.size() > 64) position_history.erase(position_history.begin());

    // it is getting all moves together for the given state of board will help later
        map<pair<int,int>,vector<Move>> moves_for_current;
        int n= board.size();
        int m=board[0].size();

        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                if(!board[i][j].empty()){
                    moves_for_current[{i,j}] = generate_all_moves(board,i,j);
                }
            }
        }

        // Hard preference: build rivers on wings before center
        {
            int mcols = G_COLS;
            int sc_min = 0, sc_max = mcols-1;
            if(!G_SCORE_COLS.empty()){
                sc_min = *min_element(G_SCORE_COLS.begin(), G_SCORE_COLS.end());
                sc_max = *max_element(G_SCORE_COLS.begin(), G_SCORE_COLS.end());
            } else {
                int w = (mcols<=12)?4:((mcols<=14)?5:6);
                int s = max(0, mcols/2 - w/2); sc_min = s; sc_max = min(mcols-1, s+w-1);
            }
            int centerL = max(0, sc_min - 1);
            int centerR = min(mcols-1, sc_max + 1);

            auto col_occupancy = [&](int col){
                int occ=0; for(int r=0;r<G_ROWS;++r){ if(!board[r][col].empty()) ++occ; } return occ;
            };

            bool has_wing_flip = false;
            bool has_empty_wing_flip = false;
            for(const auto &kv : moves_for_current){
                int r = kv.first.first, c = kv.first.second;
                if(board[r][c].empty()) continue;
                if(board[r][c].at("owner")!=side) continue;
                for(const auto &mv : kv.second){
                    if(mv.action=="flip" && (c < centerL || c > centerR)) {
                        has_wing_flip = true; 
                        if(col_occupancy(c) <= 1) has_empty_wing_flip = true;
                        break; 
                    }
                }
                if(has_wing_flip) break;
            }
            // Removed hard ban on center flips - now using penalty in quick_order_score instead
            // This allows search to prove center flips valuable when they are (e.g., instant wins)
            /*
            if(has_wing_flip){
                for(auto &kv : moves_for_current){
                    int r = kv.first.first, c = kv.first.second;
                    if(!(c>=centerL && c<=centerR)) continue;
                    auto &vec = kv.second;
                    vec.erase(remove_if(vec.begin(), vec.end(), [&](const Move& m){ return m.action=="flip"; }), vec.end());
                }
            }
            */

            // If we have any "empty-wing" flip available, restrict flips to only those columns (outer and low occupancy)
            // FIXED: add exception for plan-critical flips (active lane vertical rungs and cap)
            if(has_empty_wing_flip){
                for(auto &kv : moves_for_current){
                    int r = kv.first.first, c = kv.first.second;
                    int occ = col_occupancy(c);
                    auto &vec = kv.second;
                    vec.erase(remove_if(vec.begin(), vec.end(), [&](const Move& m){
                        // Keep plan-critical flips: vertical in active lane (SEED_VERT) or horizontal cap (SEED_CAP)
                        bool keep_for_plan =
                            (m.action=="flip" && c==active_lane &&
                             ((phase==SEED_VERT && m.orientation=="vertical") ||
                              (phase==SEED_CAP && r==cap_row_self && m.orientation=="horizontal")));
                        bool wing_empty = (c < centerL || c > centerR) && (occ <= 1);
                        return (!keep_for_plan) && !wing_empty && (m.action=="flip");
                    }), vec.end());
                }
            }
        }

        // Gather all legal moves for fallback (flat list)
        vector<Move> flat_all_moves;
        for(auto &kv : moves_for_current){ for(auto &mv : kv.second){ flat_all_moves.push_back(mv); } }

        // Enhanced anti-repetition filters (pre-search):
        //  - drop moves that recreate any of the last 8 simple positions
        //  - drop moves that would create a position that already appeared 2+ times (triple repetition)
        if(!flat_all_moves.empty()){
            auto creates_recent_position = [&](const Move& mv){
                if(mv.action=="flip" || mv.action=="rotate") return false; // don't block flips/rotations on simple hash
                auto nb = update_board(board, mv);
                uint64_t pos_h = simple_position_hash(nb);
                size_t check_count = std::min(position_history.size(), (size_t)8);
                for(size_t i = position_history.size() - check_count; i < position_history.size(); i++){
                    if(position_history[i] == pos_h) return true;
                }
                return false;
            };
            auto creates_triple_repetition = [&](const Move& mv){
                if(mv.action=="flip" || mv.action=="rotate") return false; // allow local topology changes
                auto nb = update_board(board, mv);
                uint64_t pos_h = simple_position_hash(nb);
                int count = 0;
                for(size_t i = 0; i < position_history.size(); i++) if(position_history[i]==pos_h) count++;
                return count >= 2;
            };
            for(auto &kv : moves_for_current){
                auto &vec = kv.second;
                vec.erase(remove_if(vec.begin(), vec.end(), [&](const Move& m){
                    return creates_recent_position(m) || creates_triple_repetition(m);
                }), vec.end());
            }
            // rebuild flat list after filtering
            flat_all_moves.clear();
            for(auto &kv : moves_for_current){ for(auto &mv : kv.second){ flat_all_moves.push_back(mv); } }
        }
        if(flat_all_moves.empty()){
            // if all filtered away, regenerate once to ensure we have legal options
            for(int i=0;i<n;i++) for(int j=0;j<m;j++) if(!board[i][j].empty()) moves_for_current[{i,j}] = generate_all_moves(board,i,j);
            for(auto &kv : moves_for_current){ for(auto &mv : kv.second){ flat_all_moves.push_back(mv); } }
        }

        // Fallback: if absolutely no moves (e.g., side has no pieces), return a benign pass-like flip if possible
        if(flat_all_moves.empty()){
            // scan again for any piece of ours to flip or rotate
            for(int i=0;i<n;i++){
                for(int j=0;j<m;j++){
                    if(!board[i][j].empty() && board[i][j].at("owner")==side){
                        if(board[i][j].at("side")=="stone"){
                            return convert_move_format({"flip",{i,j},{i,j},{},"horizontal"});
                        } else if(board[i][j].at("side")=="river"){
                            return convert_move_format({"rotate",{i,j},{i,j},{},"horizontal"});
                        }
                    }
                }
            }
            // final fallback: return a dummy move (engine should ignore) but avoids undefined behavior
            return convert_move_format({"move",{0,0},{0,0},{},""});
        }

        // Root-level anti-repetition filtering: drop moves that recreate recent positions
        {
            auto would_repeat_recent = [&](const Move& mv){
                auto nb = update_board(board, mv);
                uint64_t h = simple_board_hash_no_turn(nb);
                int cnt=0; int window=12;
                for(int k=(int)board_history_no_turn.size()-1, c=0; k>=0 && c<window; --k,++c){ if(board_history_no_turn[k]==h) ++cnt; }
                return cnt>=2; // already appears twice in recent history
            };
            for(auto &kv : moves_for_current){
                auto &vec = kv.second;
                vector<Move> filtered; filtered.reserve(vec.size());
                for(const auto &mv : vec){ if(!would_repeat_recent(mv)) filtered.push_back(mv); }
                if(!filtered.empty()) vec.swap(filtered); // keep at least one move per piece
            }
            // rebuild flat list
            flat_all_moves.clear();
            for(auto &kv : moves_for_current){ for(auto &mv : kv.second){ flat_all_moves.push_back(mv); } }
            if(flat_all_moves.empty()){
                // if filtering wiped everything, restore original unfiltered moves
                for(int i=0;i<n;i++) for(int j=0;j<m;j++) if(!board[i][j].empty()) moves_for_current[{i,j}] = generate_all_moves(board,i,j);
                for(auto &kv : moves_for_current){ for(auto &mv : kv.second){ flat_all_moves.push_back(mv); } }
            }
        }

        // Detect position cycles (multi-move loops). If same simple position seen 2+ times, we are in a loop
        Move instant_win = check_for_instant_win(board, side,moves_for_current);
        if(instant_win.action != ""){
            return convert_move_format(instant_win);
        }
        
        // Check if opponent has an instant win and block it
        string opponent = (side == "circle") ? "square" : "circle";
        Move opponent_win = check_for_instant_win(board, opponent,moves_for_current);
        if(opponent_win.action != ""){
            // Find a move that blocks opponent's instant win
            Move blocking_move = find_blocking_move(board, opponent_win, side,moves_for_current);
            if(blocking_move.action != ""){
                return convert_move_format(blocking_move);
            }
        }
        
    // Iterative deepening with dynamic time control already set
    auto start_time = chrono::high_resolution_clock::now();
    // Establish hard deadline (slightly above intended limit to include overhead) and reset timeout flag
    G_TIMEOUT_HIT = false;
    double hard_limit = time_limit * 1.05; // 5% grace
    G_DEADLINE = start_time + chrono::duration_cast<chrono::high_resolution_clock::duration>(chrono::duration<double>(hard_limit));
        
    // Start with minimal depth and dynamic cap based on board size and move count
    // Estimate branching to adapt depth
    size_t est_moves = 0; for(auto &p : moves_for_current) est_moves += p.second.size();
    int max_depth_cap = 4;
    if(G_ROWS>=15) max_depth_cap = 3;
    if(G_ROWS>=17) max_depth_cap = 3;
    if(est_moves > 200) max_depth_cap = min(max_depth_cap, 2);
    else if(est_moves > 120) max_depth_cap = min(max_depth_cap, 3);
    // On medium boards with modest branching, allow one deeper search for better quality
    if(G_ROWS>=15 && est_moves <= 100) max_depth_cap = max(max_depth_cap, 4);
    int depth = 1;
    Move best_move = flat_all_moves[0]; // seed with first legal move to avoid invalid default
        
        // Iterative deepening loop
        while(true){
            // Get best move at current depth
            Move current_best = minimax_decision(board,side,depth);
            if(time_up()) break; // stop iterative deepening if time exceeded
            
            // Update our best move
            if(current_best.action != ""){
                best_move = current_best;
            }
            
            // Check if we have time for another iteration
            auto current_time = chrono::high_resolution_clock::now();
            chrono::duration<double>elapsed = current_time - start_time;
            
            // If we're getting close to our time limit, or reached a reasonable depth, stop
            if(elapsed.count() > time_limit || depth >= max_depth_cap || time_up()){
                break;
            }
            
            // Otherwise, try a deeper search
            depth++;
        }
        
    // Record hashes for repetition detection
    uint64_t board_hash_no_turn = simple_board_hash_no_turn(board);
    board_history_no_turn.push_back(board_hash_no_turn);
    if(board_history_no_turn.size()>128) board_history_no_turn.erase(board_history_no_turn.begin());

        // If chosen move recreates a very recent board state (side-agnostic) or simple-position loop, try to select a non-repeating alternative
        {
            auto next_board = update_board(board, best_move);
            uint64_t next_hash = simple_board_hash_no_turn(next_board);
            auto recent_count = [&](uint64_t h,int window){int cnt=0; for(int k=(int)board_history_no_turn.size()-1,c=0;k>=0&&c<window;--k,++c){ if(board_history_no_turn[k]==h) ++cnt;} return cnt;};
            bool repeats_recent = (recent_count(next_hash, 10) >= 2);
            bool ab_cycle=false;
            if(board_history_no_turn.size()>=3){
                uint64_t last = board_history_no_turn.back();
                uint64_t prev = board_history_no_turn[board_history_no_turn.size()-2];
                uint64_t prev2= board_history_no_turn[board_history_no_turn.size()-3];
                if(prev2==last && next_hash==prev) ab_cycle=true; // ABAB
            }
            // count simple-position repetitions
            int simple_reps = 0; {
                uint64_t cur = position_history.empty()?0:position_history.back();
                for(size_t i=0;i+1<position_history.size();++i) if(position_history[i]==cur) simple_reps++;
            }
            if(repeats_recent || ab_cycle || simple_reps>=2){
                // Put contested lane on cooldown to force exploration elsewhere
                if(!best_move.from.empty()){
                    int cf = best_move.from[1]; lane_cooldown[cf] = max(lane_cooldown[cf], 5);
                    piece_cooldown[pos_key(best_move.from[0], best_move.from[1])] = 5;
                }
                if(!best_move.to.empty()){
                    int ct = best_move.to[1]; lane_cooldown[ct] = max(lane_cooldown[ct], 5);
                }
                vector<Move> alt_moves;
                unordered_set<string> seen;
                auto enc = [](const Move& m){ ostringstream oss; oss<<m.action<<":"; for(int x: m.from) oss<<x<<","; oss<<"/"; for(int x: m.to) oss<<x<<","; oss<<"/"; for(int x: m.pushed_to) oss<<x<<","; oss<<m.orientation; return oss.str(); };
                for(int i=0;i<G_ROWS;i++){
                    for(int j=0;j<G_COLS;j++){
                        if(!board[i][j].empty() && board[i][j].at("owner")==side){
                            for(auto &mv: generate_all_moves(board,i,j)){
                                string key = enc(mv); if(!seen.insert(key).second) continue;
                                auto nb = update_board(board, mv);
                                uint64_t h = simple_board_hash_no_turn(nb);
                                bool bad = (recent_count(h,10) >= 2);
                                // avoid recent simple positions too
                                uint64_t sh = simple_position_hash(nb);
                                size_t chk = std::min(position_history.size(), (size_t)12);
                                for(size_t ii=position_history.size()-chk; ii<position_history.size(); ++ii){ if(position_history[ii]==sh){ bad=true; break; } }
                                if(!bad && board_history_no_turn.size()>=3){
                                    uint64_t last = board_history_no_turn.back();
                                    uint64_t prev = board_history_no_turn[board_history_no_turn.size()-2];
                                    uint64_t prev2= board_history_no_turn[board_history_no_turn.size()-3];
                                    if(prev2==last && h==prev) bad=true;
                                }
                                // Avoid immediate reverse push of last move (break push ping-pong)
                                if(!bad && last_move_simple.valid && last_move_simple.action=="push" && mv.action=="push"){
                                    if(last_move_simple.tx==mv.from[0] && last_move_simple.ty==mv.from[1] && last_move_simple.fx==mv.to[0] && last_move_simple.fy==mv.to[1]){
                                        bad = true;
                                    }
                                }
                                if(!bad) alt_moves.push_back(mv);
                            }
                        }
                    }
                }
                // Emergency: heavy simple-position cycling, pick random move that changes simple hash
                if(alt_moves.empty() && simple_reps>=5){
                    vector<Move> any_diff;
                    for(int i=0;i<G_ROWS;i++) for(int j=0;j<G_COLS;j++) if(!board[i][j].empty() && board[i][j].at("owner")==side){
                        for(const auto &mv: generate_all_moves(board,i,j)){
                            auto nb = update_board(board, mv);
                            if(simple_position_hash(nb) != simple_position_hash(board)) any_diff.push_back(mv);
                        }
                    }
                    if(!any_diff.empty()){
                        std::uniform_int_distribution<int> dist(0,(int)any_diff.size()-1);
                        best_move = any_diff[dist(gen)];
                    }
                }
                if(!alt_moves.empty()){
                    int bestH=-1e9; Move repl=alt_moves[0];
                    for(const auto& mv: alt_moves){
                        int h=quick_order_score(board,mv);
                        auto nb = update_board(board, mv);
                        uint64_t hh = simple_board_hash_no_turn(nb);
                        int rc = recent_count(hh,10);
                        h += (rc==0?15:(rc==1?5:0));
                        // Lane & piece cooldown penalty to steer away from contested columns/pieces
                        if(!mv.from.empty() && lane_cooldown.count(mv.from[1])) h -= 800;
                        if(!mv.to.empty()   && lane_cooldown.count(mv.to[1]))   h -= 500;
                        if(!mv.from.empty() && piece_cooldown.count(pos_key(mv.from[0], mv.from[1]))) h -= 400;
                        if(h>bestH){bestH=h; repl=mv;}
                    }
                    best_move = repl;
                }
            }
        }

        //here is where we check for three cosecutive same moves and move the first move not equal to it found 
        SimpleMove current_simple = to_simple(best_move);
        if(last_move_simple.valid && same_simple(last_move_simple, current_simple)){
            consecutive_repeat_count++;
        } else {
            consecutive_repeat_count = 0;
        }
        if(consecutive_repeat_count >= 2){ // allow at most 2 repeats (3rd triggers change)
            // Build alternate move list excluding repeating pattern and flips of same cell
            vector<Move> alt_moves;
            unordered_set<string> seen_keys;
            auto encode = [](const Move& m){
                ostringstream oss; oss<<m.action<<":";
                for(int x: m.from) oss<<x<<","; oss<<"/"; for(int x: m.to) oss<<x<<","; oss<<"/"; for(int x: m.pushed_to) oss<<x<<","; oss<<m.orientation; return oss.str(); };
            for(int i=0;i<G_ROWS;i++){
                for(int j=0;j<G_COLS;j++){
                    if(!board[i][j].empty() && board[i][j].at("owner")==side){
                        auto gm = generate_all_moves(board,i,j);
                        for(auto &mv : gm){
                            auto sm = to_simple(mv);
                            if(same_simple(sm,current_simple)) continue; // skip exact repeats
                            // Avoid flipping same cell back immediately unless scoring
                            if(mv.action=="flip" && current_simple.action=="flip" && current_simple.fx==sm.fx && current_simple.fy==sm.fy) continue;
                            // Avoid direct reverse of last push (break push ping-pong)
                            if(current_simple.action=="push" && sm.action=="push" && current_simple.tx==sm.fx && current_simple.ty==sm.fy && current_simple.fx==sm.tx && current_simple.fy==sm.ty) continue;
                            string key = encode(mv);
                            if(seen_keys.insert(key).second) alt_moves.push_back(mv);
                        }
                    }
                }
            }
            if(!alt_moves.empty()){
                // Pick heuristic-best alternative using quick_order_score
                int bestH = -1e9; Move repl = alt_moves[0];
                for(const auto& mv : alt_moves){
                    int h = quick_order_score(board, mv);
                    // Penalize moves that recreate immediate earlier board (ping-pong)
                    if(board_history_no_turn.size()>=3){
                        uint64_t last   = board_history_no_turn.back();
                        uint64_t prev   = board_history_no_turn[board_history_no_turn.size()-2];
                        uint64_t prev2  = board_history_no_turn[board_history_no_turn.size()-3];
                        if(prev2==last){
                            h -= 900; // stronger discourage returning to position A B A
                        }
                    }
                    // Lane & piece cooldown penalty
                    if(!mv.from.empty() && lane_cooldown.count(mv.from[1])) h -= 800;
                    if(!mv.to.empty()   && lane_cooldown.count(mv.to[1]))   h -= 500;
                    if(!mv.from.empty() && piece_cooldown.count(pos_key(mv.from[0], mv.from[1]))) h -= 400;
                    // novelty bonus: reward states not seen recently
                    auto nb = update_board(board, mv);
                    uint64_t hsh = simple_board_hash_no_turn(nb);
                    int seen_cnt=0; for(int k=(int)board_history_no_turn.size()-1,c=0;k>=0&&c<10;--k,++c){ if(board_history_no_turn[k]==hsh) ++seen_cnt; }
                    h += (seen_cnt==0?15:(seen_cnt==1?5:0));
                    if(h > bestH){ bestH = h; repl = mv; }
                }
                best_move = repl;
                current_simple = to_simple(best_move);
            }
            consecutive_repeat_count = 0;
        }
        last_move_simple = current_simple;
        recent_moves.push_back(current_simple);
        if(recent_moves.size()>40) recent_moves.erase(recent_moves.begin());

        return convert_move_format(best_move);
    }

private:
    string side;
    random_device rd;
    mt19937 gen;

    // Repetition detection
    vector<uint64_t> position_history; // rolling hash list
    vector<uint64_t> board_history_no_turn; // side-agnostic repetition list

    // Lane cooldown to avoid contested columns for a few root turns
    unordered_map<int,int> lane_cooldown; // col -> turns remaining
    unordered_map<string,int> piece_cooldown; // "r,c" -> turns remaining
    static inline string pos_key(int r,int c){ ostringstream os; os<<r<<","<<c; return os.str(); }

    // Lane planner persistent state
    enum PlanPhase { SEED_VERT=0, SEED_CAP=1, LADDER_PUSH=2, CONSOLIDATE=3 };
    int active_lane = -1;          // chosen sparse column
    int backup_lane_L = -1, backup_lane_R = -1;
    PlanPhase phase = SEED_VERT;
    int cap_row_self = -1;         // row where horizontal cap river should live
    int lane_stuck_turns = 0;      // if we stop progressing on this lane

    SimpleMove last_move_simple;          // last move we executed
    int consecutive_repeat_count = 0;     // repeats of same move
    vector<SimpleMove>recent_moves;      // history window (optional future use)
    int last_best_md = 999;              // last root best MD to our scoring band
    int no_progress_counter = 0;         // root turns without improving best MD

    SimpleMove to_simple(const Move& m){
        SimpleMove s; s.action=m.action;s.valid=true;
        if(!m.from.empty()){ 
            s.fx=m.from[0]; 
            s.fy=m.from[1]; 
        }
        if(!m.to.empty()){ 
            s.tx=m.to[0];   
            s.ty=m.to[1]; 
        }
        if(!m.pushed_to.empty()){ 
            s.px=m.pushed_to[0]; 
            s.py=m.pushed_to[1]; 
        }
        s.orientation=m.orientation; 
        return s; 
    }

    bool same_simple(const SimpleMove& a,const SimpleMove& b){
        if(a.action!=b.action) return false;
        if(a.action=="push") return a.fx==b.fx && a.fy==b.fy && a.tx==b.tx && a.ty==b.ty && a.px==b.px && a.py==b.py;
        if(a.action=="move") return a.fx==b.fx && a.fy==b.fy && a.tx==b.tx && a.ty==b.ty;
        if(a.action=="flip"||a.action=="rotate") return a.fx==b.fx && a.fy==b.fy && a.orientation==b.orientation; 
        return a.fx==b.fx && a.fy==b.fy && a.tx==b.tx && a.ty==b.ty; 
    }

    // Opponent reconstruction removed (simplified approach).
    
    // Helper to convert our internal move format to the expected external format
    Move convert_move_format(const Move& m){
        Move converted_move;
        converted_move.action = m.action;
        
        // Swap the coordinates for from_pos: [row,col] -> [col,row]
        if(!m.from.empty()){
            converted_move.from = {m.from[1],m.from[0]};
        }
        
        // Swap the coordinates for to_pos: [row,col] -> [col,row]
        if(!m.to.empty()){
            converted_move.to = {m.to[1], m.to[0]};
        }
        
        // Swap the coordinates for pushed_to: [row,col] -> [col,row]
        if(!m.pushed_to.empty()){
            converted_move.pushed_to = {m.pushed_to[1],m.pushed_to[0]};
        }else{
            converted_move.pushed_to = {};
        }
        
        // Pass through the orientation
        converted_move.orientation = m.orientation;
        
        return converted_move;
    }
};


//this the update board function , it updates the board on the basisi fo given move 
std::vector<std::vector<std::map<std::string, std::string>>> update_board(const std::vector<std::vector<std::map<std::string, std::string>>>& board,Move move){
    std::vector<std::vector<std::map<std::string, std::string>>> new_board = board;
    int start_x = move.from[1];
    int start_y = move.from[0];
    
    if(new_board[start_y][start_x].empty()){
        return new_board;
    }

    // move
    if(move.action == "move"){
        int end_x = move.to[1];
        int end_y = move.to[0];
        // moving the piece
        new_board[end_y][end_x] = new_board[start_y][start_x];
        new_board[start_y][start_x].clear(); 
    }
    // push
    else if(move.action == "push"){
        int opponent_end_x = move.pushed_to[1];
        int opponent_end_y = move.pushed_to[0];
        int new_x = move.to[1];
        int new_y = move.to[0];

        string pusher_side = new_board[start_y][start_x].at("side");
        // pushing a stone piece
        
        new_board[opponent_end_y][opponent_end_x] = new_board[new_y][new_x];
        new_board[new_y][new_x].clear();
        new_board[new_y][new_x] = new_board[start_y][start_x];

        if(pusher_side == "river"){
            new_board[new_y][new_x]["side"] = "stone";
            new_board[new_y][new_x].erase("orientation");
        }

        new_board[start_y][start_x].clear();
    }
    // flip
    else if(move.action == "flip"){
        string current_side = new_board[start_y][start_x].at("side"); // stone or river currently
        if(current_side == "stone"){
            // flipping to river and choosing orientation
            new_board[start_y][start_x]["side"] = "river";
            new_board[start_y][start_x]["orientation"] = move.orientation;
        }
        else if(current_side == "river"){
            new_board[start_y][start_x]["side"] = "stone";
            new_board[start_y][start_x].erase("orientation");
        }
        
    }
    // rotate
    else if(move.action == "rotate"){
        if(new_board[start_y][start_x].at("orientation") == "horizontal"){
            new_board[start_y][start_x]["orientation"] = "vertical";
        }
        else{
            new_board[start_y][start_x]["orientation"] = "horizontal";
        }
    }

    // check if after the move i dont get the same board again
    return new_board;
}

// Blocking move => 

// evaluation function helpers:
// 1. n_self n_opp
// 2. manhattan distance (to be enhanced) => river bhi - current manhattan distance
// 3. threat detection =>  number of threats
// 4. Stone-River Synergy: Stones positioned to use friendly rivers effectively => number of stones that can reach to a river 

// in assigning weights check the values of n_self and n_opp

//this is to track the numbe rof path that i can get from my own river moves 
int river_synergy(const std::vector<std::vector<std::map<std::string, std::string>>>& board, const string& owner,const map<pair<int,int>,vector<Move>>& moves){
    int n = board.size();
    int m = board[0].size();
    string opponent = (owner=="circle")?"square":"circle";
    vector<pair<int,int>> dirs = {{-1,0},{1,0},{0,-1},{0,1}};

    int score = 0;
    int friendly_rivers = 0;
    int used_friendly_rivers = 0;

    // Track which rivers are used
    vector<vector<int>> river_used(n, vector<int>(m,0));

    auto inside=[&](int x,int y){return x>=0 && x<n && y>=0 && y<m;};
    //getting all my rivers and naming them as friendly riverrs

    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            if(board[i][j].empty()) continue;
            if(board[i][j].at("side")=="river" && board[i][j].at("owner")==owner){
                friendly_rivers++;
            }
        }
    }


    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            if(board[i][j].empty()) continue;
            if(board[i][j].at("side")!="stone" || board[i][j].at("owner")!=owner) continue;

            int local = 0; 
            bool has_friendly_adj=false;
            for(auto [dx,dy]: dirs){
                int nx=i+dx, ny=j+dy; if(!inside(nx,ny) || board[nx][ny].empty()) continue;
                if(board[nx][ny].at("side")=="river"){
                    //incraesing local for the presence of the river 
                    if(board[nx][ny].at("owner")==owner){
                        local += 5; 
                        has_friendly_adj=true; 
                        river_used[nx][ny]=1;
                    } else if(board[nx][ny].at("owner")==opponent){
                        local -= 2; // adjacency to enemy river slightly bad
                    }
                }
            }
            if(!has_friendly_adj){
                //check if any friendly river at manhattan 2 reachable by moving to an empty neighbor first
                bool potential=false;
                for(auto [dx,dy]: dirs){
                    int midx=i+dx, midy=j+dy; if(!inside(midx,midy) || !board[midx][midy].empty()) continue;
                    for(auto [dx2,dy2]: dirs){
                        int rx=midx+dx2;
                        int ry=midy+dy2; 
                        if(!inside(rx,ry) || board[rx][ry].empty()) continue;
                        if(board[rx][ry].at("side")=="river" && board[rx][ry].at("owner")==owner){ 
                            potential=true; 
                            river_used[rx][ry]=1; 
                            break; 
                        }
                    }
                    if(potential) break;
                }
                if(potential) local += 2; // smaller than direct adjacency
            }
            // Cap per-stone
            if(local>7) local=7;
            if(local<-3) local=-3;
            score += local;
        }
    }

    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            if(!board[i][j].empty() && board[i][j].at("side")=="river" && board[i][j].at("owner")==owner){
                if(river_used[i][j]) used_friendly_rivers++;
            }
        }
    }
    int idle = friendly_rivers - used_friendly_rivers;
    score -= idle * 4; // penalize idle rivers
    // Soft cap overall to avoid dominance
    if(score>60) score=60; if(score<-40) score=-40;
    return score;
}
//computing the n self and n opp and returning its differncve , it is numbe rof stones in scor area 

pair<int,int> compute_n_self_opp(const std::vector<std::vector<std::map<std::string, std::string>>>& board,const string& owner){
    int num_circle = count_in_score_area(board, "circle");
    int num_square = count_in_score_area(board, "square");
    int n_self = (owner=="circle") ? num_circle : num_square;
    int n_opp  = (owner=="circle") ? num_square : num_circle;
    return {n_self, n_opp};
}

//computing m self  where i get the improvemnt after one move 
//that is the number of stones in score area increases
int compute_m_self(const std::vector<std::vector<std::map<std::string, std::string>>>& board,const string& owner,const map<pair<int,int>,vector<Move>>& moves){
    vector<pair<int,int>> score_area;
    int target_row = (owner=="circle")? top_score_row_dynamic() : bottom_score_row_dynamic();
    if(G_SCORE_COLS.empty()){
        for(int c: {4,5,6,7}) score_area.push_back({target_row, c});
    } else {
        for(int c: G_SCORE_COLS) score_area.push_back({target_row, c});
    }

    int n = board.size();
    int m = board[0].size();

    int current_stones_in_score = 0;
    for(const auto& pos : score_area){
        if(!board[pos.first][pos.second].empty() && board[pos.first][pos.second].at("owner") == owner && board[pos.first][pos.second].at("side") == "stone"){
            current_stones_in_score++;
        }
    }

    int score_bonus = 0;
    if(current_stones_in_score == 1) score_bonus = 1000;
    else if(current_stones_in_score == 2) score_bonus = 3000;
    else if(current_stones_in_score == 3) score_bonus = 8000;
    else if((int)score_area.size()>=4 && current_stones_in_score >= (int)score_area.size()) score_bonus = 1000000;

    int potential_moves = 0;
    for(int row = 0; row < n; row++){
        for(int col = 0; col < m; col++){
            if(!board[row][col].empty() && board[row][col].at("owner") == owner){
                auto it = moves.find(make_pair(row, col));
                if (it == moves.end()) continue;
                const vector<Move>& current_moves = it->second;
                for(const auto& move : current_moves){
                    int change = 0;
                    if(board[row][col].at("side") == "stone" && move.action == "move"){
                        pair<int,int> old_pos = make_pair(move.from[0], move.from[1]);
                        pair<int,int> new_pos = make_pair(move.to[0], move.to[1]);
                        for(const auto& p : score_area){ if(old_pos==p) change--; if(new_pos==p) change++; }
                    } else if(board[row][col].at("side") == "river" && move.action == "flip"){
                        pair<int,int> new_pos = make_pair(move.to[0], move.to[1]);
                        for(const auto& p : score_area){ if(new_pos==p) change++; }
                    } else if(move.action == "push"){
                        pair<int,int> old_pos = make_pair(move.from[0], move.from[1]);
                        pair<int,int> new_pos = make_pair(move.to[0], move.to[1]);
                        pair<int,int> pushed_pos = make_pair(move.pushed_to[0], move.pushed_to[1]);
                        bool is_pushing_stone = !board[move.to[0]][move.to[1]].empty() && board[move.to[0]][move.to[1]].at("side") == "stone";
                        for(const auto& p : score_area){
                            if(old_pos==p && board[move.from[0]][move.from[1]].at("side") == "stone") change--;
                            if(new_pos==p && board[move.from[0]][move.from[1]].at("side") == "stone") change++;
                            if(is_pushing_stone && board[move.to[0]][move.to[1]].at("owner")==owner){
                                if(new_pos==p) change--; if(pushed_pos==p) change++;
                            }
                        }
                    }
                    if(change>0) potential_moves++;
                }
            }
        }
    }
    return score_bonus + 100 * potential_moves;
}


int evaluate_function(const std::vector<std::vector<std::map<std::string, std::string>>>& board, const string& owner) {

    //generating all moves for the board
    int n= board.size();
    int m= board[0].size();
    map<pair<int,int>,vector<Move>> all_moves;
    string opponent = (owner == "circle") ? "square" : "circle";

    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            vector<Move>current_pos = generate_all_moves(board,i,j);
            all_moves[make_pair(i,j)]=current_pos;
        }
    }

    // ATTACK defaults
    int weight_score      = 1500;
    int weight_manhattan  = -420;   
    int weight_threat     = 600;
    int weight_river_syn  = 50;     
    int weight_m_score    = 1100;   
    int weight_oppbest    = 150; 

    auto in_own_score = [&](const string& who, int x, int y){
        return is_own_score_area(x,y,who);
    };

    // Optional flip penalty: discourage flipping unless it creates value.
    int friendly_rivers=0, friendly_stones=0;
    for(int i=0;i<n;i++) for(int j=0;j<m;j++) if(!board[i][j].empty() && board[i][j].at("owner")==owner){
        if(board[i][j].at("side")=="river") friendly_rivers++; else friendly_stones++; }
    int allow_margin = 2; // allow a couple of rivers without penalty
    int excess = max(0, friendly_rivers - (friendly_stones + allow_margin)); 
    int flip_penalty = excess * 80;

    // heuristics
    auto it = compute_n_self_opp(board, owner);
    int n_self = it.first;
    int n_opp = it.second;
    int score = n_self - n_opp;

    auto it1 = manhattan_distance(board, owner,all_moves);
    int num_manhattan = it1[0] - it1[1];

    int num_threats = it1[2] - it1[3];

    // unpack for phase switch:
    int self_best = it1[0];
    int opp_best  = it1[1];
    int self_thr  = it1[2];
    int opp_thr   = it1[3];

    int river_syn_self = river_synergy(board,owner,all_moves);
    int river_syn_opp = river_synergy(board,opponent,all_moves);

    // Calculate potential to enter scoring area
    int m_self = compute_m_self(board, owner, all_moves);
    int m_opp = compute_m_self(board, opponent, all_moves);

    int opp_best_live = 100, opp_thr_live = 0;
    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            if (board[i][j].empty()) continue;
            if (board[i][j].at("owner") != opponent) continue;

            // ignore opponent stones that are already scored
            if (board[i][j].at("side")=="stone" && in_own_score(opponent,i,j)) continue;

            // reuse the same single-ply cost logic as in manhattan_distance:
            int r = find_next_river_distance(board, opponent, i, j, all_moves);
            auto MD = [&](int grow){
                int best = 100;
                if(!G_SCORE_COLS.empty()){
                    for(int sc: G_SCORE_COLS) best = min(best, abs(i-grow) + abs(j-sc));
                } else {
                    int mcols = G_COLS; int w=(mcols<=12)?4:((mcols<=14)?5:6); int s=max(0,mcols/2 - w/2); int e=min(mcols-1,s+w-1);
                    for(int sc=s; sc<=e; ++sc) best = min(best, abs(i-grow) + abs(j-sc));
                }
                return best;
            };
            int goal_row = (opponent=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
            int bfs_dist = MD(goal_row);
            int cost;
            if (board[i][j].at("side")=="stone"){
                int via_river = (r>=100) ? 100 : (1 + r);
                cost = min(bfs_dist, via_river);
            }else{
                int walk_after_flip  = 1 + bfs_dist;
                int river_after_flip = (r>=100) ? 100 : (2 + r);
                cost = min(walk_after_flip, river_after_flip);
            }

            opp_best_live = min(opp_best_live, cost);
            if (cost <= 3) ++opp_thr_live;
        }
    }

    // defense or attack
    const bool DEF = ((opp_best_live <= 2) && (opp_thr_live >= 2));

    if (DEF) {
        weight_score     = 1400;
        weight_manhattan =  -60;
        weight_threat    = -900;  
        weight_river_syn =   10;  
        weight_m_score   =  600;  
        weight_oppbest   =  400;  
    }

    // lane analysis
    auto inside = [&](int x,int y){ return 0<=x && x<n && 0<=y && y<m; };
    int goal_row = (owner=="circle") ? 2 : 10;
    int start_row = (owner=="circle") ? 10 : 2;          
    int step      = (owner=="circle") ? -1  : +1;

    // Count blockers in a column, but distinguish between stones and rivers
    auto col_analysis = [&](int col) -> pair<int,int> {
        int stone_blockers = 0;
        int friendly_rivers = 0;
        
        if (owner=="circle"){
            for(int r=start_row; r!=goal_row+step; r+=step){
                if(!inside(r,col)) continue;
                if(board[r][col].empty()) continue;
                
                string piece_owner = board[r][col].at("owner");
                string piece_side = board[r][col].at("side");
                
                if(piece_owner != owner) {
                    stone_blockers++; // opponent piece blocks
                } else if(piece_side == "river") {
                    friendly_rivers++; // our river helps
                }
            }
        } else {
            for(int r=start_row; r!=goal_row+step; r+=step){
                if(!inside(r,col)) continue;
                if(board[r][col].empty()) continue;
                
                string piece_owner = board[r][col].at("owner");
                string piece_side = board[r][col].at("side");
                
                if(piece_owner != owner) {
                    stone_blockers++;
                } else if(piece_side == "river") {
                    friendly_rivers++;
                }
            }
        }
        return {stone_blockers, friendly_rivers};
    };

    // Analyzing all columns and categorize them
    int best_wing_clear = 1e9, best_center_clear = 1e9;
    int best_wing_supported = 1e9, best_center_supported = 1e9;
    
    // Dynamic center band around scoring columns
    int sc_min = 0, sc_max = m-1;
    if(!G_SCORE_COLS.empty()){
        sc_min = *min_element(G_SCORE_COLS.begin(), G_SCORE_COLS.end());
        sc_max = *max_element(G_SCORE_COLS.begin(), G_SCORE_COLS.end());
    } else {
        int w = (m<=12)?4:((m<=14)?5:6);
        int s = max(0, m/2 - w/2);
        sc_min = s; sc_max = min(m-1, s+w-1);
    }
    int centerL = max(0, sc_min - 1);
    int centerR = min(m-1, sc_max + 1);

    for(int c=0; c<m; c++){
        auto [blockers, rivers] = col_analysis(c);
        bool is_wing = (c < centerL || c > centerR);
        
        int clear_score = blockers;
        int supported_score = max(0, blockers - rivers); // rivers reduce effective blockers
        
        if(is_wing) {
            best_wing_clear = min(best_wing_clear, clear_score);
            best_wing_supported = min(best_wing_supported, supported_score);
        } 
        else {
            best_center_clear = min(best_center_clear, clear_score);
            best_center_supported = min(best_center_supported, supported_score);
        }
    }

    // Count pieces in different zones
    int center_pieces = 0, wing_pieces = 0;
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            if(!board[i][j].empty() && board[i][j].at("owner")==owner && board[i][j].at("side")=="stone"){
                if(j >= centerL && j <= centerR) center_pieces++;
                else wing_pieces++;
            }
        }
    }
    
    int lane_preference_bonus = 0;
    if(best_wing_clear <= best_center_clear - 2) {
        lane_preference_bonus += 300;
    }
    else if(best_center_supported <= best_wing_clear && center_pieces <= 3) {
        lane_preference_bonus -= 100; 
    }
    else if(best_wing_clear <= best_center_clear) {
        lane_preference_bonus += 100; 
    }

    // Only penalizing center when it's actually problematic
    int center_congestion = 0;
    for(int i=0; i<n; i++){
        for(int c=centerL; c<=centerR; c++){
            if(!board[i][c].empty()) center_congestion++;
        }
    }

    // Progressive penalty scaled by center area — more tolerant on larger boards
    int corridor_penalty = 0;
    int center_w = max(1, centerR - centerL + 1);
    int total_center_cells = n * center_w;
    double density = total_center_cells>0? (double)center_congestion/(double)total_center_cells : 0.0;
    double dens_thresh = (G_COLS<=12?0.45:(G_COLS<=14?0.48:0.52));
    if(density > dens_thresh){
        double excess = density - dens_thresh;
        double scale = (G_COLS<=12?4000.0:(G_COLS<=14?3600.0:3200.0));
        corridor_penalty = (int)(excess * scale);
    }
    
    // Additional penalty for having too many of our own pieces in center without support
    int our_center_overuse = max(0, center_pieces - 4); // Allow up to 4 center pieces
    if(our_center_overuse > 0 && best_center_supported > 2) { // Only if center is poorly supported
        corridor_penalty += our_center_overuse * our_center_overuse * 100;
    }

    int bridge_hits = 0, enemy_bridge_hits = 0, wing_bridge = 0;
    auto in_wing_col = [&](int y){ return (y < centerL || y > centerR); };

    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            if (board[i][j].empty()) continue;
            if (board[i][j].at("owner")!=owner || board[i][j].at("side")!="stone") continue;

            const pair<int,int> dirs[4]={{1,0},{-1,0},{0,1},{0,-1}};
            for (auto [dx,dy] : dirs){
                int nx=i+dx, ny=j+dy;
                if (!inside(nx,ny) || board[nx][ny].empty()) continue;
                if (board[nx][ny].at("side")!="river") continue;

                bool river_is_mine = (board[nx][ny].at("owner")==owner);
                string ori = board[nx][ny].at("orientation");

                bool helps = (ori=="vertical") || (ori=="horizontal");

                if (helps){
                    if (river_is_mine) bridge_hits++; else enemy_bridge_hits++;
                    if (in_wing_col(ny)) wing_bridge++; 
                }
            }
        }
    }

    bridge_hits       = min(bridge_hits, 6);
    enemy_bridge_hits = min(enemy_bridge_hits, 6);
    wing_bridge       = min(wing_bridge, 6);

    // --- RUNWAY + CAP REWARD (sparse edge columns) ---
    // Goal: encourage a vertical "runway" of our vertical rivers stacked in a sparse edge column
    // and a horizontal "cap" river on the scoring row in the same column that can push sideways
    // into a scoring cell.
    auto is_scoring_col = [&](int c){
        if(!G_SCORE_COLS.empty()) return find(G_SCORE_COLS.begin(), G_SCORE_COLS.end(), c)!=G_SCORE_COLS.end();
        int mcols=m; int w=(mcols<=12)?4:((mcols<=14)?5:6); int s=max(0,mcols/2 - w/2); int e=min(mcols-1,s+w-1); return c>=s && c<=e;
    };
    int edge_k = (G_COLS<=12?3:(G_COLS<=14?3:4));
    auto is_edge_sparse_col = [&](int c){
        if(!(c<edge_k || c>=m-edge_k)) return false; // edge band
        int occ=0; for(int r=0;r<n;++r) if(!board[r][c].empty()) ++occ;
        return occ<=1; // nearly empty column
    };
    int target_row = (owner=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
    int toward = (owner=="circle") ? +1 : -1; // row direction from scoring row into field

    int runway_score = 0, cap_score = 0, runway_cap_synergy = 0;
    for(int c=0;c<m;++c){
        if(!is_edge_sparse_col(c)) continue;
        // Count vertical runway segments in this column
        int vcount = 0, near_goal = 0;
        for(int r=0;r<n;++r){
            if(board[r][c].empty()) continue;
            if(board[r][c].at("owner")==owner && board[r][c].at("side")=="river" && board[r][c].at("orientation")=="vertical"){
                ++vcount; if(abs(r - target_row) <= 2) ++near_goal;
            }
        }
        runway_score += vcount*30 + near_goal*10;
        // Cap at scoring row in same column
        bool cap=false;
        if(target_row>=0 && target_row<n && !board[target_row][c].empty()){
            auto &cell = board[target_row][c];
            if(cell.at("owner")==owner && cell.at("side")=="river" && cell.at("orientation")=="horizontal") cap=true;
        }
        if(cap){
            cap_score += 200;
            // Check immediate push potential: friendly stone along the row with clear path to a scoring column
            for(int dir : {-1,+1}){
                int y = c + dir; bool blocked=false; int steps=0; const int MAXS=10;
                bool saw_stone=false, can_score=false;
                while(y>=0 && y<m && steps<MAXS){
                    if(!board[target_row][y].empty()){
                        if(!saw_stone && board[target_row][y].at("owner")==owner && board[target_row][y].at("side")=="stone"){
                            saw_stone=true; // candidate to push
                        } else { blocked=true; break; }
                    } else {
                        if(is_scoring_col(y)) { can_score=true; break; }
                    }
                    y += dir; ++steps;
                }
                if(saw_stone && can_score && !blocked){ cap_score += 160; break; }
            }
            // Synergy: cap + vertical river just inside the field (adjacent row)
            int adj = target_row + toward;
            if(adj>=0 && adj<n && !board[adj][c].empty()){
                if(board[adj][c].at("owner")==owner && board[adj][c].at("side")=="river" && board[adj][c].at("orientation")=="vertical"){
                    runway_cap_synergy += 260;
                }
            }
        }
    }

    // ---- BALANCED WEIGHTS ----
    int w_lane_pref_att   = 4;     // Stronger adaptive lane preference
    int w_bridge_att      = 220;   
    int w_enemybridge_att = 110;    
    int w_wingbridge_att  = 220;   
    int w_corridor_pen_att = 3;    // Stronger corridor crowding penalty

    int w_lane_pref_def   = 2;
    int w_bridge_def      = 160;
    int w_enemybridge_def = 90;
    int w_wingbridge_def  = 170;
    int w_corridor_pen_def = 2;

    int w_lane_pref   = DEF ? w_lane_pref_def   : w_lane_pref_att;
    int w_bridge      = DEF ? w_bridge_def      : w_bridge_att;
    int w_enemyb      = DEF ? w_enemybridge_def : w_enemybridge_att;
    int w_wingbridge  = DEF ? w_wingbridge_def  : w_wingbridge_att;
    int w_corridor_pen = DEF ? w_corridor_pen_def : w_corridor_pen_att;

    long long eval_total =
          weight_score     * score
        + weight_manhattan * num_manhattan
        + weight_threat    * num_threats
        + weight_river_syn * (river_syn_self - river_syn_opp)
        + weight_m_score   * (m_self - m_opp)
        + weight_oppbest   * opp_best_live  // FIXED: use opp_best_live instead of opp_best
        - flip_penalty;

    eval_total += w_lane_pref   * lane_preference_bonus;    // Intelligent path selection
    eval_total += w_bridge      * bridge_hits;
    eval_total += w_enemyb      * enemy_bridge_hits;
    eval_total += w_wingbridge  * wing_bridge;
    eval_total -= w_corridor_pen * corridor_penalty;       // Adaptive crowding penalty

    // Apply runway + cap incentives
    eval_total += runway_score;
    eval_total += cap_score;
    eval_total += runway_cap_synergy;

    set<int> occupied_wing_cols, occupied_center_cols;
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            if(!board[i][j].empty() && board[i][j].at("owner")==owner && board[i][j].at("side")=="stone"){
                if(j < centerL || j > centerR) {
                    occupied_wing_cols.insert(j);
                } else {
                    occupied_center_cols.insert(j);
                }
            }
        }
    }
    
    int total_dispersion = occupied_wing_cols.size() + occupied_center_cols.size();
    int dispersion_bonus = total_dispersion * 80; // Moderate bonus for spreading out
    eval_total += dispersion_bonus;

    int ride_gain_self_sum = 0, ride_gain_opp_sum = 0;
    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            if (board[i][j].empty()) continue;

            // own stones
            if (board[i][j].at("owner")==owner && board[i][j].at("side")=="stone"){
                int before = min_distance_from_scoring_area(owner, i, j);
                int after  = find_next_river_distance(board, owner, i, j, all_moves); // best landing in 1 move
                if (after < 100) ride_gain_self_sum += max(0, before - after);
            }
            // opponent stones
            else if (board[i][j].at("owner")==opponent && board[i][j].at("side")=="stone"){
                int before = min_distance_from_scoring_area(opponent, i, j);
                int after  = find_next_river_distance(board, opponent, i, j, all_moves);
                if (after < 100) ride_gain_opp_sum += max(0, before - after);
            }
        }
    }

    // Small, safe weights: reward our ride gain; punish theirs (slightly more)
    const int W_RIDE_SELF = 28;
    const int W_RIDE_OPP  = 18;

    eval_total += W_RIDE_SELF * ride_gain_self_sum;
    eval_total -= W_RIDE_OPP  * ride_gain_opp_sum;

    int in_score_self = 0, in_score_opp = 0;
    int near1_self = 0, near2_self = 0, near4_self = 0;
    int near1_opp  = 0, near2_opp  = 0, near4_opp  = 0;
    int can_score_next_self = 0, can_score_next_opp = 0;

    string opp = (owner=="circle") ? "square" : "circle";

    for (int r = 0; r < (int)board.size(); ++r){
        for (int c = 0; c < (int)board[0].size(); ++c){
            if (board[r][c].empty() || board[r][c].at("side")!="stone") continue;

            if (board[r][c].at("owner") == owner){
                int dist = min_distance_from_scoring_area(owner, r, c);
                if (is_own_score_area(r, c, owner)) ++in_score_self;
                else if (dist == 1) ++near1_self;
                else if (dist <= 2) ++near2_self;
                else if (dist <= 4) ++near4_self;

                // Can enter scoring area in ONE move (plain move or river ride)
                int d1 = find_next_river_distance(board, owner, r, c, all_moves);
                if (d1 == 0) ++can_score_next_self;
            } 
            else { // opponent stone
                int dist = min_distance_from_scoring_area(opp, r, c);
                if (is_own_score_area(r, c, opp)) ++in_score_opp;
                else if (dist == 1) ++near1_opp;
                else if (dist <= 2) ++near2_opp;
                else if (dist <= 4) ++near4_opp;

                int d1 = find_next_river_distance(board, opp, r, c, all_moves);
                if (d1 == 0) ++can_score_next_opp;
            }
        }
    }

    const int W_IN_SCORE_SELF    = 2200;
    const int W_IN_SCORE_OPP     = 2400;

    const int W_NEAR1_SELF       = 700;   // 1 step away → push to finish
    const int W_NEAR2_SELF       = 380;   // 2 steps away
    const int W_NEAR4_SELF       = 160;   // ≤4 steps away

    const int W_NEAR1_OPP        = 820;   // deny their finish
    const int W_NEAR2_OPP        = 420;
    const int W_NEAR4_OPP        = 180;

    const int W_SCORE_NEXT_SELF  = 1600;  // can enter in one move (incl. river)
    const int W_SCORE_NEXT_OPP   = 1800;

    // applying
    eval_total += W_IN_SCORE_SELF * in_score_self;
    eval_total -= W_IN_SCORE_OPP  * in_score_opp;

    eval_total += W_NEAR1_SELF * near1_self + W_NEAR2_SELF * near2_self + W_NEAR4_SELF * near4_self;
    eval_total -= W_NEAR1_OPP  * near1_opp  + W_NEAR2_OPP  * near2_opp  + W_NEAR4_OPP  * near4_opp;

    eval_total += W_SCORE_NEXT_SELF * can_score_next_self;
    eval_total -= W_SCORE_NEXT_OPP  * can_score_next_opp;


    // pulling stones that are close in rows toward scoring columns (4..7)
    auto nearest_scoring_col_dist = [&](const string& /*pov*/, int /*r*/, int c){
        int best = 100;
        if(!G_SCORE_COLS.empty()){
            for(int sc: G_SCORE_COLS) best = min(best, abs(sc - c));
        } else {
            for(int sc=sc_min; sc<=sc_max; ++sc) best = min(best, abs(sc - c));
        }
        return best; // 0 means already in a scoring column
    };
    auto row_gap_to_goal = [&](const string& pov, int r){
        return abs(r - ((pov=="circle")? top_score_row_dynamic() : bottom_score_row_dynamic()));
    };

    int side_pull_self = 0, side_pull_opp = 0;

    for(int r=0; r<n; ++r){
        for(int c=0; c<m; ++c){
            if (board[r][c].empty() || board[r][c].at("side")!="stone") continue;

            if (board[r][c].at("owner")==owner){
                int rd = row_gap_to_goal(owner, r);
                if (rd <= 2){
                    int cd = nearest_scoring_col_dist(owner, r, c);
                    // stronger when just 1 row away
                    side_pull_self += (rd==1 ? (4 - cd) : (2 - cd));
                }
            } else {
                int rd = row_gap_to_goal(opp, r);
                if (rd <= 2){
                    int cd = nearest_scoring_col_dist(opp, r, c);
                    side_pull_opp += (rd==1 ? (4 - cd) : (2 - cd));
                }
            }
        }
    }

    const int W_SIDE_PULL = 140; // bump if still shy about sliding left
    eval_total += W_SIDE_PULL * (side_pull_self - side_pull_opp);

    int finish_now_self = 0, finish_now_opp = 0;
    for (int r=0; r<n; ++r){
        for (int c=0; c<m; ++c){
            if (board[r][c].empty() || board[r][c].at("side")!="stone") continue;

            const string who = board[r][c].at("owner");
            auto itmv = all_moves.find({r,c});
            if (itmv == all_moves.end()) continue;
            for (const auto& mv : itmv->second){
                if (mv.action=="move"){
                    if (who==owner && is_own_score_area(mv.to[0], mv.to[1], owner)){
                        ++finish_now_self; break;
                    }
                    if (who!=owner){
                        string oppo = (owner=="circle")? "square":"circle";
                        if (is_own_score_area(mv.to[0], mv.to[1], oppo)){
                            ++finish_now_opp; break;
                        }
                    }
                }
            }
        }
    }
    const int W_FINISH_NOW_SELF = 2600;
    const int W_FINISH_NOW_OPP  = 2800;
    eval_total += W_FINISH_NOW_SELF * finish_now_self;
    eval_total -= W_FINISH_NOW_OPP  * finish_now_opp;

    if (eval_total > 2000000000LL) eval_total = 2000000000LL;
    if (eval_total < -2000000000LL) eval_total = -2000000000LL;

    return (int)eval_total;
}


// ---- PyBind11 bindings ----
PYBIND11_MODULE(student_agent_module, m) {
    py::class_<Move>(m, "Move")
        .def_readonly("action", &Move::action)
        .def_readonly("from_pos", &Move::from)
        .def_readonly("to_pos", &Move::to)
        .def_readonly("pushed_to", &Move::pushed_to)
        .def_readonly("orientation", &Move::orientation);

    py::class_<StudentAgent>(m, "StudentAgent")
        .def(py::init<std::string>())
        .def("choose", &StudentAgent::choose);
}
