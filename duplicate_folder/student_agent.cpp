#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <set>
#include <array>
#include <chrono>
#include <algorithm>
#include <queue>
#include <sstream>
#include <climits>
#include <deque>
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
static inline int scored_delta_after(const vector<vector<map<string, string>>>& board, const Move& mv, const string& side);
bool is_terminal(const vector<vector<map<string, string>>>& board);
bool is_in_score_area(int row,int column,string owner);
// Helper returning true if (row,col) is in OWNER's own scoring area
static inline bool is_own_score_area(int row,int col,const string& owner){
    return is_own_score_cell_dyn(row,col,owner);
}

// Forward declarations for heuristics used in quick_order_score
int min_distance_from_scoring_area(const string& owner,int x,int y);
static inline bool is_own_score_cell_fixed(int row, int col, const string& owner);
int count_in_score_area(const vector<vector<map<string, string>>>& board,string owner);

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

// Symmetry-aware hash: vertical mirror around board center
static inline uint64_t simple_fullpos_hash_mirror(const vector<vector<map<string,string>>>& board, const string& side_to_move){
    // Compute vertical-mirrored board, reuse your hashing
    int R=(int)board.size(), C=(int)board[0].size();
    vector<vector<map<string,string>>> M = board;
    for(int r=0;r<R;r++) for(int c=0;c<C;c++) M[r][c] = board[r][C-1-c];
    return simple_fullpos_hash(M, side_to_move);
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

// Lightweight move-ordering heuristic: higher is better for the mover
static int quick_order_score(const vector<vector<map<string,string>>>& board, const Move& mv){
    if(mv.from.size()<2) return 0;
    int fr = mv.from[0], fc = mv.from[1];
    if(fr<0 || fr>=(int)board.size() || fc<0 || fc>=(int)board[0].size()) return 0;
    if(board[fr][fc].empty()) return 0;
    string mover = board[fr][fc].at("owner");
    string side  = board[fr][fc].at("side");

    auto md = [&](int r,int c){ return min_distance_from_scoring_area(mover, r, c); };

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
        // pushing opponent away from their goal is good
        string pushed_owner = board[target_r][target_c].at("owner");
        auto md_pushed = [&](int r,int c){ return min_distance_from_scoring_area(pushed_owner, r, c); };

        int before = md(fr, fc);
        int after  = md(mv.to[0], mv.to[1]);
        score += (before - after) * (prog_weight/5 + 10);

        int before_p = md_pushed(target_r, target_c);
        int after_p  = md_pushed(pr, pc);
        int delta_p  = after_p - before_p;

        if(pushed_owner != mover){
            // Pushing opponent *away* from their goal is very good
            score += delta_p * 24;  // was 8, now much stronger

            // Extra bonus if they were already a big threat (close to scoring)
            if(before_p <= 3 && delta_p > 0){
                score += 200 * delta_p;
            }
        } else {
            // Pushing our own stone away from our goal is bad
            if(delta_p > 0){
                score -= delta_p * 16;
            }
        }

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

// ----- Transposition Table + Move Ordering State -----
struct TTEntry { int depth; int value; int flag; Move best; };
static unordered_map<uint64_t, TTEntry> TT;
static const int TT_EXACT=0, TT_LOWER=1, TT_UPPER=2;

static const int MAX_PLY = 128;
static Move killer1[MAX_PLY], killer2[MAX_PLY];

static unordered_map<long long,int> historyHeu;
static inline long long histKey(const Move& m){
    int a = (m.action=="move"?1: m.action=="push"?2: m.action=="flip"?3:4);
    int fr=m.from.empty()?0:(m.from[0]*64+m.from[1]);
    int to=m.to.empty()?0:(m.to[0]*64+m.to[1]);
    return ((long long)fr<<21) | ((long long)to<<9) | a;
}

struct ScoredMove { int s; Move m; };

static int finish_now_delta(const vector<vector<map<string,string>>>& b,
                            const Move& mv, const string& side){
    if(mv.action=="move" && !mv.to.empty()){
        if(is_own_score_cell_dyn(mv.to[0], mv.to[1], side)) return 200000;
    }
    if(mv.action=="push" && !mv.pushed_to.empty()){
        int r=mv.pushed_to[0], c=mv.pushed_to[1];
        if(is_own_score_cell_dyn(r,c,side)) return 120000;
    }
    return 0;
}

static void order_moves(const vector<vector<map<string,string>>>& board,
                        vector<Move>& moves, const Move* hashBest,
                        int ply, const string& side){
    // Compute center band for opening wing flip bonus
    // Change #4: use G_SCORE_COLS only for center bounds
    int C = board.empty()?0:(int)board[0].size();
    int sc_min = *min_element(G_SCORE_COLS.begin(), G_SCORE_COLS.end());
    int sc_max = *max_element(G_SCORE_COLS.begin(), G_SCORE_COLS.end());
    int centerL = max(0, sc_min - 1);
    int centerR = min(C-1, sc_max + 1);
    
    vector<ScoredMove> tmp; tmp.reserve(moves.size());
    for(auto &mv: moves){
        int s = 0;
        if(hashBest && 
           mv.action==hashBest->action && mv.from==hashBest->from &&
           mv.to==hashBest->to && mv.pushed_to==hashBest->pushed_to &&
           mv.orientation==hashBest->orientation) s += 500000;
        s += finish_now_delta(board, mv, side);
        s += quick_order_score(board, mv) * 10;
        
        // Opening bias: hard-prefer river seeds in the opening
        int opening_bias = (ply==0 ? 1 : 0);
        if(opening_bias && mv.action=="flip" && !mv.from.empty()){
            int fc = mv.from[1];
            // outer means fc < centerL or fc > centerR
            if(fc < centerL || fc > centerR) s += 12000;
        }
        
        if(ply<MAX_PLY){
            if(mv.action==killer1[ply].action && mv.from==killer1[ply].from &&
               mv.to==killer1[ply].to && mv.pushed_to==killer1[ply].pushed_to) s += 9000;
            else if(mv.action==killer2[ply].action && mv.from==killer2[ply].from &&
               mv.to==killer2[ply].to && mv.pushed_to==killer2[ply].pushed_to) s += 7000;
        }
        s += min(6000, historyHeu[histKey(mv)]);
        tmp.push_back({s, mv});
    }
    sort(tmp.begin(), tmp.end(), [](const ScoredMove& a, const ScoredMove& b){
        return a.s > b.s;
    });
    for(size_t i=0;i<moves.size();++i) moves[i]=tmp[i].m;
}

static bool tt_probe(uint64_t key, int depth, int alpha, int beta, int& out, Move& best){
    auto it=TT.find(key);
    if(it==TT.end()) return false;
    const TTEntry &e = it->second;
    best = e.best; // even if depth is shallow, we can use hash move for ordering
    if(e.depth < depth) return false;
    if(e.flag==TT_EXACT){ out = e.value; return true; }
    if(e.flag==TT_LOWER && e.value >= beta){ out = e.value; return true; }
    if(e.flag==TT_UPPER && e.value <= alpha){ out = e.value; return true; }
    return false;
}
static void tt_store(uint64_t key, int depth, int val, int flag, const Move& best){
    auto it=TT.find(key);
    if(it==TT.end() || it->second.depth <= depth){
        TT[key] = TTEntry{depth, val, flag, best};
    }
}

static int tactical_quiescence(const vector<vector<map<string,string>>>& board,
                               const string& side, int alpha, int beta){
    string opp = (side=="circle")?"square":"circle";
    int stand = evaluate_function(board, side);
    if(stand >= beta) return stand;
    if(stand > alpha) alpha = stand;

    vector<Move> tMoves;
    int R=board.size(), C=board[0].size();
    bool have=false;
    for(int i=0;i<R;i++) for(int j=0;j<C;j++){
        if(board[i][j].empty() || board[i][j].at("owner")!=side) continue;
        auto ms = generate_all_moves(board,i,j);
        for(auto &mv: ms){
            bool keep=false;
            if(mv.action=="move" && !mv.to.empty() && is_own_score_cell_dyn(mv.to[0], mv.to[1], side)) keep=true;
            if(mv.action=="push" && !mv.pushed_to.empty() && is_own_score_cell_dyn(mv.pushed_to[0], mv.pushed_to[1], side)) keep=true;
            if(!keep){
                if(!mv.to.empty() && is_opponent_score_cell_dyn(mv.to[0], mv.to[1], side)) keep=true;
                if(!mv.pushed_to.empty() && is_opponent_score_cell_dyn(mv.pushed_to[0], mv.pushed_to[1], side)) keep=true;
            }
            if(keep){ tMoves.push_back(mv); have=true; }
        }
    }
    if(!have) return stand;

    sort(tMoves.begin(), tMoves.end(), [&](const Move& a, const Move& b){
        return quick_order_score(board,a) > quick_order_score(board,b);
    });

    for(auto &mv: tMoves){
        auto nb = update_board(board, mv);
        int sc = evaluate_function(nb, side);
        if(sc >= beta) return sc;
        if(sc > alpha) alpha = sc;
    }
    return alpha;
}

// Global stats for debugging
static int G_NODES_SEARCHED = 0;
static int G_ALPHA_CUTOFFS = 0;
static int G_BETA_CUTOFFS = 0;
static int G_MAX_DEPTH_REACHED = 0;

// Helper: Detect critical position needing deeper search
// Returns extension depth (0 = no extension, 1-2 = extend by this much)
static inline int get_extension_depth(const vector<vector<map<string,string>>>& board,
                                     const Move& mv, const string& side) {
    // Check if move directly scores
    if(mv.action == "move" && !mv.to.empty()) {
        if(is_own_score_cell_dyn(mv.to[0], mv.to[1], side)) return 2; // scoring move = +2 depth
    }
    if(mv.action == "push" && !mv.pushed_to.empty()) {
        if(is_own_score_cell_dyn(mv.pushed_to[0], mv.pushed_to[1], side)) return 2; // push to score = +2 depth
    }
    
    // Check if opponent can score next move (critical defense needed)
    string opp = (side == "circle") ? "square" : "circle";
    int opp_score_row = (opp == "circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
    
    // Count opponent pieces very close to scoring (distance <= 2)
    int near_threats = 0;
    for(int i = 0; i < (int)board.size(); i++) {
        for(int j = 0; j < (int)board[0].size(); j++) {
            if(board[i][j].empty()) continue;
            if(board[i][j].at("owner") != opp || board[i][j].at("side") != "stone") continue;
            int dist = min_distance_from_scoring_area(opp, i, j);
            if(dist <= 2) near_threats++;
        }
    }
    
    if(near_threats >= 2) return 1; // multiple threats = +1 depth
    
    return 0; // no extension
}

static int pvs_search(const vector<vector<map<string,string>>>& board,
                      int depth, int ply, int alpha, int beta, const string& side,
                      Move& pv_best)
{
    G_NODES_SEARCHED++;
    if(ply > G_MAX_DEPTH_REACHED) G_MAX_DEPTH_REACHED = ply;
    
    // Hard ply limit to prevent runaway selective deepening
    const int MAX_PLY_LIMIT = 20;
    if(ply > MAX_PLY_LIMIT) {
        pv_best = {"",{},{},{}, ""};
        return evaluate_function(board, side);
    }
    
    if(time_up()) { pv_best={"",{},{},{}, ""}; return evaluate_function(board, side); }
    if(depth <= 0){
        pv_best = {"",{},{},{}, ""};
        return tactical_quiescence(board, side, alpha, beta);
    }
    if(is_terminal(board)){
        pv_best = {"",{},{},{}, ""};
        return evaluate_function(board, side);
    }

    // Symmetry-aware hash: use canonical mirror
    uint64_t h1 = simple_fullpos_hash(board, side);
    uint64_t h2 = simple_fullpos_hash_mirror(board, side);
    uint64_t key = std::min(h1, h2);
    Move tt_best{"",{},{},{}, ""};
    int ttv;
    if(tt_probe(key, depth, alpha, beta, ttv, tt_best)){
        pv_best = tt_best;
        return ttv;
    }

    // Generate
    vector<Move> moves;
    int R=board.size(), C=board[0].size();
    for(int i=0;i<R;i++) for(int j=0;j<C;j++){
        if(board[i][j].empty() || board[i][j].at("owner")!=side) continue;
        auto ms = generate_all_moves(board,i,j);
        moves.insert(moves.end(), ms.begin(), ms.end());
    }
    if(moves.empty()){ pv_best={"",{},{},{}, ""}; return evaluate_function(board, side); }

    order_moves(board, moves, tt_best.action==""?nullptr:&tt_best, ply, side);
    
    // Debug: Check if moves are being ordered (remove after testing)
    if(ply == 0 && moves.size() > 1) {
        // Verify first move has highest score
        int first_score = quick_order_score(board, moves[0]);
        int second_score = moves.size() > 1 ? quick_order_score(board, moves[1]) : -999999;
        if(second_score > first_score) {
            cout << "[WARNING] Move ordering failed at root! Scores: " << first_score << " vs " << second_score << endl;
        }
    }

    bool first=true;
    Move local_best{"",{},{},{}, ""};
    int bestVal = -1000000000;
    int origAlpha = alpha;

    for(size_t idx=0; idx<moves.size(); ++idx){
        if(time_up()) break; // Check time limit inside loop
        
        const Move &mv = moves[idx];
        auto nb = update_board(board, mv);

        // SELECTIVE DEEPENING: Check if this position needs extended search
        int extension = get_extension_depth(board, mv, side);
        
        // LMR (Late Move Reduction)
        int rdepth = depth - 1 + extension; // base depth + extension
        bool doReduce = (!first && depth>=3 && idx >= 6 && extension == 0); // don't reduce critical moves
        if(doReduce) rdepth = depth - 2; // reduce if not critical

        Move child_best;
        int score;
        const string nxt = (side=="circle")?"square":"circle";

        if(first){
            score = -pvs_search(nb, rdepth, ply+1, -beta, -alpha, nxt, child_best);
            first=false;
        }else{
            score = -pvs_search(nb, rdepth, ply+1, -alpha-1, -alpha, nxt, child_best);
            if(score > alpha && score < beta){
                score = -pvs_search(nb, rdepth, ply+1, -beta, -alpha, nxt, child_best);
            }
        }

        if(score > bestVal){
            bestVal = score;
            local_best = mv;
            if(score > alpha){
                alpha = score;
                if(alpha >= beta){
                    G_BETA_CUTOFFS++;
                    if(ply<MAX_PLY){
                        if(!(mv.action==killer1[ply].action && mv.from==killer1[ply].from &&
                             mv.to==killer1[ply].to && mv.pushed_to==killer1[ply].pushed_to)){
                            killer2[ply] = killer1[ply];
                            killer1[ply] = mv;
                        }
                        historyHeu[histKey(mv)] += depth*depth;
                    }
                    tt_store(key, depth, bestVal, TT_LOWER, local_best);
                    pv_best = local_best;
                    return bestVal;
                }
            }
        }
    }

    int flag = (bestVal <= origAlpha) ? TT_UPPER : (bestVal >= beta) ? TT_LOWER : TT_EXACT;
    tt_store(key, depth, bestVal, flag, local_best);
    pv_best = local_best;
    return bestVal;
}

static Move search_root(const vector<vector<map<string,string>>>& board,
                        const string& side, int time_ms, int max_depth_cap)
{
    G_TIMEOUT_HIT = false;
    G_DEADLINE = chrono::high_resolution_clock::now() +
                 chrono::milliseconds((int)(time_ms * 1.05));
    
    // Reset debug counters
    G_NODES_SEARCHED = 0;
    G_ALPHA_CUTOFFS = 0;
    G_BETA_CUTOFFS = 0;
    G_MAX_DEPTH_REACHED = 0;

    vector<Move> rootMoves;
    int R=board.size(), C=board[0].size();
    for(int i=0;i<R;i++) for(int j=0;j<C;j++){
        if(board[i][j].empty() || board[i][j].at("owner")!=side) continue;
        auto ms = generate_all_moves(board,i,j);
        rootMoves.insert(rootMoves.end(), ms.begin(), ms.end());
    }
    if(rootMoves.empty()) return {"move",{0,0},{0,0},{}, ""};
    order_moves(board, rootMoves, nullptr, 0, side);
    
    // Debug: Check move ordering quality at root
    if(rootMoves.size() >= 3) {
        vector<int> top_scores;
        for(size_t i = 0; i < min((size_t)5, rootMoves.size()); i++) {
            top_scores.push_back(quick_order_score(board, rootMoves[i]));
        }
        cout << "[MOVE ORDER] Top 5 scores: ";
        for(int s : top_scores) cout << s << " ";
        cout << "| Total moves: " << rootMoves.size() << endl;
    }

    int alpha = -1000000000, beta = 1000000000;
    int lastScore = 0;
    Move bestRoot = rootMoves[0];
    int completed_depth = 0;
    
    auto search_start_time = chrono::high_resolution_clock::now();

    for(int depth=1; depth<=max_depth_cap && !time_up(); ++depth){
        // Wider aspiration window to reduce re-searches (improves effective move ordering)
        int asp = (depth <= 2) ? 500 : 200;  // very wide for shallow depths
        int a = max(alpha, lastScore - asp);
        int b = min(beta,  lastScore + asp);

        Move depthBest = bestRoot; // try PV first
        int v = -1000000000;

        for(size_t i=0;i<rootMoves.size() && !time_up(); ++i){
            auto &mv = rootMoves[i];
            auto nb = update_board(board, mv);
            Move cb;
            int sc = -pvs_search(nb, depth-1, 1, -b, -a,
                                 (side=="circle")?"square":"circle", cb);
            if(sc <= a || sc >= b){
                sc = -pvs_search(nb, depth-1, 1, -beta, -alpha,
                                 (side=="circle")?"square":"circle", cb);
            }
            
            // Prefer earlier win / later loss (depth bias)
            int depth_bias = 2*depth;
            if(sc>0) sc += depth_bias;  // earlier positive is better
            else     sc -= depth_bias;  // delay negatives
            
            if(sc > v){
                v = sc;
                depthBest = mv;
                // move to front
                rotate(rootMoves.begin(), rootMoves.begin()+i, rootMoves.begin()+i+1);
                i = 0;
                a = max(a, v);
            }
        }
        if(depthBest.action!=""){ bestRoot = depthBest; lastScore = v; completed_depth = depth; }
    }
    
    // Calculate actual time used
    auto search_end_time = chrono::high_resolution_clock::now();
    double time_used_ms = chrono::duration<double, milli>(search_end_time - search_start_time).count();
    
    // Debug output
    double effective_branching = G_NODES_SEARCHED > 0 ? pow(G_NODES_SEARCHED, 1.0/max(1, completed_depth)) : 0;
    
    cout << "\n========================================" << endl;
    cout << "   SEARCH COMPLETED - DEPTH: " << completed_depth << endl;
    cout << "========================================" << endl;
    cout << "[SEARCH STATS] Depth: " << completed_depth 
         << " | Max Ply: " << G_MAX_DEPTH_REACHED
         << " | Nodes: " << G_NODES_SEARCHED 
         << " | Beta Cutoffs: " << G_BETA_CUTOFFS
         << " | Cutoff Rate: " << (G_NODES_SEARCHED > 0 ? (100.0 * G_BETA_CUTOFFS / G_NODES_SEARCHED) : 0) << "%"
         << " | Eff. Branching: " << fixed << setprecision(1) << effective_branching
         << " | Time: " << (int)time_used_ms << "ms/" << time_ms << "ms"
         << endl;
    cout << "========================================\n" << endl;

    return bestRoot;
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

    // Forbid pushes where the pusher would have to step into the opponent's scoring cell (the current pushed cell)
    if(is_in_score_area(or_push_row, or_push_col, pusher_owner)) return;

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

        string owner_to_be_pushed = board[nx][ny].at("owner");
        string pusher_owner = board[row][col].at("owner");

        if(!board[nx][ny].empty() && board[nx][ny].at("side")=="stone"){  
            // Skip invalid destination
            if(nx2 < 0 || nx2 >= rows || ny2 < 0 || ny2 >= cols)continue;
            // Do NOT push a piece through or into scoring areas
            if(is_in_score_area(nx2, ny2, owner_to_be_pushed))continue;
            // Engine also forbids pushing a piece into OUR opponent scoring cells
            // (i.e., the band we cannot enter). Mirror that to avoid proposing illegal pushes.
            if(is_in_score_area(nx2, ny2, pusher_owner)) continue;
            if(!board[nx2][ny2].empty())continue;
            
            if(is_in_score_area(nx, ny, pusher_owner)) {
                continue;
            }

            // If we're pushing our own stone that is currently in our scoring area,
            // forbid moves that would take it OUT of our scoring area.
            // But allow moves that keep it inside the scoring area.
            if (owner_to_be_pushed == pusher_owner &&
                is_own_score_cell_fixed(nx,  ny,  pusher_owner) &&   // currently in own SA
            !is_own_score_cell_fixed(nx2, ny2, pusher_owner)) {   // destination outside own SA
                continue;
            }

            //  stone push
            if(board[row][col].at("side") == "stone"){
                moves.push_back({"push", {row, col}, {nx, ny}, {nx2, ny2}, ""});
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
        // Suppress flipping scored stones; flipping here rarely helps and causes churn
        if(!is_own_score_cell_fixed(row, col, owner)){
            moves.push_back({"flip",{row,col},{row,col},{},"horizontal"});
            moves.push_back({"flip",{row,col},{row,col},{},"vertical"});
        }
    }
    else{
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

    explicit StudentAgent(string side) : side(std::move(side)), gen(rd()){}

    Move choose(const vector<vector<map<string, string>>>& board,int row,int col,const vector<int>& score_cols,float current_player_time,float opponent_time){
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
        double base_iter_time = 0.25; // increased from 0.10
        if(G_ROWS>=15) base_iter_time = 0.35; // increased from 0.14
        if(G_ROWS>=17) base_iter_time = 0.45; // increased from 0.18
        double time_limit = base_iter_time;
        if(current_player_time < 10) time_limit *= 0.7;  // less aggressive reduction
        else if(current_player_time < 30) time_limit *= 0.85;

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

        // Decay lane & piece cooldowns (per root call)
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
            if(has_wing_flip){
                for(auto &kv : moves_for_current){
                    int r = kv.first.first, c = kv.first.second;
                    if(!(c>=centerL && c<=centerR)) continue;
                    auto &vec = kv.second;
                    vec.erase(remove_if(vec.begin(), vec.end(), [&](const Move& m){ return m.action=="flip"; }), vec.end());
                }
            }

            // If we have any "empty-wing" flip available, restrict flips to only those columns (outer and low occupancy)
            if(has_empty_wing_flip){
                for(auto &kv : moves_for_current){
                    int r = kv.first.first, c = kv.first.second;
                    int occ = col_occupancy(c);
                    bool keep = (c < centerL || c > centerR) && (occ <= 1);
                    if(keep) continue;
                    auto &vec = kv.second;
                    vec.erase(remove_if(vec.begin(), vec.end(), [&](const Move& m){ return m.action=="flip"; }), vec.end());
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
                for (const auto &mv : vec) {
                    if (scored_delta_after(board, mv, side) > 0) { filtered.push_back(mv); continue; }
                    if (!would_repeat_recent(mv)) filtered.push_back(mv);
                }
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
        
    // ---- Greedy take-the-point: if any move increases scored stones, do it. ----
    {
        vector<Move> scoring_moves;
        scoring_moves.reserve(32);
        for (auto &kv : moves_for_current) {
            if (board[kv.first.first][kv.first.second].empty()) continue;
            if (board[kv.first.first][kv.first.second].at("owner") != side) continue;
            for (const auto &mv : kv.second) {
                if (scored_delta_after(board, mv, side) > 0) {
                    scoring_moves.push_back(mv);
                }
            }
        }
        if (!scoring_moves.empty()) {
            // Pick the safest/strongest of the scoring moves by static eval after the move.
            int best = INT_MIN;
            Move pick = scoring_moves[0];
            for (const auto &mv : scoring_moves) {
                auto nb = update_board(board, mv);
                int sc  = evaluate_function(nb, side);
                if (sc > best) { best = sc; pick = mv; }
            }
            return convert_move_format(pick);
        }
    }

    // ---- Defensive pushes: if we can knock back a near-scoring opponent stone, do it. ----
    {
        string opponent = (side == "circle") ? "square" : "circle";
        vector<Move> defensive_pushes;

        for (const auto &kv : moves_for_current) {
            int r = kv.first.first;
            int c = kv.first.second;
            if (board[r][c].empty() || board[r][c].at("owner") != side) continue;

            for (const auto &mv : kv.second) {
                if (mv.action != "push" || mv.to.size() < 2 || mv.pushed_to.size() < 2) continue;

                int tr = mv.to[0], tc = mv.to[1];      // stone being pushed
                if (board[tr][tc].empty() || board[tr][tc].at("owner") != opponent) continue;

                int pr = mv.pushed_to[0], pc = mv.pushed_to[1];

                int before = min_distance_from_scoring_area(opponent, tr, tc);
                int after  = min_distance_from_scoring_area(opponent, pr, pc);
                int delta  = after - before;

                // We care only about knocking back real threats
                if (before <= 3 && delta >= 2) {
                    defensive_pushes.push_back(mv);
                }
            }
        }

        if (!defensive_pushes.empty()) {
            // Pick best defensive push by static eval
            Move pick = defensive_pushes[0];
            int best = INT_MIN;
            for (const auto &mv : defensive_pushes) {
                auto nb = update_board(board, mv);
                int sc = evaluate_function(nb, side);
                if (sc > best) { best = sc; pick = mv; }
            }
            return convert_move_format(pick);
        }
    }
        
    // Deterministic symmetry-breaking in openings
    bool early_game = (position_history.size() <= 4);
    if(!early_game){
        // Check if anyone has scored
        // Change #4: use G_SCORE_COLS only
        bool anyone_scored = false;
        int circle_score_row = top_score_row_dynamic();
        int square_score_row = bottom_score_row_dynamic();
        for(int c: G_SCORE_COLS){
            if(!board[circle_score_row][c].empty() && board[circle_score_row][c].at("side")=="stone") anyone_scored=true;
            if(!board[square_score_row][c].empty() && board[square_score_row][c].at("side")=="stone") anyone_scored=true;
        }
        early_game = !anyone_scored;
    }
    
    if(early_game){
        // Deterministic tie-break: prefer left (smaller column indices)
        for(auto &kv : moves_for_current){
            if(kv.second.empty()) continue;
            stable_sort(kv.second.begin(), kv.second.end(), [&](const Move& a,const Move& b){
                int ac = a.from.empty()?1000:a.from[1];
                int bc = b.from.empty()?1000:b.from[1];
                // prefer smaller 'from' column; if equal, smaller 'to' column
                if(ac!=bc) return ac < bc;
                int at = a.to.empty()?1000:a.to[1];
                int bt = b.to.empty()?1000:b.to[1];
                return at < bt;
            });
        }
    }
        
    // ----- PVS root search (time-managed) -----
    size_t est_moves = 0; for(auto &p : moves_for_current) est_moves += p.second.size();
    int time_ms = (int)(time_limit * 1000.0);
    int max_depth_cap = 7;  // increased from 5 to allow deeper search
    if(G_ROWS>=15) max_depth_cap = 6;  // increased from 4
    if(G_ROWS>=17) max_depth_cap = 5;  // increased from 4
    if(est_moves > 250) max_depth_cap = min(max_depth_cap, 4);  // less aggressive reduction

    Move best_move = search_root(board, side, time_ms, max_depth_cap);
        
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
                        bool point_gain = (scored_delta_after(board, mv, side) > 0);
                        if (!point_gain) {
                            if(!mv.from.empty() && lane_cooldown.count(mv.from[1])) h -= 800;
                            if(!mv.to.empty()   && lane_cooldown.count(mv.to[1]))   h -= 500;
                            if(!mv.from.empty() && piece_cooldown.count(pos_key(mv.from[0], mv.from[1]))) h -= 400;
                        }
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
                    bool point_gain = (scored_delta_after(board, mv, side) > 0);
                    if (!point_gain) {
                        if(!mv.from.empty() && lane_cooldown.count(mv.from[1])) h -= 800;
                        if(!mv.to.empty()   && lane_cooldown.count(mv.to[1]))   h -= 500;
                        if(!mv.from.empty() && piece_cooldown.count(pos_key(mv.from[0], mv.from[1]))) h -= 400;
                    }
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

// ---------- Small helpers ----------
static inline bool inb(int r,int c,int R,int C){ return (r>=0 && r<R && c>=0 && c<C); }
static inline bool emptyC(const vector<vector<map<string,string>>>& B,int r,int c){ return B[r][c].empty(); }
static inline bool stoneC(const vector<vector<map<string,string>>>& B,int r,int c){ return !B[r][c].empty() && B[r][c].at("side")=="stone"; }
static inline bool riverC(const vector<vector<map<string,string>>>& B,int r,int c){ return !B[r][c].empty() && B[r][c].at("side")=="river"; }
static inline bool ownScore(const string& owner,int r,int c){ return is_own_score_cell_dyn(r,c,owner); }
static inline bool oppScore(const string& owner,int r,int c){ return is_opponent_score_cell_dyn(r,c,owner); }

static inline bool friendlyVert(const vector<vector<map<string,string>>>& B,int r,int c,const string& owner){
    return riverC(B,r,c) && B[r][c].at("orientation")=="vertical" && B[r][c].at("owner")==owner;
}
static inline bool friendlyHoriz(const vector<vector<map<string,string>>>& B,int r,int c,const string& owner){
    return riverC(B,r,c) && B[r][c].at("orientation")=="horizontal" && B[r][c].at("owner")==owner;
}

// Collect all landing squares reachable by ONE ride that starts by entering
// a friendly river cell (rr,cc). We allow chaining through *any* river cells
// (owner/orientation agnostic as in your generator) but we can only *enter*
// the system via a friendly river adjacent to our stone.
//
// Landings are:
//   - Any EMPTY cell along the axis
//   - Any OWN scoring cell
// We DO NOT land on river cells; they are transit only.
// We stop on stones or at board edge; we never pass through opp scoring cells.
static void collect_ride_landings_one_move(
    const vector<vector<map<string,string>>>& B, const string& owner,
    int rr, int cc,                      // we start by stepping *onto* this friendly river
    vector<pair<int,int>>& out,
    int node_cap = 48                    // safety cap on river nodes explored
){
    const int R = (int)B.size(), C = (int)B[0].size();
    if(!inb(rr,cc,R,C) || !riverC(B,rr,cc)) return;

    // We will explore a small frontier of "river nodes" that we can stand on during the ride,
    // and from each, scan rays along *its* orientation, pushing empties as landings and
    // queuing new river nodes when encountered (chaining).
    deque<pair<int,int>> q;
    vector<vector<char>> seen(R, vector<char>(C, 0));

    auto push_river = [&](int r,int c){
        if(!inb(r,c,R,C) || !riverC(B,r,c)) return;
        if(seen[r][c]) return;
        seen[r][c]=1; q.push_back({r,c});
    };

    push_river(rr,cc);
    int popped=0;

    while(!q.empty() && popped < node_cap){
        auto [r,c] = q.front(); q.pop_front(); ++popped;
        const bool vert = (B[r][c].at("orientation")=="vertical");
        const int dr[2] = { vert?+1:0, vert?-1:0 };
        const int dc[2] = { vert?0:+1, vert?0:-1 };

        for(int arm=0; arm<2; ++arm){
            int nr = r + dr[arm], nc = c + dc[arm];
            int steps=0;
            while(inb(nr,nc,R,C) && steps<64){
                if(oppScore(owner,nr,nc)) break;       // cannot pass through opponent scoring
                if(stoneC(B,nr,nc)) break;             // blocked

                if(emptyC(B,nr,nc) || ownScore(owner,nr,nc)){
                    out.push_back({nr,nc});            // valid landing
                    // continue scanning; you can ride further in same axis
                    nr += dr[arm]; nc += dc[arm]; ++steps;
                    continue;
                }

                // It's a river: you cannot *land* here, but you can chain.
                if(riverC(B,nr,nc)){
                    push_river(nr,nc);                 // add this river node to frontier
                    break;                             // stop this ray; queued node will expand
                }

                break;
            }
        }
    }

    // Deduplicate landings
    sort(out.begin(), out.end());
    out.erase(unique(out.begin(), out.end()), out.end());
}

// Returns how many stones in OUR scoring area after mv minus before.
static inline int scored_delta_after(
    const vector<vector<map<string,string>>>& B,
    const Move& mv,
    const string& side
){
    int before = count_in_score_area(B, side);
    auto nb = update_board(B, mv);
    int after  = count_in_score_area(nb, side);
    return after - before;
}


// Return minimal plies (<= ply_cap) for this piece to enter its OWN scoring cell.
// 999 means "not reachable within cap".
static int min_plies_to_score_piece(
    const vector<vector<map<string,string>>>& B,
    const string& owner, int sr, int sc,
    int ply_cap = 5, int exp_node_cap = 200
){
    const int R=(int)B.size(), C=(int)B[0].size();
    if(!inb(sr,sc,R,C) || B[sr][sc].empty()) return 999;

    int start_cost = 0;
    if(riverC(B,sr,sc)){
        if(B[sr][sc].at("owner")!=owner) return 999; // won’t consider flipping enemy piece
        start_cost = 1; // flip to stone first
        if(start_cost > ply_cap) return 999;
    }

    // Dijkstra with uniform edge costs => BFS on layers by "plies"
    struct Node{int r,c,cost;};
    deque<Node> dq;
    vector<vector<int>> best(R, vector<int>(C, INT_MAX));

    auto push = [&](int r,int c,int cost){
        if(!inb(r,c,R,C)) return;
        if(cost>ply_cap) return;
        if(cost>=best[r][c]) return;
        best[r][c]=cost; dq.push_back({r,c,cost});
    };

    // Start at our cell as a stone (after optional flip)
    push(sr,sc,start_cost);

    const int d4r[4]={+1,-1,0,0}, d4c[4]={0,0,+1,-1};
    int expansions=0;

    while(!dq.empty() && expansions<exp_node_cap){
        auto [r,c,cost] = dq.front(); dq.pop_front(); ++expansions;

        // Success if this square is an own scoring cell and can be occupied by a stone.
        if(ownScore(owner,r,c) && !riverC(B,r,c)) return cost;

        // 1) Plain orthogonal step (+1) into empty (cannot step into opponent scoring)
        for(int k=0;k<4;k++){
            int nr=r+d4r[k], nc=c+d4c[k];
            if(!inb(nr,nc,R,C)) continue;
            if(oppScore(owner,nr,nc)) continue;
            if(emptyC(B,nr,nc)) push(nr,nc,cost+1);
        }

        // 2) River rides (+1): if any adjacent cell is a river (ours *or* theirs),
        //    collect every reachable landing in ONE move, including river chaining.
        vector<pair<int,int>> adj_rivers;
        for(int k=0;k<4;k++){
            int nr = r + d4r[k], nc = c + d4c[k];
            if(!inb(nr,nc,R,C)) continue;
            if(riverC(B,nr,nc)) adj_rivers.push_back({nr,nc});
        }
        if(!adj_rivers.empty()){
            // For each adjacent river, run a bounded river BFS to get all landings
            for(auto [rr,cc] : adj_rivers){
                vector<pair<int,int>> landings;
                collect_ride_landings_one_move(B, owner, rr, cc, landings, /*node_cap*/48);
                for(auto [lr,lc] : landings){
                    if(!oppScore(owner,lr,lc)) push(lr,lc,cost+1);
                }
            }
        }
    }
    return 999;
}


int evaluate_function(const vector<vector<map<string,string>>>& B, const string& owner){
    const string opp = (owner=="circle") ? "square" : "circle";
    const int R=(int)B.size(), C=(int)B[0].size();

    // Stones already scored (win pressure)
    auto sc = compute_n_self_opp(B, owner);
    int n_self = sc.first, n_opp = sc.second;

    // Dynamic scoring band bounds
    int sc_min = 0, sc_max = C-1;
    if(!G_SCORE_COLS.empty()){
        sc_min = *min_element(G_SCORE_COLS.begin(), G_SCORE_COLS.end());
        sc_max = *max_element(G_SCORE_COLS.begin(), G_SCORE_COLS.end());
    } else {
        int w = (C<=12)?4:((C<=14)?5:6);
        int s = max(0, C/2 - w/2); sc_min = s; sc_max = min(C-1, s+w-1);
    }
    int centerL = max(0, sc_min - 1);
    int centerR = min(C-1, sc_max + 1);

    // Column blockers (stones only) along path to the goal row
    auto blockers = [&](const string& who,int col){
        int start = (who=="circle")? bottom_score_row_dynamic() : top_score_row_dynamic();
        int goal  = (who=="circle")? top_score_row_dynamic()    : bottom_score_row_dynamic();
        int step  = (who=="circle")? -1 : +1;
        int cnt=0;
        for(int r=start; r!=goal+step; r+=step){
            if(!inb(r,col,R,C)) continue;
            if(stoneC(B,r,col) && B[r][col].at("owner")!=who) cnt++;
        }
        return cnt;
    };

    // River-adjacency count (friendly vertical or horizontal)
    auto adj_friend_river = [&](const string& who,int r,int c){
        static const int d4r[4]={+1,-1,0,0}, d4c[4]={0,0,+1,-1};
        for(int k=0;k<4;k++){
            int nr=r+d4r[k], nc=c+d4c[k];
            if(!inb(nr,nc,R,C)) continue;
            if((riverC(B,nr,nc) && B[nr][nc].at("owner")==who)) return true;
        }
        return false;
    };

    // Multi-ply reachability signals
    int self_min=999, opp_min=999;
    int self_le1=0, self_le2=0, self_le3=0, self_le4=0;
    int opp_le1=0,  opp_le2=0,  opp_le3=0,  opp_le4=0;
    int self_radj=0, opp_radj=0;

    // Center congestion
    int center_occ=0, center_cells = R * max(0, centerR-centerL+1);

    for(int r=0;r<R;r++){
        for(int c=0;c<C;c++){
            if(B[r][c].empty()) continue;
            if(stoneC(B,r,c)){
                const string who = B[r][c].at("owner");
                if(c>=centerL && c<=centerR) center_occ++;

                int cost = min_plies_to_score_piece(B, who, r, c, /*ply_cap*/4, /*exp_node_cap*/150);
                if(who==owner){
                    self_min = min(self_min, cost);
                    if(cost<=1) self_le1++;
                    if(cost<=2) self_le2++;
                    if(cost<=3) self_le3++;
                    if(cost<=4) self_le4++;
                    if(adj_friend_river(owner,r,c)) self_radj++;
                }else{
                    opp_min = min(opp_min, cost);
                    if(cost<=1) opp_le1++;
                    if(cost<=2) opp_le2++;
                    if(cost<=3) opp_le3++;
                    if(cost<=4) opp_le4++;
                    if(adj_friend_river(opp,r,c)) opp_radj++;
                }
            }
        }
    }

    // Choose best wing vs center clarity
    int best_wing=1e9,best_center=1e9;
    for(int c=0;c<C;c++){
        int b = blockers(owner,c);
        if(c<centerL || c>centerR) best_wing   = min(best_wing,b);
        else                       best_center = min(best_center,b);
    }

    // Center density penalty
    double density = center_cells>0 ? (double)center_occ/(double)center_cells : 0.0;
    double dens_thr = (G_COLS<=12?0.45:(G_COLS<=14?0.48:0.52));
    int corridor_pen = 0;
    if(density > dens_thr){
        double excess = density - dens_thr;
        double scale  = (G_COLS<=12?3200.0:(G_COLS<=14?2900.0:2600.0));
        corridor_pen  = (int)(excess*scale);
    }

    // 1) DUAL-THREAT PRESSURE: Count distinct columns where stones can score within K plies
    auto columns_can_score_in = [&](const string& who, int ply_cap){
        unordered_set<int> cols;
        for(int r=0;r<R;r++) for(int c=0;c<C;c++){
            if(stoneC(B,r,c) && B[r][c].at("owner")==who){
                int t = min_plies_to_score_piece(B, who, r, c, ply_cap, /*exp_node_cap*/150);
                if(t<=ply_cap){
                    // If any scoring cell in this column is ours, mark the column.
                    for(int sc : (G_SCORE_COLS.empty() ? vector<int>{max(0,C/2-2),max(0,C/2-1),min(C-1,C/2),min(C-1,C/2+1)} : G_SCORE_COLS)){
                        if(c==sc || std::abs(c-sc)<=0) { cols.insert(c); break; }
                    }
                }
            }
        }
        return (int)cols.size();
    };

    int self_cols_le2 = columns_can_score_in(owner, 2);
    int opp_cols_le2  = columns_can_score_in(opp,    2);
    int self_cols_le3 = columns_can_score_in(owner, 3);
    int opp_cols_le3  = columns_can_score_in(opp,   3);

    // 2) RIVER NETWORK BRIDGE SCORE: Connected river components touching back half and near goal
    auto river_components_bridge_score = [&](const string& who){
        vector<vector<int>> vis(R, vector<int>(C,0));
        int goal = (who=="circle")? top_score_row_dynamic() : bottom_score_row_dynamic();
        int back_min = (who=="circle")? (R/2) : 0;
        int back_max = (who=="circle")? (R-1) : (R/2);
        int score=0;
        for(int r=0;r<R;r++) for(int c=0;c<C;c++){
            if(vis[r][c]||!riverC(B,r,c)||B[r][c].at("owner")!=who) continue;
            deque<pair<int,int>> q; q.push_back({r,c}); vis[r][c]=1;
            bool touches_back=false, near_goal=false, has_vertical=false;
            while(!q.empty()){
                auto [x,y]=q.front(); q.pop_front();
                if(x>=min(back_min,back_max) && x<=max(back_min,back_max)) touches_back=true;
                if(abs(x-goal)<=2) near_goal=true;
                if(B[x][y].at("orientation")=="vertical") has_vertical=true;
                static int d4[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
                for(auto &d: d4){
                    int nx=x+d[0], ny=y+d[1];
                    if(inb(nx,ny,R,C)&&!vis[nx][ny]&&riverC(B,nx,ny)&&B[nx][ny].at("owner")==who){
                        vis[nx][ny]=1; q.push_back({nx,ny});
                    }
                }
            }
            if(touches_back && near_goal) score += has_vertical? 9 : 5;
        }
        return score;
    };
    int net_self = river_components_bridge_score(owner);
    int net_opp  = river_components_bridge_score(opp);

    // 3) COLUMN RUNWAY: Longest straight sequence of empty/river cells toward goal
    auto column_runway = [&](const string& who, int col){
        int start = (who=="circle")? bottom_score_row_dynamic() : top_score_row_dynamic();
        int goal  = (who=="circle")? top_score_row_dynamic()    : bottom_score_row_dynamic();
        int step  = (who=="circle")? -1 : +1;
        int best=0, cur=0;
        for(int r=start; r!=goal+step; r+=step){
            if(!inb(r,col,R,C)) break;
            if(emptyC(B,r,col) || riverC(B,r,col)) { cur++; best=max(best,cur); }
            else cur=0;
        }
        return best;
    };
    int best_self_run=0, best_opp_run=0;
    for(int c=0;c<C;c++){
        best_self_run=max(best_self_run, column_runway(owner,c));
        best_opp_run =max(best_opp_run,  column_runway(opp,c));
    }

    // 4) ANTI-DISLODGE: Penalty if scored stones can be pushed out by opponent
    auto can_opp_dislodge_scored = [&](){
        int r_goal = (owner=="circle")? top_score_row_dynamic() : bottom_score_row_dynamic();
        vector<int> cols = G_SCORE_COLS.empty()? vector<int>{max(0,C/2-2),max(0,C/2-1),min(C-1,C/2),min(C-1,C/2+1)} : G_SCORE_COLS;
        int risk=0;
        for(int c: cols){
            if(inb(r_goal,c,R,C) && stoneC(B,r_goal,c) && B[r_goal][c].at("owner")==owner){
                static int d4[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
                for(auto& d: d4){
                    int pr=r_goal-d[0], pc=c-d[1];
                    int ep=r_goal+d[0], ec=c+d[1];
                    if(inb(pr,pc,R,C) && inb(ep,ec,R,C) && stoneC(B,pr,pc) && B[pr][pc].at("owner")==opp && emptyC(B,ep,ec)){
                        risk++;
                        break;
                    }
                }
            }
        }
        return risk;
    };
    int dislodge_risk = can_opp_dislodge_scored();

    // ---------------- Weights (tuned to be smooth and search-friendly) ----------------
    const int W_SCORE    = 2200;

    const int W_LE1      = 1800;
    const int W_LE2      = 1200;
    const int W_LE3      = 560;
    const int W_LE4      = 200;

    const int W_MINPLY   = 320;    // bonus if our best piece needs fewer plies than theirs
    const int W_RADJ     = 60;     // small support signal
    const int W_WING     = 180;    // prefer wings if clearly cleaner
    const int W_CORRIDOR = 1;
    
    // New weights for advanced features
    const int W_DUAL2    = 1500;   // dual-threat pressure at ply 2
    const int W_DUAL3    = 700;    // dual-threat pressure at ply 3
    const int W_NETWORK  = 80;     // river network bridge score
    const int W_RUNWAY   = 35;     // column runway advantage
    const int W_DISLODGE = 900;    // anti-dislodge penalty

    long long eval = 0;

    // Scored stones
    eval += W_SCORE * (n_self - n_opp);

    // Multi-ply pressure (ours minus theirs)
    eval += W_LE1 * (self_le1 - opp_le1);
    eval += W_LE2 * (self_le2 - opp_le2);
    eval += W_LE3 * (self_le3 - opp_le3);
    eval += W_LE4 * (self_le4 - opp_le4);

    // Best single-piece time-to-score differential
    {
        int s = (self_min>=999? 6 : self_min);
        int o = (opp_min >=999? 6 : opp_min);
        eval += W_MINPLY * (o - s); // fewer plies for us → positive
    }

    // Friendly river adjacency
    eval += W_RADJ * (self_radj - opp_radj);

    // Wing preference if wings are cleaner by ≥2 blockers
    if(best_wing + 2 <= best_center) eval += W_WING;

    // Corridor congestion penalty
    eval -= W_CORRIDOR * corridor_pen;

    // Dual-threat/tri-threat pressure (bigger for earlier k)
    eval += W_DUAL2 * ( (self_cols_le2>=2) - (opp_cols_le2>=2) );
    eval += W_DUAL3 * ( (self_cols_le3>=2) - (opp_cols_le3>=2) );

    // River network advantage
    eval += W_NETWORK * (net_self - net_opp);

    // Column runway advantage
    eval += W_RUNWAY * (best_self_run - best_opp_run);

    // Anti-dislodge penalty
    eval -= W_DISLODGE * dislodge_risk;

    // Score convexity & urgency (finish faster)
    if((int)G_SCORE_COLS.size()>=4){
        if(n_self==3) eval += 1800;
        if(n_opp==3)  eval -= 1800;
    }

    // Initiative tilt: earlier plies to score matter more
    auto init_curve = [&](int t){
        if(t>=999) return 0;
        if(t<=1) return 220;
        if(t==2) return 140;
        if(t==3) return 70;
        if(t==4) return 30;
        return 0;
    };
    eval += init_curve(opp_min) - init_curve(self_min);

    // Clamp
    if(eval > 2000000000LL) eval = 2000000000LL;
    if(eval < -2000000000LL) eval = -2000000000LL;
    return (int)eval;
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
