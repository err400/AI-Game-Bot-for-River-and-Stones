// Clean student_agent.cpp with MCTS core and essential helpers only
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <set>
#include <array>
#include <chrono>
#include <algorithm>
#include <queue>
#include <cmath>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
using namespace std;

namespace py = pybind11;

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------
enum Owner { OWN_NONE=0, OWN_CIRCLE=1, OWN_SQUARE=2 };
enum Side  { SIDE_EMPTY=0, SIDE_STONE=1, SIDE_RIVER=2 };
enum Ori   { ORI_NONE=0, ORI_HORIZONTAL=1, ORI_VERTICAL=2 };

struct Cell {
    Owner owner = OWN_NONE;
    Side side = SIDE_EMPTY;
    Ori orientation = ORI_NONE;
};
using Board = vector<vector<Cell>>;

// Value semantics for Cell to allow vector comparisons (used in Board equality checks)
static inline bool operator==(const Cell& a, const Cell& b){
    return a.owner==b.owner && a.side==b.side && a.orientation==b.orientation;
}
static inline bool operator!=(const Cell& a, const Cell& b){
    return !(a==b);
}

static inline string owner_to_string(Owner o){ if(o==OWN_CIRCLE) return "circle"; if(o==OWN_SQUARE) return "square"; return ""; }
static inline Owner string_to_owner(const string& s){ if(s=="circle") return OWN_CIRCLE; if(s=="square") return OWN_SQUARE; return OWN_NONE; }
static inline string side_to_string(Side s){ if(s==SIDE_STONE) return "stone"; if(s==SIDE_RIVER) return "river"; return ""; }
static inline Side string_to_side(const string& s){ if(s=="stone") return SIDE_STONE; if(s=="river") return SIDE_RIVER; return SIDE_EMPTY; }
static inline string ori_to_string(Ori o){ if(o==ORI_HORIZONTAL) return "horizontal"; if(o==ORI_VERTICAL) return "vertical"; return ""; }
static inline Ori string_to_ori(const string& s){ if(s=="horizontal") return ORI_HORIZONTAL; if(s=="vertical") return ORI_VERTICAL; return ORI_NONE; }

// Convert legacy board (map<string,string>) to compact Board once
static Board convert_board(const vector<vector<map<string,string>>>& legacy){
    Board b; b.resize(legacy.size());
    for(size_t i=0;i<legacy.size();++i){
        b[i].resize(legacy[i].size());
        for(size_t j=0;j<legacy[i].size();++j){
            const auto &cell = legacy[i][j];
            if(cell.empty()){ b[i][j] = Cell{}; continue; }
            Owner o = string_to_owner(cell.at("owner"));
            Side s  = string_to_side(cell.at("side"));
            Ori  orv = ORI_NONE;
            if(s==SIDE_RIVER){ auto it = cell.find("orientation"); if(it!=cell.end()) orv = string_to_ori(it->second); }
            b[i][j] = Cell{o,s,orv};
        }
    }
    return b;
}

struct Move {
    string action;             // "move" | "push" | "flip" | "rotate"
    vector<int> from;          // [row,col]
    vector<int> to;            // [row,col]
    vector<int> pushed_to;     // [row,col] for push
    string orientation;        // for flip/rotate
};

// ------------------------------------------------------------
// Global dynamic scoring context (set in choose())
// ------------------------------------------------------------
static int G_ROWS = 13;
static int G_COLS = 12;
static vector<int> G_SCORE_COLS;

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

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------
static bool allows_flow_into_opponent_score_after_orientation(
    const Board& board,
    int r, int c,
    Owner owner,
    Ori orientation)
{
    int n = (int)board.size();
    int m = (int)board[0].size();
    auto inb = [&](int x,int y){ return x>=0 && x<n && y>=0 && y<m; };

    queue<pair<int,int>> q;
    set<pair<int,int>> vis;
    q.push({r,c}); vis.insert({r,c});

    while(!q.empty()){
        auto [x,y] = q.front(); q.pop();
        Ori ori;
        if (x==r && y==c) ori = orientation; else {
            if (board[x][y].side!=SIDE_RIVER) continue;
            ori = board[x][y].orientation;
        }
        vector<pair<int,int>> dirs = (ori==ORI_HORIZONTAL) ? vector<pair<int,int>>{{0,1},{0,-1}} : vector<pair<int,int>>{{1,0},{-1,0}};
        for (auto [dx,dy] : dirs){
            int nx = x + dx, ny = y + dy;
            while(inb(nx,ny)){
                if (board[nx][ny].side==SIDE_EMPTY){
                    if (is_opponent_score_cell_dyn(nx, ny, owner_to_string(owner))) return true;
                } else if (board[nx][ny].side==SIDE_RIVER){
                    if (!vis.count({nx,ny})){
                        vis.insert({nx,ny}); q.push({nx,ny});
                    }
                    break;
                } else break;
                nx += dx; ny += dy;
            }
        }
    }
    return false;
}

// Evaluate how unsafe a flip/rotate is by checking whether it opens flow into opponent's scoring area
static inline double unsafe_flow_penalty_after_orientation(
    const Board& board,
    int r, int c,
    Owner owner,
    Ori orientation)
{
    if(allows_flow_into_opponent_score_after_orientation(board, r, c, owner, orientation)){
        // Strong penalty for enabling immediate opponent scoring flow
        return -0.8;
    }
    return 0.0;
}

// Explore river for moves (stone or river rides)
static void explore_river_move(const Board& board,int x, int y, vector<Move>& moves,
                               int rows, int cols, int from_row, int from_col, set<pair<int,int>>& visited){
    if (visited.count({x, y})) return;
    visited.insert({x, y});
    if (board[x][y].side != SIDE_RIVER) return;
    Ori orientation = board[x][y].orientation;
    vector<pair<int,int>> directions = (orientation == ORI_HORIZONTAL) ? vector<pair<int,int>>{{0,1},{0,-1}}
                                                                      : vector<pair<int,int>>{{1,0},{-1,0}};
    Owner owner = board[from_row][from_col].owner;
    for(auto [dx, dy] : directions){
        int nx = x + dx, ny = y + dy;
        while (nx >= 0 && nx < rows && ny >= 0 && ny < cols){
            if (nx == from_row && ny == from_col) { nx += dx; ny += dy; continue; }
            if (board[nx][ny].side==SIDE_EMPTY){
                if (!is_opponent_score_cell_dyn(nx, ny, owner_to_string(owner))) moves.push_back({"move", {from_row, from_col}, {nx, ny}, {}, ""});
            } else if (board[nx][ny].side == SIDE_RIVER){
                explore_river_move(board, nx, ny, moves, rows, cols, from_row, from_col, visited);
                break;
            } else break;
            nx += dx; ny += dy;
        }
    }
}

// Explore river for pushes
static void explore_river_push(const Board& board,int push_row, int push_col, vector<Move>& moves, int rows, int cols,int from_row, int from_col,
                               set<pair<int,int>>& visited,int or_push_row,int or_push_col, int or_from_row, int or_from_col, int depth=0){
    // Add depth limit to prevent stack overflow
    if(depth > 50) return;
    if (visited.count({push_row, push_col})) return;
    visited.insert({push_row, push_col});
    if (board[from_row][from_col].side != SIDE_RIVER) return;
    Ori orientation = board[from_row][from_col].orientation;
    vector<pair<int,int>> directions = (orientation == ORI_HORIZONTAL) ? vector<pair<int,int>>{{0,1},{0,-1}}
                                                                      : vector<pair<int,int>>{{1,0},{-1,0}};
    Owner owner_to_be_pushed = board[or_push_row][or_push_col].owner;
    for(auto [dx, dy] : directions){
        int nx = push_row + dx, ny = push_col + dy;
        while (nx >= 0 && nx < rows && ny >= 0 && ny < cols) {
            if(nx == from_row && ny == from_col){ nx += dx; ny += dy; continue; }
            if(board[nx][ny].side==SIDE_EMPTY){
                if(!is_opponent_score_cell_dyn(nx, ny, owner_to_string(owner_to_be_pushed))) moves.push_back({"push",{or_from_row, or_from_col},{or_push_row, or_push_col},{nx, ny},""});
                else break;
            } else if(board[nx][ny].side == SIDE_RIVER){
                // FIXED: Maintain original from_row/from_col for the river, update push_row/push_col
                explore_river_push(board, nx, ny, moves, rows, cols, nx, ny, visited,or_push_row,or_push_col, or_from_row, or_from_col, depth+1);
                break;
            } else break;
            nx += dx; ny += dy;
        }
    }
}

// Generate all moves for a piece at (row,col)
static vector<Move> generate_all_moves(const Board& board,int row,int col){
    vector<Move> moves;
    int rows = (int)board.size();
    int cols = (int)board[0].size();
    if(board[row][col].side==SIDE_EMPTY) return moves;
    Owner owner = board[row][col].owner;
    Side side  = board[row][col].side;
    Ori  orientation = (side==SIDE_RIVER) ? board[row][col].orientation : ORI_NONE;
    vector<pair<int,int>> dirs={{1,0},{-1,0},{0,1},{0,-1}};
    set<pair<int,int>> visited;
    // MOVE
    for(auto [dx,dy]: dirs){
        int nx = row + dx, ny = col + dy;
        if (nx < 0 || nx >= rows || ny < 0 || ny >= cols) continue;
        if(is_opponent_score_cell_dyn(nx,ny, owner_to_string(owner))) continue;
        if(board[nx][ny].side==SIDE_EMPTY) moves.push_back({"move", {row,col}, {nx,ny}, {}, ""});
        else if(board[nx][ny].side==SIDE_RIVER) explore_river_move(board, nx, ny, moves, rows, cols, row, col, visited);
    }
    // PUSH
    for(auto [dx,dy]: dirs){
        int nx=row+dx, ny=col+dy, nx2=row+2*dx, ny2=col+2*dy;
        if(nx<0||nx>=rows||ny<0||ny>=cols) continue;
        if(is_opponent_score_cell_dyn(nx,ny, owner_to_string(owner))) continue;
        if(board[nx][ny].side==SIDE_EMPTY || board[nx][ny].side!=SIDE_STONE) continue;
        if(nx2<0||nx2>=rows||ny2<0||ny2>=cols) continue;
        if(is_opponent_score_cell_dyn(nx2,ny2, owner_to_string(board[nx][ny].owner))) continue;
        if(board[nx2][ny2].side!=SIDE_EMPTY) continue;
        if(side==SIDE_STONE) moves.push_back({"push", {row,col}, {nx,ny}, {nx2,ny2}, ""});
        else if(side==SIDE_RIVER){
            set<pair<int,int>> visited_push; explore_river_push(board,nx,ny,moves,rows,cols,row,col,visited_push,nx,ny,row,col);
        }
    }
    // FLIP
    if(side==SIDE_STONE){
        // Only flip stone to river if not already a river and if flips would be safe
        if(!allows_flow_into_opponent_score_after_orientation(board,row,col,owner,ORI_HORIZONTAL)) 
            moves.push_back({"flip",{row,col},{row,col},{},"horizontal"});
        if(!allows_flow_into_opponent_score_after_orientation(board,row,col,owner,ORI_VERTICAL)) 
            moves.push_back({"flip",{row,col},{row,col},{},"vertical"});
    } else if(side==SIDE_RIVER){
        // Flip river back to stone (always allowed, don't add duplicate)
        moves.push_back({"flip",{row,col},{row,col},{},""});
    }
    // ROTATE
    if(side==SIDE_RIVER){
        Ori new_ori = (orientation==ORI_HORIZONTAL)?ORI_VERTICAL:ORI_HORIZONTAL;
        // Only rotate if it changes orientation and doesn't flow into opponent score
        if(!allows_flow_into_opponent_score_after_orientation(board,row,col,owner,new_ori))
            moves.push_back({"rotate",{row,col},{row,col},{}, ori_to_string(new_ori)});
    }
    
    // Deduplicate moves (in case of rare duplicates from river exploration)
    // Use simple comparison to remove exact duplicates
    sort(moves.begin(), moves.end(), [](const Move& a, const Move& b){
        if(a.action != b.action) return a.action < b.action;
        if(a.from != b.from) return a.from < b.from;
        if(a.to != b.to) return a.to < b.to;
        if(a.pushed_to != b.pushed_to) return a.pushed_to < b.pushed_to;
        return a.orientation < b.orientation;
    });
    moves.erase(unique(moves.begin(), moves.end(), [](const Move& a, const Move& b){
        return a.action == b.action && a.from == b.from && a.to == b.to && 
               a.pushed_to == b.pushed_to && a.orientation == b.orientation;
    }), moves.end());
    
    return moves;
}

// Apply a move to return a new board
static Board update_board(const Board& board, const Move& move){
    Board nb = board;
    // Validate from position
    if(move.from.size()<2) return nb;
    int sy = move.from[0], sx = move.from[1];
    if(sy<0||sy>=(int)nb.size()||sx<0||sx>=(int)nb[0].size()) return nb;
    if(nb[sy][sx].side==SIDE_EMPTY) return nb;
    
    if(move.action=="move"){
        // Validate to position
        if(move.to.size()<2) return nb;
        int ey = move.to[0], ex = move.to[1];
        if(ey<0||ey>=(int)nb.size()||ex<0||ex>=(int)nb[0].size()) return nb;
        nb[ey][ex] = nb[sy][sx]; nb[sy][sx] = Cell{};
    } else if(move.action=="push"){
        // Validate all required positions for push
        if(move.to.size()<2 || move.pushed_to.size()<2) return nb;
        int dy = move.pushed_to[0], dx = move.pushed_to[1];
        int ny = move.to[0], nx = move.to[1];
        if(dy<0||dy>=(int)nb.size()||dx<0||dx>=(int)nb[0].size()) return nb;
        if(ny<0||ny>=(int)nb.size()||nx<0||nx>=(int)nb[0].size()) return nb;
        
        Side pusher_side = nb[sy][sx].side;
        nb[dy][dx] = nb[ny][nx]; nb[ny][nx] = nb[sy][sx];
        if(pusher_side==SIDE_RIVER){ nb[ny][nx].side = SIDE_STONE; nb[ny][nx].orientation = ORI_NONE; }
        nb[sy][sx] = Cell{};
    } else if(move.action=="flip"){
        if(nb[sy][sx].side==SIDE_STONE){ nb[sy][sx].side = SIDE_RIVER; nb[sy][sx].orientation = string_to_ori(move.orientation); }
        else if(nb[sy][sx].side==SIDE_RIVER){ nb[sy][sx].side = SIDE_STONE; nb[sy][sx].orientation = ORI_NONE; }
    } else if(move.action=="rotate"){
        if(nb[sy][sx].side==SIDE_RIVER){
            if(!move.orientation.empty()) nb[sy][sx].orientation = string_to_ori(move.orientation);
            else nb[sy][sx].orientation = (nb[sy][sx].orientation==ORI_HORIZONTAL?ORI_VERTICAL:ORI_HORIZONTAL);
        }
    }
    return nb;
}

// Count stones in own score area
static int count_in_score_area(const Board& board,string owner){
    int r = (owner=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
    int cnt = 0; for(int c : G_SCORE_COLS){ if(r>=0 && r<G_ROWS && c>=0 && c<G_COLS){ const auto& cell = board[r][c]; if(cell.side==SIDE_STONE && owner_to_string(cell.owner)==owner) cnt++; } }
    return cnt;
}

// Terminal test
static bool is_terminal(const Board& board){
    int need = (G_COLS<=12)?4:((G_COLS<=14)?5:6);
    return count_in_score_area(board,"circle")>=need || count_in_score_area(board,"square")>=need;
}

// (Old evaluate_function removed; evaluation now integrated into leaf_value using quick_state_score)

// ---------- Quick heuristics for priors ----------
// Manhattan distance to own scoring area (row distance + nearest scoring column distance)
static inline int mdist_to_own_score(const string& owner, int r, int c){
    int target_row = (owner=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
    int row_d = abs(r - target_row);
    int col_d = 1000000;
    for(int sc : G_SCORE_COLS){ col_d = min(col_d, abs(c - sc)); }
    return row_d + (col_d>=1000000?0:col_d);
}

// Count number of river cells owned by a player
static int count_owned_rivers(const Board& board, const string& owner){
    int R = (int)board.size(); if(R==0) return 0; int C = (int)board[0].size(); int cnt=0;
    for(int i=0;i<R;i++) for(int j=0;j<C;j++) if(board[i][j].side==SIDE_RIVER && owner_to_string(board[i][j].owner)==owner) cnt++;
    return cnt;
}

// Compute potential run span (inclusive) starting from (r,c) if a stone were flipped to given orientation
static int potential_run_span_if_flipped(const Board& board, int r, int c, const string& orientation){
    int R=(int)board.size(); if(R==0) return 0; int C=(int)board[0].size();
    auto inb=[&](int x,int y){return x>=0&&x<R&&y>=0&&y<C;};
    vector<pair<int,int>> dirs = (orientation=="horizontal")?vector<pair<int,int>>{{0,1},{0,-1}}:vector<pair<int,int>>{{1,0},{-1,0}};
    int span=1; // count self
    for(auto [dx,dy]:dirs){
        int nx=r+dx, ny=c+dy;
        while(inb(nx,ny)){
            if(board[nx][ny].side==SIDE_EMPTY) { span++; nx+=dx; ny+=dy; }
            else if(board[nx][ny].side==SIDE_RIVER) { span++; nx+=dx; ny+=dy; }
            else break;
        }
    }
    return span;
}

// Whether flipping could create a direct ride to own scoring row (rough approximation)
static bool flip_opens_scoring_path(const Board& board, int r, int c, const string& owner, const string& orientation){
    int target_row = (owner=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
    int R=(int)board.size(); if(R==0) return false; int C=(int)board[0].size();
    auto inb=[&](int x,int y){return x>=0&&x<R&&y>=0&&y<C;};
    vector<pair<int,int>> dirs = (orientation=="horizontal")?vector<pair<int,int>>{{0,1},{0,-1}}:vector<pair<int,int>>{{1,0},{-1,0}};
    for(auto [dx,dy]:dirs){
        int nx=r, ny=c;
        while(inb(nx,ny)){
            if(board[nx][ny].side==SIDE_EMPTY){
                if(nx==target_row && col_in_score(ny)) return true;
            } else if(board[nx][ny].side==SIDE_RIVER){
                // continue riding
            } else break;
            nx+=dx; ny+=dy;
        }
    }
    return false;
}

// Move-centric progress delta for priors: positive means progress for mover side
static double move_progress_delta_for(const Board& board,
                                      const Move& mv,
                                      const string& mover){
    // Basic owner and positions
    if(mv.from.empty()) return 0.0;
    int fr = mv.from[0], fc = mv.from[1];
    if(fr<0 || fr>=(int)board.size() || fc<0 || fc>=(int)board[0].size()) return 0.0;
    if(board[fr][fc].side==SIDE_EMPTY) return 0.0;
    const string owner = owner_to_string(board[fr][fc].owner);
    (void)mover; // should match owner in our generator

    auto base = mdist_to_own_score(owner, fr, fc);

    if(mv.action=="move"){
        if(mv.to.size()<2) return 0.0;
        int tr = mv.to[0], tc = mv.to[1];
        auto nd = mdist_to_own_score(owner, tr, tc);
        int improvement = base - nd;
        double val = (double)improvement;
        // New: amplify multi-step river rides that give >=2 improvement
        int manh = abs(tr - fr) + abs(tc - fc);
        if(manh > 1 && improvement >= 2){
            val += 0.75 * (improvement - 1); // extra reward for big river jumps
        }
        return val;
    }
    if(mv.action=="push"){
        if(mv.to.size()<2 || mv.pushed_to.size()<2) return 0.0;
        int tr = mv.to[0], tc = mv.to[1];
        int pr = mv.pushed_to[0], pc = mv.pushed_to[1];
        // mover progress
        auto nd = mdist_to_own_score(owner, tr, tc);
        double delta = (double)(base - nd);
        // effect on pushed piece
        if(tr>=0 && tr<(int)board.size() && tc>=0 && tc<(int)board[0].size() && board[tr][tc].side!=SIDE_EMPTY){
            string pushed_owner = owner_to_string(board[tr][tc].owner);
            int pb = mdist_to_own_score(pushed_owner, tr, tc);
            int pn = mdist_to_own_score(pushed_owner, pr, pc);
            if(pushed_owner == owner){
                // small bonus if we advance our own pushed piece
                delta += 0.25 * (double)(pb - pn);
            } else {
                // clearer reward: only when we push opponent further from scoring
                int inc = pn - pb;
                if(inc > 0) delta += 0.6 * (double)inc;
            }
        }
        return delta;
    }
    if(mv.action=="flip"){
        // Evaluate potential benefit of creating a river piece
        // If currently a stone, mv.orientation holds target orientation; if a river -> stone flip, treat as mild penalty
    Side side = board[fr][fc].side;
    if(side==SIDE_RIVER){
            // Demoting back to stone: discourage unless near scoring row where stone placement may matter more
            double near_score_bonus = (mdist_to_own_score(owner, fr, fc) <= 2) ? 0.1 : 0.0;
            return -0.35 + near_score_bonus;
        }
    string target_ori = mv.orientation.empty() ? string("horizontal") : mv.orientation;
        int span = potential_run_span_if_flipped(board, fr, fc, target_ori);
        bool opens_path = flip_opens_scoring_path(board, fr, fc, owner, target_ori);
        int rivers_self = count_owned_rivers(board, owner);
        int rivers_opp  = count_owned_rivers(board, (owner=="circle"?"square":"circle"));
        // Distance improvement heuristic: simulate best endpoint along orientation and compute mdist gain
        int best_mdist = base;
        vector<pair<int,int>> dirs = (target_ori=="horizontal")?vector<pair<int,int>>{{0,1},{0,-1}}:vector<pair<int,int>>{{1,0},{-1,0}};
        int R=(int)board.size(); int C=(int)board[0].size(); auto inb=[&](int x,int y){return x>=0&&x<R&&y>=0&&y<C;};
        for(auto [dx,dy]:dirs){
            int nx=fr+dx, ny=fc+dy; int steps=0;
            while(inb(nx,ny)){
                if(board[nx][ny].side==SIDE_EMPTY || board[nx][ny].side==SIDE_RIVER){
                    best_mdist = min(best_mdist, mdist_to_own_score(owner, nx, ny));
                } else break;
                nx+=dx; ny+=dy; steps++; if(steps>20) break; // safety bound
            }
        }
        double dist_gain = (double)(base - best_mdist); // >=0
        double span_bonus = 0.05 * (double)span; // modest impact
        double path_bonus = opens_path ? 0.6 : 0.0;
        double network_bonus = 0.0;
        if(rivers_self < 3) network_bonus += 0.25; // encourage first rivers
        if(rivers_self < rivers_opp) network_bonus += 0.2; // catch up if behind
        // Aggregate, ensure flips with no benefit remain slightly negative to avoid spam
    double safety = unsafe_flow_penalty_after_orientation(board, fr, fc, string_to_owner(owner), string_to_ori(target_ori));
        double raw = dist_gain * 0.4 + span_bonus + path_bonus + network_bonus + safety;
        if(raw < -0.2) raw = -0.2; // floor
        return raw;
    }
    if(mv.action=="rotate"){
        // Rotating an existing river: assess change in run span and potential scoring path opening
    if(board[fr][fc].side!=SIDE_RIVER) return -0.3; // rotating stone -> treat as mild penalty
    string current_ori = ori_to_string(board[fr][fc].orientation);
        string new_ori = mv.orientation.empty()? (current_ori=="horizontal"?"vertical":"horizontal") : mv.orientation;
        int span_current = potential_run_span_if_flipped(board, fr, fc, current_ori);
        int span_new = potential_run_span_if_flipped(board, fr, fc, new_ori);
    bool opens_path = flip_opens_scoring_path(board, fr, fc, owner, new_ori);
        double span_gain = (double)(span_new - span_current);
        double base_gain = span_gain * 0.06; // small scaling
        if(opens_path) base_gain += 0.45;
        // Slight bonus if close to scoring row (helps fine adjustments)
        if(mdist_to_own_score(owner, fr, fc) <= 3) base_gain += 0.05;
        // Safety: penalize rotations enabling opponent flows
    base_gain += unsafe_flow_penalty_after_orientation(board, fr, fc, string_to_owner(owner), string_to_ori(new_ori));
        // Rotations that worsen span strongly discouraged
        if(base_gain < -0.25) base_gain = -0.25;
        return base_gain;
    }
    return 0.0;
}

// Estimate how many of owner's pieces can reach own scoring row in one move (approximate)
// OPTIMIZED: Use geometric heuristic instead of full move generation to avoid O(n³) complexity
// Quick check: from a stone at (r,c), can it reach own scoring row via adjacent river network in one move?
static bool can_reach_score_via_river_one_move(const Board& board, int r, int c, const string& owner){
    int rows = (int)board.size(); if(rows==0) return false;
    int cols = (int)board[0].size();
    int target_row = (owner=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
    if(target_row<0 || target_row>=rows) return false;
    auto inb = [&](int x,int y){ return x>=0 && x<rows && y>=0 && y<cols; };
    auto col_in_sc = [&](int col){ for(int sc: G_SCORE_COLS){ if(sc==col) return true; } return false; };
    // Explore from any adjacent river cell
    vector<pair<int,int>> adj{{1,0},{-1,0},{0,1},{0,-1}};
    vector<vector<char>> vis(rows, vector<char>(cols, 0));
    deque<pair<int,int>> dq;
    for(auto [dx,dy]: adj){
        int nr=r+dx, nc=c+dy; if(!inb(nr,nc)) continue;
        if(board[nr][nc].side==SIDE_RIVER){ vis[nr][nc]=1; dq.emplace_back(nr,nc); }
    }
    while(!dq.empty()){
        auto [x,y]=dq.front(); dq.pop_front();
        Ori ori = board[x][y].orientation;
        vector<pair<int,int>> dirs = (ori==ORI_HORIZONTAL) ? vector<pair<int,int>>{{0,1},{0,-1}}
                                                           : vector<pair<int,int>>{{1,0},{-1,0}};
        for(auto [dx,dy]: dirs){
            int nx=x+dx, ny=y+dy;
            while(inb(nx,ny)){
                if(board[nx][ny].side==SIDE_EMPTY){
                    if(nx==target_row && col_in_sc(ny)) return true; // empty scoring cell reachable
                } else if(board[nx][ny].side==SIDE_RIVER){
                    if(!vis[nx][ny]){ vis[nx][ny]=1; dq.emplace_back(nx,ny); }
                    break; // river continues, switch to exploring from there
                } else {
                    break; // blocked by stone
                }
                nx+=dx; ny+=dy;
            }
        }
    }
    return false;
}

static int estimate_m_reachable_in_one(const Board& board, const string& owner){
    int rows = (int)board.size();
    if(rows==0) return 0; 
    int cols = (int)board[0].size();
    int target_row = (owner=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
    if(target_row<0 || target_row>=rows) return 0;
    
    int n0 = count_in_score_area(board, owner);
    
    // Fast approximation: count pieces within Manhattan distance <= 3 that are in scoring columns
    // or pieces within distance <= 2 that could reach via river
    int potential = 0;
    for(int r=0;r<rows;r++){
        for(int c=0;c<cols;c++){
            if(board[r][c].side==SIDE_EMPTY || owner_to_string(board[r][c].owner)!=owner) continue;
            if(board[r][c].side!=SIDE_STONE) continue; // only stones can score
            
            int dist_to_target = abs(r - target_row);
            
            // Direct moves: within 1-2 squares
            if(dist_to_target <= 2) {
                for(int sc : G_SCORE_COLS){
                    if(sc<0 || sc>=cols) continue;
                    int col_dist = abs(c - sc);
                    // Can reach if Manhattan distance <= 2 (conservative)
                    if(dist_to_target + col_dist <= 2) {
                        potential++;
                        break; // count this piece once
                    }
                }
            }
            // River rides: adjacent into a river network that reaches scoring row
            else {
                if(can_reach_score_via_river_one_move(board, r, c, owner)){
                    potential++;
                }
            }
        }
    }
    
    int m_est = potential - n0;
    if(m_est < 0) m_est = 0;
    return m_est;
}

static inline double quick_state_score(const Board& board, const string& owner){
    int n = count_in_score_area(board, owner);
    int m = estimate_m_reachable_in_one(board, owner);
    return (double)n + 0.1 * (double)m;
}

static vector<double> softmax_from_deltas(const vector<double>& deltas, double tau){
    vector<double> w(deltas.size(), 0.0);
    if(deltas.empty()) return w;
    double maxv = -1e100; for(double d : deltas) maxv = max(maxv, d);
    double Z = 0.0;
    for(size_t i=0;i<deltas.size();++i){
        double z = (deltas[i] - maxv) / max(1e-6, tau);
        double e = exp(z);
        w[i] = e; Z += e;
    }
    if(Z <= 0.0){
        double u = (deltas.empty()?0.0:(1.0/(double)deltas.size()));
        for(size_t i=0;i<deltas.size();++i) w[i]=u;
    } else {
        for(size_t i=0;i<deltas.size();++i) w[i] /= Z;
    }
    return w;
}

// ---------- Feature extraction for learned value (Phase 2) ----------
struct StateFeatures {
    // 16 dims
    // [0] n_self, [1] n_opp
    // [2] m_self, [3] m_opp
    // [4] min_dist_self, [5] min_dist_opp
    // [6] avg_dist_self, [7] avg_dist_opp
    // [8] rivers_self, [9] rivers_opp
    // [10] mobility_self, [11] mobility_opp
    // [12] center_pressure_self, [13] center_pressure_opp
    // [14] threat_self, [15] threat_opp
    array<double,16> v{};
};

static inline pair<double,double> min_avg_dist_to_score(const Board& b, const string& owner){
    int R=b.size(); if(R==0) return {0.0,0.0}; int C=b[0].size();
    double mn = 1e9, sum=0.0; int cnt=0;
    for(int i=0;i<R;i++) for(int j=0;j<C;j++){
        if(b[i][j].side==SIDE_STONE && owner_to_string(b[i][j].owner)==owner){
            int d = mdist_to_own_score(owner, i, j);
            mn = min(mn, (double)d); sum += d; cnt++;
        }
    }
    if(cnt==0) return {0.0, 0.0};
    return { (mn>=1e9?0.0:mn), sum/(double)cnt };
}

static inline int approx_mobility(const Board& b, const string& owner){
    int R=b.size(); if(R==0) return 0; int C=b[0].size();
    int mob=0; vector<pair<int,int>> dirs={{1,0},{-1,0},{0,1},{0,-1}};
    for(int i=0;i<R;i++) for(int j=0;j<C;j++){
        if(b[i][j].side!=SIDE_EMPTY && owner_to_string(b[i][j].owner)==owner){
            for(auto [dx,dy]:dirs){ int ni=i+dx, nj=j+dy; if(ni>=0&&ni<R&&nj>=0&&nj<C){ if(b[ni][nj].side==SIDE_EMPTY) mob++; }}
        }
    }
    return mob;
}

static inline double center_pressure(const Board& b, const string& owner){
    int R=b.size(); if(R==0) return 0.0; int C=b[0].size();
    int start = max(0, C/2 - 1); int endc = min(C-1, start+1); // two central columns
    int own=0, ownTot=0;
    for(int i=0;i<R;i++) for(int j=0;j<C;j++){
    if(b[i][j].side!=SIDE_EMPTY && owner_to_string(b[i][j].owner)==owner){ ownTot++; if(j>=start && j<=endc) own++; }
    }
    if(ownTot==0) return 0.0; return (double)own/(double)ownTot;
}

static inline int threat_one_move_score(const Board& b, const string& owner){
    int R=b.size(); if(R==0) return 0; int C=b[0].size();
    int need=(G_COLS<=12)?4:((G_COLS<=14)?5:6);
    // estimate_m_reachable_in_one already subtracts current n0; if >0, we have a threat
    int m_est = estimate_m_reachable_in_one(b, owner);
    if(m_est>0) return 1;
    // quick extra: if already at need-1, consider threat
    if(count_in_score_area(b, owner) >= need-1) return 1;
    return 0;
}

static StateFeatures extract_features(const Board& b, const string& root_side){
    StateFeatures F{};
    string opp = (root_side=="circle"?"square":"circle");
    int n_self = count_in_score_area(b, root_side);
    int n_opp  = count_in_score_area(b, opp);
    int m_self = estimate_m_reachable_in_one(b, root_side);
    int m_opp  = estimate_m_reachable_in_one(b, opp);
    auto da = min_avg_dist_to_score(b, root_side);
    auto db = min_avg_dist_to_score(b, opp);
    int r_self = count_owned_rivers(b, root_side);
    int r_opp  = count_owned_rivers(b, opp);
    int mob_s  = approx_mobility(b, root_side);
    int mob_o  = approx_mobility(b, opp);
    double cp_s= center_pressure(b, root_side);
    double cp_o= center_pressure(b, opp);
    int th_s   = threat_one_move_score(b, root_side);
    int th_o   = threat_one_move_score(b, opp);

    F.v[0]=n_self; F.v[1]=n_opp;
    F.v[2]=m_self; F.v[3]=m_opp;
    F.v[4]=da.first; F.v[5]=db.first;
    F.v[6]=da.second; F.v[7]=db.second;
    F.v[8]=r_self; F.v[9]=r_opp;
    F.v[10]=mob_s; F.v[11]=mob_o;
    F.v[12]=cp_s; F.v[13]=cp_o;
    F.v[14]=th_s; F.v[15]=th_o;
    return F;
}

// Instant win detection and blocking
static Move check_for_instant_win(const vector<vector<map<string, string>>>& legacy_board,string owner,const map<pair<int,int>,vector<Move>>& moves){
    // Use legacy board for path tracing only; move legality computed via new Board already.
    const auto& board = legacy_board; // alias
    
    // Add bounds validation
    if(board.empty() || board[0].empty()) return {"",{}, {}, {}, ""};
    int n=(int)board.size(), m=(int)board[0].size();
    
    Board b = convert_board(legacy_board);
    int need = (G_COLS<=12)?4:((G_COLS<=14)?5:6);
    
    for(int i=0;i<n;i++) {
        // Validate row size before accessing
        if(i >= (int)board.size()) continue;
        if(board[i].empty()) continue;
        
        for(int j=0;j<(int)board[i].size();j++){
            // Bounds check before accessing board[i][j]
            if(j >= (int)board[i].size()) continue;
            if(board[i][j].empty()) continue;
            
            auto owner_it = board[i][j].find("owner");
            if(owner_it == board[i][j].end() || owner_it->second != owner) continue;
            
            auto it = moves.find({i,j}); 
            if(it==moves.end()) continue;
            
            for(const auto& mv : it->second){ 
                auto nb = update_board(b,mv); 
                if(count_in_score_area(nb,owner)>=need) return mv; 
            }
        }
    }
    return {"",{}, {}, {}, ""};
}

static vector<pair<int,int>> trace_winning_path(const vector<vector<map<string, string>>>& board,const Move& win_move,const string& opponent);
static bool simulate_move_path(const vector<vector<map<string, string>>>& board,int current_row,int current_col,int target_row,int target_col,vector<pair<int,int>>& path_stack,set<pair<int,int>>& visited,int dx1,int dy1);
static Move find_path_blocking_moves(const vector<vector<map<string,string>>>& board,const vector<pair<int,int>>& path,const string& owner,const map<pair<int,int>,vector<Move>>& moves);

static Move find_blocking_move(const vector<vector<map<string,string>>>& legacy_board,Move win_move,string owner,const map<pair<int,int>,vector<Move>>& moves){
    string opponent = (owner=="circle")?"square":"circle";
    auto path = trace_winning_path(legacy_board, win_move, opponent);
    if(path.empty()) return {"",{},{},{},""};
    return find_path_blocking_moves(legacy_board, path, owner, moves);
}

static vector<pair<int,int>> trace_winning_path(const vector<vector<map<string, string>>>& board,const Move& win_move,const string& opponent){
    vector<pair<int,int>> res; (void)opponent;
    if(win_move.action=="move"){
        int sr=win_move.from[0], sc=win_move.from[1], tr=win_move.to[0], tc=win_move.to[1];
        vector<pair<int,int>> st; set<pair<int,int>> vis;
        if(simulate_move_path(board,sr,sc,tr,tc,st,vis,-1,-1)) res=st;
    } else if(win_move.action=="push"){
        int pr=win_move.from[0], pc=win_move.from[1]; (void)pr; (void)pc;
        int tr=win_move.to[0], tc=win_move.to[1];
        int dr=win_move.pushed_to[0], dc=win_move.pushed_to[1];
        if(!board[tr][tc].empty() && board[tr][tc].at("side")=="river") return res;
        vector<pair<int,int>> st; set<pair<int,int>> vis;
        if(!board[win_move.from[0]][win_move.from[1]].empty() && board[win_move.from[0]][win_move.from[1]].at("side")=="river"){
            string ori=board[win_move.from[0]][win_move.from[1]].at("orientation"); vector<pair<int,int>> dirs=(ori=="horizontal")?vector<pair<int,int>>{{0,1},{0,-1}}:vector<pair<int,int>>{{1,0},{-1,0}};
            for(auto [dx,dy]:dirs) if(simulate_move_path(board,tr,tc,dr,dc,st,vis,dx,dy)) res=st;
        } else { st.push_back({tr,tc}); st.push_back({dr,dc}); res=st; }
    }
    return res;
}

static bool simulate_move_path(const vector<vector<map<string, string>>>& board,int cr,int cc,int tr,int tc,vector<pair<int,int>>& st,set<pair<int,int>>& vis,int dx1,int dy1){
    int n=board.size(), m=board[0].size();
    st.push_back({cr,cc}); if(cr==tr && cc==tc) return true; vis.insert({cr,cc});
    if(!board[cr][cc].empty() && board[cr][cc].at("side")=="river"){
        string ori=board[cr][cc].at("orientation"); auto dirs=(ori=="horizontal")?vector<pair<int,int>>{{0,1},{0,-1}}:vector<pair<int,int>>{{1,0},{-1,0}};
        for(auto [dx,dy]:dirs){ int nx=cr+dx, ny=cc+dy; if(nx>=0&&nx<n&&ny>=0&&ny<m){ if(board[nx][ny].empty()){ if(!vis.count({nx,ny}) && simulate_move_path(board,nx,ny,tr,tc,st,vis,dx,dy)) return true; } else if(board[nx][ny].at("side")=="river"){ if(!vis.count({nx,ny}) && simulate_move_path(board,nx,ny,tr,tc,st,vis,dx,dy)) return true; } } }
    } else {
        vector<pair<int,int>> dirs={{-1,0},{1,0},{0,-1},{0,1}};
        if(dx1==-1 && dy1==-1){
            for(auto [dx,dy]:dirs){ int nx=cr+dx, ny=cc+dy; if(nx>=0&&nx<n&&ny>=0&&ny<m && !vis.count({nx,ny})){
                if(board[nx][ny].empty()){ if(nx==tr && ny==tc){ st.push_back({nx,ny}); return true; } }
                else if(board[nx][ny].at("side")=="river"){ if(simulate_move_path(board,nx,ny,tr,tc,st,vis,dx,dy)) return true; }
            }}
        } else {
            int nx=cr+dx1, ny=cc+dy1; if(nx>=0&&nx<n&&ny>=0&&ny<m && !vis.count({nx,ny})) if(simulate_move_path(board,nx,ny,tr,tc,st,vis,dx1,dy1)) return true;
        }
    }
    st.pop_back(); vis.erase({cr,cc}); return false;
}

static Move find_path_blocking_moves(const vector<vector<map<string,string>>>& board,const vector<pair<int,int>>& path,const string& owner,const map<pair<int,int>,vector<Move>>& moves){
    for(const auto& pos: path){ int row=pos.first, col=pos.second; if(board[row][col].empty()){
        for(const auto& kv: moves){ int r=kv.first.first, c=kv.first.second; if(board[r][c].empty()||board[r][c].at("owner")!=owner) continue; for(const Move& mv: kv.second){ if(mv.action=="move" && mv.to.size()>=2 && mv.to[0]==row && mv.to[1]==col) return mv; if(mv.action=="push" && !mv.pushed_to.empty() && mv.pushed_to[0]==row && mv.pushed_to[1]==col) return mv; } }
    } else if(!board[row][col].empty() && board[row][col].at("side")=="river"){ if(board[row][col].at("owner")==owner) return {"flip", {row, col}, {row, col}, {}, {}}; } }
    return {"",{},{},{},""};
}

// ------------------------------------------------------------
// StudentAgent with MCTS (PUCT + TT + Zobrist)
// ------------------------------------------------------------
class StudentAgent {
public:
    explicit StudentAgent(string s) : side(std::move(s)), gen(rd()){
        load_value_model();
    }

    Move choose(const vector<vector<map<string, string>>>& legacy_board,int row,int col,const vector<int>& score_cols,float current_player_time,float opponent_time){
        (void)opponent_time;
        // Update turn indices for guards and priors
        turn_index_ += 1;
        current_turn_index_ = turn_index_;
        
        // Increment generation for TT nodes (used for GC)
        current_generation_++;
        // Lightweight GC trigger: if table too large, sweep unreachable old nodes
        if(tt.size() > TT_SIZE_HARD_LIMIT) {
            perform_gc(nullptr); // We'll supply root later after it's known; this call just enforces size cap
        }
        
        G_ROWS=row; G_COLS=col; G_SCORE_COLS=score_cols;
        map<pair<int,int>,vector<Move>> moves_for_current;
    // Precompute moves ONLY for our pieces (avoids expensive precomputation for ALL pieces)
    Board board = convert_board(legacy_board);
    int n=board.size(), m=board[0].size();
    // Compute current (board, side-to-move) hash for ko tracking
    uint64_t current_hash = hash_board(board, side);
    for(int i=0;i<n;i++) {
        for(int j=0;j<m;j++) {
            if(board[i][j].side!=SIDE_EMPTY && owner_to_string(board[i][j].owner)==side) {
                moves_for_current[{i,j}] = generate_all_moves(board,i,j);
            }
        }
    }

        // River-push ko filter: disallow moves that recreate the previous full position
        auto is_ko = [&](const Move& mv)->bool{
            if(mv.action != "push") return false; // ko applies to river-push sequences
            Board nb = update_board(board, mv);
            string next = (side=="circle"?"square":"circle");
            uint64_t h = hash_board(nb, next);
            return (h == last_fullpos_hash_);
        };
        for(auto &kv : moves_for_current){
            auto &vec = kv.second;
            vec.erase(remove_if(vec.begin(), vec.end(), is_ko), vec.end());
        }

        
        // NEW: Immediate flip of any of our rivers already sitting in scoring row if we still need stones
        {
            int need = (G_COLS<=12)?4:((G_COLS<=14)?5:6);
            int have = count_in_score_area(board, side);
            if(have < need){
                int score_row = (side=="circle")? top_score_row_dynamic() : bottom_score_row_dynamic();
                for(int c : G_SCORE_COLS){
                    if(score_row>=0 && score_row<row && c>=0 && c<col){
                        if(board[score_row][c].side==SIDE_RIVER && owner_to_string(board[score_row][c].owner)==side){
                            // Flip this river to stone (orientation irrelevant)
                            Move flip_move{"flip",{score_row,c},{score_row,c},{}, ""};
                            last_fullpos_hash_ = current_hash;
                            return convert_move_format(flip_move);
                        }
                    }
                }
            }
        }

        // Tactical guardrails: instant win or block
    Move iw = check_for_instant_win(legacy_board, side, moves_for_current); 
    if(iw.action!=""){ last_fullpos_hash_ = current_hash; return convert_move_format(iw); }
    
        // Only check for opponent instant win if they're close to winning (optimization)
        // This avoids expensive opponent move generation in early game
        string opp = (side=="circle")?"square":"circle";
        int need = (G_COLS<=12)?4:((G_COLS<=14)?5:6);
        int opp_have = count_in_score_area(board, opp);
        if(opp_have >= need - 2) {  // Only check if opponent is within 2 stones of winning
            map<pair<int,int>,vector<Move>> opp_moves;
            for(int i=0;i<n;i++) {
                for(int j=0;j<m;j++) {
                    if(board[i][j].side!=SIDE_EMPTY && owner_to_string(board[i][j].owner)==opp) {
                        opp_moves[{i,j}] = generate_all_moves(board,i,j);
                    }
                }
            }
            Move ow = check_for_instant_win(legacy_board, opp, opp_moves);
            if(ow.action!=""){ 
                Move block = find_blocking_move(legacy_board, ow, side, moves_for_current); 
                if(block.action!=""){ last_fullpos_hash_ = current_hash; return convert_move_format(block); } 
            }
        }

        // Time budget: cap at 0.7s max to prevent memory explosion
        // Reduced from 0.5-0.8s to 0.3-0.7s for safety
        double time_limit = 0.3 + 0.4 * std::min(1.0, std::max(0.0, (double)current_player_time / 20.0));
        time_limit = std::min(time_limit, 0.7);  // Hard cap at 0.7s
    Move best = mcts_decide(board, side, n, m, score_cols, time_limit);
        if(best.action==""){
            // fallback: first legal move we own
            for(int i=0;i<n;i++) for(int j=0;j<m;j++) if(board[i][j].side!=SIDE_EMPTY && owner_to_string(board[i][j].owner)==side){ auto v=generate_all_moves(board,i,j); if(!v.empty()){ best=v[0]; break; } }
        }

        // Backtracking guard: avoid immediate undo of previous move
        if(last_move_simple.valid && best.action=="move" && last_move_simple.action=="move"){
            if(!best.from.empty() && !best.to.empty()){
                if(best.from[0]==last_move_simple.tx && best.from[1]==last_move_simple.ty &&
                   best.to[0]==last_move_simple.fx && best.to[1]==last_move_simple.fy){
                    // pick an alternative with non-negative progress if possible
                    double bestProg = -1e9; bool replaced=false; Move alt=best;
                    for(const auto &kv : moves_for_current){
                        int r = kv.first.first, c = kv.first.second;
                        if(board[r][c].side==SIDE_EMPTY || owner_to_string(board[r][c].owner)!=side) continue;
                        for(const auto &mv : kv.second){
                            if(mv.action=="move" && !mv.from.empty() && !mv.to.empty()){
                                if(mv.from[0]==last_move_simple.tx && mv.from[1]==last_move_simple.ty &&
                                   mv.to[0]==last_move_simple.fx && mv.to[1]==last_move_simple.fy) continue; // skip exact undo
                            }
                            double d = move_progress_delta_for(board, mv, side);
                            if(d >= 0.0 && d > bestProg){ bestProg = d; alt = mv; replaced=true; }
                        }
                    }
                    if(replaced) best = alt;
                }
            }
        }

        // Anti-oscillation: avoid immediate flip-back on the same piece unless clearly beneficial
        if(last_move_simple.valid && last_move_simple.action=="flip" && best.action=="flip"){
            if(!best.from.empty() && last_move_simple.fx==best.from[0] && last_move_simple.fy==best.from[1]){
                bool replaced = false;
                double baseS = quick_state_score(board, side);
                // Prefer any non-flip/rotate that isn't clearly worse
                for(const auto &kv : moves_for_current){
                    int r = kv.first.first, c = kv.first.second;
                            if(board[r][c].side==SIDE_EMPTY || owner_to_string(board[r][c].owner)!=side) continue;
                    for(const auto &mv : kv.second){
                        if(mv.action=="flip" || mv.action=="rotate") continue;
                        auto nb = update_board(board, mv);
                        double d = quick_state_score(nb, side) - baseS;
                        if(d >= -0.05){ best = mv; replaced = true; break; }
                    }
                    if(replaced) break;
                }
                if(!replaced){
                    // Allow only beneficial flips/rotates
                    for(const auto &kv : moves_for_current){
                        int r = kv.first.first, c = kv.first.second;
                        if(board[r][c].side==SIDE_EMPTY || owner_to_string(board[r][c].owner)!=side) continue;
                        for(const auto &mv : kv.second){
                            if(!(mv.action=="flip" || mv.action=="rotate")) continue;
                            auto nb = update_board(board, mv);
                            double d = quick_state_score(nb, side) - baseS;
                            if(d >= 0.2){ best = mv; replaced = true; break; }
                        }
                        if(replaced) break;
                    }
                }
            }
        }

        // Flip cooldown: don't flip same piece again within 2 of our turns unless clearly beneficial
        if(best.action=="flip" && last_flip_turn_ >= current_turn_index_ - 2){
            if(!best.from.empty() && last_flip_rc_.first==best.from[0] && last_flip_rc_.second==best.from[1]){
                bool replaced=false; double baseS = quick_state_score(board, side);
                for(const auto &kv : moves_for_current){
                    int r = kv.first.first, c = kv.first.second;
                    if(board[r][c].side==SIDE_EMPTY || owner_to_string(board[r][c].owner)!=side) continue;
                    for(const auto &mv : kv.second){
                        if(mv.action=="flip" || mv.action=="rotate") continue;
                        auto nb = update_board(board, mv);
                        double d = quick_state_score(nb, side) - baseS;
                        if(d >= -0.05){ best = mv; replaced=true; break; }
                    }
                    if(replaced) break;
                }
                if(!replaced){
                    for(const auto &kv : moves_for_current){
                        int r = kv.first.first, c = kv.first.second;
                        if(board[r][c].side==SIDE_EMPTY || owner_to_string(board[r][c].owner)!=side) continue;
                        for(const auto &mv : kv.second){
                            if(!(mv.action=="flip" || mv.action=="rotate")) continue;
                            auto nb = update_board(board, mv);
                            double d = quick_state_score(nb, side) - baseS;
                            if(d >= 0.4){ best = mv; replaced=true; break; }
                        }
                        if(replaced) break;
                    }
                }
            }
        }

        // Anti-repetition guard: avoid repeating the same move 3 times in a row
        SimpleMove current_simple = to_simple(best);
        if(last_move_simple.valid && same_simple(last_move_simple, current_simple)){
            consecutive_repeat_count++;
        } else {
            consecutive_repeat_count = 0;
        }
        if(consecutive_repeat_count >= 3){
            // Try to choose a different move among our legal moves
            bool replaced = false;
            for(const auto &kv : moves_for_current){
                int r = kv.first.first, c = kv.first.second;
                if(board[r][c].side==SIDE_EMPTY || owner_to_string(board[r][c].owner)!=side) continue;
                for(const auto &mv : kv.second){
                    if(!same_simple(to_simple(mv), current_simple)){
                        best = mv; replaced = true; break;
                    }
                }
                if(replaced) break;
            }
            consecutive_repeat_count = 0;
        }
        last_move_simple = to_simple(best);
        if(best.action=="flip" && !best.from.empty()){
            last_flip_rc_ = {best.from[0], best.from[1]};
            last_flip_turn_ = current_turn_index_;
        }
        recent_moves.push_back(last_move_simple);
        if(recent_moves.size() > 20) recent_moves.erase(recent_moves.begin());


        if(!best.from.empty() && !best.to.empty()){
            double chosen_prog = move_progress_delta_for(board, best, side);
            if(chosen_prog < 1.0){
                double best_ride_prog = chosen_prog;
                Move best_ride = best;
                for(const auto &kv : moves_for_current){
                    int r = kv.first.first, c = kv.first.second;
                    if(board[r][c].side==SIDE_EMPTY || owner_to_string(board[r][c].owner)!=side) continue;
                    for(const auto &mv : kv.second){
                        if(mv.action!="move" || mv.from.empty() || mv.to.empty()) continue;
                        int manh = abs(mv.to[0]-mv.from[0]) + abs(mv.to[1]-mv.from[1]);
                        if(manh <= 1) continue; // only consider river rides
                        double prog = move_progress_delta_for(board, mv, side);
                        if(prog >= 2.0 && prog > best_ride_prog + 0.1){
                            best_ride_prog = prog;
                            best_ride = mv;
                        }
                    }
                }
                if(best_ride_prog > chosen_prog + 0.1){
                    best = best_ride;
                }
            }
        }

        // Update ko reference to current full position (for next turn)
        last_fullpos_hash_ = current_hash;
        return convert_move_format(best);
    }

private:
    string side;
    random_device rd; mt19937 gen;
    // Phase 2: optional learned value weights
    vector<double> value_weights_; bool has_value_model_ = false;

    // Anti-repetition helpers
    struct SimpleMove{
        string action=""; int fx=-1, fy=-1; int tx=-1, ty=-1; int px=-1, py=-1; string orientation=""; bool valid=false;
    };
    SimpleMove last_move_simple; int consecutive_repeat_count = 0; vector<SimpleMove> recent_moves;
    // Turn tracking and flip cooldown
    int turn_index_ = 0; int current_turn_index_ = 0; pair<int,int> last_flip_rc_{-1,-1}; int last_flip_turn_ = -1000000;

    SimpleMove to_simple(const Move& m){
        SimpleMove s; s.action = m.action; s.valid = true;
        if(!m.from.empty()){ s.fx = m.from[0]; s.fy = m.from[1]; }
        if(!m.to.empty()){ s.tx = m.to[0]; s.ty = m.to[1]; }
        if(!m.pushed_to.empty()){ s.px = m.pushed_to[0]; s.py = m.pushed_to[1]; }
        s.orientation = m.orientation; return s;
    }
    bool same_simple(const SimpleMove& a, const SimpleMove& b){
        if(a.action!=b.action) return false;
        if(a.action=="push") return a.fx==b.fx && a.fy==b.fy && a.tx==b.tx && a.ty==b.ty && a.px==b.px && a.py==b.py;
        if(a.action=="move") return a.fx==b.fx && a.fy==b.fy && a.tx==b.tx && a.ty==b.ty;
        if(a.action=="flip"||a.action=="rotate") return a.fx==b.fx && a.fy==b.fy && a.orientation==b.orientation;
        return a.fx==b.fx && a.fy==b.fy && a.tx==b.tx && a.ty==b.ty;
    }

    // Move-format conversion: Internal uses [row,col], Python/engine expects [col,row]
    // This swap is CORRECT and necessary for compatibility with gameEngine.py
    Move convert_move_format(const Move& m){ 
        Move r; 
        r.action=m.action; 
        if(!m.from.empty()) r.from={m.from[1],m.from[0]}; 
        if(!m.to.empty()) r.to={m.to[1],m.to[0]}; 
        if(!m.pushed_to.empty()) r.pushed_to={m.pushed_to[1],m.pushed_to[0]}; 
        r.orientation=m.orientation; 
        return r; 
    }

    struct MCTSNode{
        int N=0; double W=0.0; double Q=0.0;
        uint64_t hash=0ULL; string to_move; Board state; // migrated to compact Board
        vector<Move> moves; vector<double> priors; vector<MCTSNode*> children; vector<int> order_by_prior; int expanded_count=0; bool terminal=false;
        // GC bookkeeping
        int generation = 0;               // created when agent generation was this value
        int last_access_generation = 0;   // last touched generation (for reachability/recency GC)
        // NOTE: Q/W are stored from ROOT player's perspective (root_side)
        // Backup does NOT flip sign; selection/tie-break also use root perspective
    };

    unordered_map<uint64_t,MCTSNode*> tt;
    bool zob_init=false; int zob_r=-1, zob_c=-1; vector<vector<array<uint64_t,6>>> zob_cell; array<uint64_t,2> zob_side{{0ULL,0ULL}};

    void init_zobrist(int r,int c){ 
        if(zob_init && r==zob_r && c==zob_c) return; 
        // Use single RNG with unique seeding per cell based on position
        std::mt19937_64 rng(0xC0FFEE1234ULL); 
        zob_cell.assign(r, vector<array<uint64_t,6>>(c)); 
        for(int i=0;i<r;i++) {
            for(int j=0;j<c;j++) {
                for(int k=0;k<6;k++) {
                    // XOR position info into each random value for uniqueness
                    zob_cell[i][j][k] = rng() ^ ((uint64_t)i<<48) ^ ((uint64_t)j<<32) ^ ((uint64_t)k<<16);
                }
            }
        }
        zob_side[0]=rng() ^ 0xAAAAAAAAAAAAAAAAULL; 
        zob_side[1]=rng() ^ 0x5555555555555555ULL; 
        zob_init=true; zob_r=r; zob_c=c; 
    }
    static inline int piece_code(const Cell& cell){
        if(cell.side==SIDE_EMPTY) return -1;
        if(cell.side==SIDE_STONE){ return (cell.owner==OWN_CIRCLE?0:1); }
        // river piece; encode owner + orientation
        if(cell.owner==OWN_CIRCLE){ return (cell.orientation==ORI_HORIZONTAL?2:3); }
        return (cell.orientation==ORI_HORIZONTAL?4:5);
    }
    uint64_t hash_board(const Board& b, const string& to_move){
        init_zobrist((int)b.size(), (int)b[0].size());
        uint64_t h=0ULL; int R=b.size(), C=b[0].size();
        for(int i=0;i<R;i++) for(int j=0;j<C;j++){ int pc=piece_code(b[i][j]); if(pc>=0) h^=zob_cell[i][j][pc]; }
        h ^= (to_move=="circle"?zob_side[0]:zob_side[1]);
        return h;
    }
    static inline string opponent_of(const string& s){ return (s=="circle"?"square":"circle"); }

    vector<Move> collect_moves(const Board& b, const string& who){ int R=b.size(); if(R==0) return {}; int C=b[0].size(); (void)C; vector<Move> mv; for(int i=0;i<R;i++) for(int j=0;j<(int)b[0].size();j++){ if(b[i][j].side!=SIDE_EMPTY && owner_to_string(b[i][j].owner)==who){ auto v=generate_all_moves(b,i,j); for(const auto& m:v){ auto nb=update_board(b,m); if(nb!=b) mv.push_back(m);} } } return mv; }
    double leaf_value(const Board& b, const string& root_side){
        int need=(G_COLS<=12)?4:((G_COLS<=14)?5:6);
        // Terminal values: reduced from ±100 to ±50 for better exploration balance
        // Still strong enough to dominate non-terminals but allows more nuanced search
        if(count_in_score_area(b,root_side)>=need) return 50.0;
        if(count_in_score_area(b,opponent_of(root_side))>=need) return -50.0;
        // If a learned value model is available, use it
        if(has_value_model_ && !value_weights_.empty()){
            StateFeatures F = extract_features(b, root_side);
            double dot=0.0; size_t L=min(value_weights_.size(), F.v.size());
            for(size_t i=0;i<L;i++) dot += value_weights_[i] * F.v[i];
            // Normalize and clamp conservatively
            double v = dot;
            if(v>10.0) v=10.0; if(v<-10.0) v=-10.0;
            return v;
        }
        // Fallback: quick_state_score differential (includes reachability)
        double qs_self = quick_state_score(b, root_side);
        double qs_opp  = quick_state_score(b, opponent_of(root_side));
        double diff = qs_self - qs_opp; // can range broader than [-need,need]
        double scaled = diff / (double)(need + 0.5);
        // Scale to larger range for better PUCT discrimination (keep non-terminals within ±8)
        scaled *= 10.0;
        if(scaled>8.0) scaled=8.0; if(scaled<-8.0) scaled=-8.0;
        return scaled;
    }

    Move mcts_decide(const Board& board,
                     const string& root_side,
                     int rows, int cols, const vector<int>& score_cols,
                     double time_limit){
        (void)score_cols;
        init_zobrist(rows, cols);

        string to_move = root_side;
    uint64_t h = hash_board(board, to_move);
        MCTSNode* root = nullptr;
        auto it = tt.find(h);
        if(it != tt.end()){
            root = it->second;
            root->state = board;
            root->to_move = to_move;
            root->hash = h;
            root->last_access_generation = current_generation_;
        } else {
            root = new MCTSNode();
            root->state = board;
            root->to_move = to_move;
            root->hash = h;
            root->terminal = is_terminal(board);
            root->moves = collect_moves(board, to_move);
            int M = (int)root->moves.size();
            // Move-centric priors: softmax over progress delta per move
            vector<double> deltas; deltas.reserve(M);
            for(int i=0;i<M;i++){
                double d = move_progress_delta_for(board, root->moves[i], to_move);
                // Stronger early-game discouragement of flips/rotates unless clearly beneficial
                if (current_turn_index_ <= 5) {
                    const string &act = root->moves[i].action;
                    if ((act=="flip" || act=="rotate") && d < 0.1) d -= 0.2;
                }
                deltas.push_back(d);
            }
            const double tau = 1.2; // slightly sharper
            root->priors = softmax_from_deltas(deltas, tau);
            root->children.assign(M, nullptr);
            root->order_by_prior.resize(M);
            iota(root->order_by_prior.begin(), root->order_by_prior.end(), 0);
            sort(root->order_by_prior.begin(), root->order_by_prior.end(), [&](int a,int b){ return root->priors[a] > root->priors[b]; });
            root->generation = current_generation_;
            root->last_access_generation = current_generation_;
            tt[h] = root;
        }
        // Provide current root for GC later
        last_root_hash_ = h;

    const double c_puct = 1.2;
    const int K0 = 6;
        const double K1 = 1.5;
        
        // Memory safety: limit nodes created per search to prevent std::bad_alloc
        int MAX_NODES_PER_SEARCH = 50000;  // default ~10MB per search
        if(current_turn_index_ == 1) {
            // Constrain first move to avoid long stall
            time_limit = std::min(time_limit, 0.45);
            MAX_NODES_PER_SEARCH = 15000;
        }
        int nodes_created_this_search = 0;

        auto start = chrono::high_resolution_clock::now();
        while(true){
            auto now = chrono::high_resolution_clock::now();
            if(chrono::duration<double>(now - start).count() >= time_limit) break;
            
            // Safety: stop if too many nodes created (prevent memory explosion)
            if(nodes_created_this_search >= MAX_NODES_PER_SEARCH) break;

            vector<pair<MCTSNode*, int>> path;
            MCTSNode* node = root;
            bool expanded = false;
            
            // Depth limit: prevent infinite depth searches that explode memory
            const int MAX_DEPTH = 100;
            int depth = 0;

            while(true){
                if(node->terminal) break;
                int M = (int)node->moves.size();
                if(M == 0) break;
                
                // Safety: limit search depth
                if(depth >= MAX_DEPTH) break;

                int K = std::max(K0, (int)floor(K1 * sqrt(std::max(1, node->N))));
                K = std::min(K, M);

                if(node->expanded_count < K){
                    int idx = node->order_by_prior[node->expanded_count++];
                    if(node->children[idx] == nullptr){
                        auto nb = update_board(node->state, node->moves[idx]);
                        string child_to_move = opponent_of(node->to_move);
                        uint64_t ch = hash_board(nb, child_to_move);
                        MCTSNode* child = nullptr;
                        bool tt_hit = false;
                        // TEMPORARY: Disable TT reuse to avoid double-counting bug
                        // TODO: Implement proper TT handling (virtual loss or separate tree)
                        auto jt = tt.find(ch);
                        if(false && jt != tt.end()){  // Disabled
                            child = jt->second;
                            tt_hit = true;  // Mark that this was a TT hit
                        } else {
                            child = new MCTSNode();
                            child->state = std::move(nb);
                            child->to_move = child_to_move;
                            child->hash = ch;
                            child->terminal = is_terminal(child->state);
                            if(!child->terminal){
                                child->moves = collect_moves(child->state, child->to_move);
                                int CM = (int)child->moves.size();
                                // Move-centric priors for child
                                vector<double> cd; cd.reserve(CM);
                                for(int j=0;j<CM;j++){
                                    double d = move_progress_delta_for(child->state, child->moves[j], child->to_move);
                                    if (current_turn_index_ <= 5) {
                                        const string &act = child->moves[j].action;
                                        if ((act=="flip" || act=="rotate") && d < 0.1) d -= 0.2;
                                    }
                                    cd.push_back(d);
                                }
                                const double tau_c = 1.2;
                                child->priors = softmax_from_deltas(cd, tau_c);
                                child->children.assign(CM, nullptr);
                                child->order_by_prior.resize(CM);
                                iota(child->order_by_prior.begin(), child->order_by_prior.end(), 0);
                                sort(child->order_by_prior.begin(), child->order_by_prior.end(), [&](int a,int b){ return child->priors[a] > child->priors[b]; });
                            }
                            child->generation = current_generation_;
                            child->last_access_generation = current_generation_;
                            tt[ch] = child;
                            nodes_created_this_search++;  // Track node creation
                        }
                        node->children[idx] = child;

                        // Evaluate or reuse existing evaluation
                        double v;
                        if(tt_hit){
                            // TT hit: use existing Q-value instead of re-evaluating
                            // This prevents double-counting the same position's value
                            v = child->Q;
                        } else {
                            // New node: evaluate it
                            v = leaf_value(child->state, root_side);
                        }

                        // Debug: Log expansions, especially high values
                        if(false && current_turn_index_ == 1 && (root->N < 5 || fabs(v) > 40.0)){
                            cerr << "  Expand[depth=" << path.size() << "]: child to_move=" << child->to_move
                                 << " v=" << v << (tt_hit ? " [TT_HIT]" : "");
                            if(fabs(v) > 40.0){
                                cerr << " TERMINAL! n_circle=" << count_in_score_area(child->state, "circle")
                                     << " n_square=" << count_in_score_area(child->state, "square");
                                // Print the IMMEDIATE move that led here
                                cerr << "\n    Last move: " << node->moves[idx].action;
                                if(!node->moves[idx].from.empty())
                                    cerr << " from(" << node->moves[idx].from[0] << "," << node->moves[idx].from[1] << ")";
                                if(!node->moves[idx].to.empty())
                                    cerr << " to(" << node->moves[idx].to[0] << "," << node->moves[idx].to[1] << ")";
                                // Print scoring row
                                int score_row = (root_side == "circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
                                cerr << "\n    Scoring row " << score_row << ": ";
                                for(int sc : G_SCORE_COLS){
                                    if(child->state[score_row][sc].side == SIDE_STONE){
                                        cerr << "col" << sc << "="
                                             << owner_to_string(child->state[score_row][sc].owner)[0] << " ";
                                    }
                                }
                                // Count total stones on board
                                int total_stones = 0;
                                for(int i=0;i<(int)child->state.size();i++){
                                    for(int j=0;j<(int)child->state[0].size();j++){
                                        if(child->state[i][j].side==SIDE_STONE) total_stones++;
                                    }
                                }
                                cerr << "\n    Total stones on board: " << total_stones << " (should be ~10)";
                            }
                            cerr << endl;
                        }

                        // Backup: maintain values from root perspective
                        // Add path edge before backup
                        path.push_back({node, idx});

                        // Backup value: root perspective everywhere (no sign flips)
                        for(auto it = path.rbegin(); it != path.rend(); ++it){
                            MCTSNode* pn = it->first;
                            pn->N += 1;
                            pn->W += v;
                            pn->Q = pn->W / std::max(1, pn->N);
                        }
                        // Also update the child itself - BUT only if it's a NEW node, not a TT hit
                        // TT hits have already been evaluated; updating them again causes double-counting
                        if(!tt_hit){
                            child->N += 1;
                            child->W += v;
                            child->Q = child->W / std::max(1, child->N);
                        }
                        child->last_access_generation = current_generation_;
                        
                        expanded = true;
                    }
                    break;
                }

                int best_i = -1; double best_score = -1e100;
                double sqrtN = sqrt((double)std::max(1, node->N));
                if(node == root){
                    // Root-only safety mask: prefer children that don't spike opponent reachability
                    int best_safe_i = -1; double best_safe_score = -1e100;
                    int best_any_i  = -1; double best_any_score  = -1e100;
                    for(int k=0;k<node->expanded_count;k++){
                        int idx = node->order_by_prior[k];
                        MCTSNode* ch = node->children[idx];
                        if(!ch) continue;
                        double prior = node->priors[idx];
                        double q_value = (node->to_move == root_side) ? ch->Q : -ch->Q;
                        double u = c_puct * prior * sqrtN / (1.0 + (double)ch->N);
                        double score = q_value + u;
                        if(score > best_any_score){ best_any_score = score; best_any_i = idx; }
                        int m_self = cached_estimate_reach(ch->state, root_side);
                        int m_opp  = cached_estimate_reach(ch->state, opponent_of(root_side));
                        bool safe = (m_opp - m_self <= 1);
                        if(safe && score > best_safe_score){ best_safe_score = score; best_safe_i = idx; }
                    }
                    if(best_safe_i != -1){ best_i = best_safe_i; best_score = best_safe_score; }
                    else { best_i = best_any_i; best_score = best_any_score; }
                } else {
                    for(int k=0;k<node->expanded_count;k++){
                        int idx = node->order_by_prior[k];
                        MCTSNode* ch = node->children[idx];
                        if(!ch) continue;
                        double prior = node->priors[idx];
                        double q_value = (node->to_move == root_side) ? ch->Q : -ch->Q;
                        double u = c_puct * prior * sqrtN / (1.0 + (double)ch->N);
                        double score = q_value + u;
                        if(score > best_score){ best_score = score; best_i = idx; }
                    }
                }
                if(best_i < 0) break;
                path.push_back({node, best_i});
                node = node->children[best_i];
                if(node) node->last_access_generation = current_generation_;
                depth++;  // Increment depth when traversing down
            }

            if(!expanded){
                // Re-evaluate the leaf that was selected but couldn't expand further
                double v = leaf_value(node->state, root_side);
                
                // Backup: root perspective everywhere (no sign flips)
                for(auto it = path.rbegin(); it != path.rend(); ++it){
                    MCTSNode* pn = it->first;
                    pn->N += 1;
                    pn->W += v;
                    pn->Q = pn->W / std::max(1, pn->N);
                }
                // Update leaf node itself
                node->N += 1;
                node->W += v;
                node->Q = node->W / std::max(1, node->N);
                node->last_access_generation = current_generation_;
            }
            // Early cutoff: if a root child dominates visits, and we've spent a chunk of time, stop early
            double elapsed = chrono::duration<double>(now - start).count();
            if(elapsed > 0.5 * time_limit && root->expanded_count > 0){
                int totalN = 0; int bestN = -1;
                for(int k=0;k<root->expanded_count;k++){
                    int idx = root->order_by_prior[k];
                    MCTSNode* ch = root->children[idx];
                    if(!ch) continue;
                    totalN += ch->N;
                    if(ch->N > bestN) bestN = ch->N;
                }
                if(totalN > 0){
                    double prop = (double)bestN / (double)totalN;
                    if(prop >= 0.8) break;
                }
            }
        }

        int M = (int)root->moves.size();
        if(M == 0) return {"",{},{},{},""};
        int best_idx = -1; int bestN = -1; double bestQ = -1e18;
        int cap = root->expanded_count;

        // Debug: Print top 5 candidates
        if(false){
            cerr << "\n=== MCTS Decision (Turn " << current_turn_index_ << ", " << root_side << ") ===" << endl;
            cerr << "Root N=" << root->N << " expanded=" << cap << "/" << M << endl;
            for(int k=0;k<min(5, cap);k++){
                int idx = root->order_by_prior[k];
                MCTSNode* ch = root->children[idx];
                if(!ch) continue;
                const Move& mv = root->moves[idx];
                cerr << k+1 << ". " << mv.action;
                if(!mv.from.empty()) cerr << " from(" << mv.from[0] << "," << mv.from[1] << ")";
                if(!mv.to.empty()) cerr << " to(" << mv.to[0] << "," << mv.to[1] << ")";
                cerr << " | N=" << ch->N << " Q=" << ch->Q << " prior=" << root->priors[idx] << endl;
            }
        }

        for(int k=0;k<cap;k++){
            int idx = root->order_by_prior[k];
            MCTSNode* ch = root->children[idx];
            if(!ch) continue;
            // Compare using root perspective for tie-break
            int n = ch->N; double q = ch->Q;
            if(n > bestN || (n==bestN && q>bestQ)){
                bestN = n; bestQ = q; best_idx = idx;
            }
        }
        if(best_idx < 0) best_idx = 0;

        if(false){ cerr << "=> Chose: " << root->moves[best_idx].action << " with N=" << bestN << " Q=" << bestQ << endl; }

        // Fortress fallback: if all root candidates look clearly negative and no instant win/block, prefer safe set
        // Safety criterion: limit opponent reachability growth and prefer better margin
        if(bestQ < -0.35 && cap > 0){
            int safe_idx = -1; double best_margin = -1e18; int bestN2=-1;
            for(int k=0;k<cap;k++){
                int idx = root->order_by_prior[k]; MCTSNode* ch = root->children[idx]; if(!ch) continue;
                int m_self = cached_estimate_reach(ch->state, root_side);
                int m_opp  = cached_estimate_reach(ch->state, opponent_of(root_side));
                // Safe if opponent reachability doesn't spike compared to self
                if(m_opp - m_self > 1) continue;
                double qs_self = quick_state_score(ch->state, root_side);
                double qs_opp  = quick_state_score(ch->state, opponent_of(root_side));
                double margin = qs_self - (qs_opp + 0.1 * m_opp);
                if(margin > best_margin || (fabs(margin-best_margin)<1e-9 && ch->N>bestN2)){
                    best_margin = margin; bestN2 = ch->N; safe_idx = idx;
                }
            }
            if(safe_idx >= 0) best_idx = safe_idx;
        }
        // Store chosen child for potential reuse next turn
        MCTSNode* chosen_child_ptr = nullptr;
        if(best_idx >=0 && best_idx < (int)root->children.size()){
            MCTSNode* chosen_child = root->children[best_idx];
            if(chosen_child){
                chosen_child_ptr = chosen_child;
                last_chosen_child_hash_ = chosen_child->hash;
                chosen_child->last_access_generation = current_generation_;
            } else {
                last_chosen_child_hash_ = 0ULL;
            }
        }
        // After decision, run GC if TT exceeds soft limit — keep only chosen subtree if available
        if(tt.size() > TT_SIZE_SOFT_LIMIT){
            perform_gc(chosen_child_ptr ? chosen_child_ptr : root);
        }
        return root->moves[best_idx];
    }

    void load_value_model(){
        // Try models/value_v1.txt (whitespace or comma separated doubles)
        vector<string> candidates = {
            string("models/value_v1.txt"),
            string("COL333_A5/models/value_v1.txt"),
            string("../models/value_v1.txt")
        };
        for(const auto &p : candidates){
            ifstream in(p);
            if(!in.good()) continue;
            string line; string all;
            while(std::getline(in, line)){
                if(!all.empty()) all.push_back(' ');
                all += line;
            }
            if(all.empty()) continue;
            // replace commas with spaces
            for(char &ch : all){ if(ch==',') ch=' '; }
            std::istringstream iss(all);
            vector<double> w; double x;
            while(iss >> x) w.push_back(x);
            if(!w.empty()){
                value_weights_ = std::move(w);
                has_value_model_ = true;
                return;
            }
        }
        has_value_model_ = false;
    }

private:
    // Reuse & GC bookkeeping
    uint64_t last_root_hash_ = 0ULL;
    uint64_t last_chosen_child_hash_ = 0ULL;
    // Ko tracking: store previous full position hash (board + side-to-move)
    uint64_t last_fullpos_hash_ = 0ULL;
    int current_generation_ = 0;
    // TT limits
    const size_t TT_SIZE_SOFT_LIMIT = 150000; // trigger GC
    const size_t TT_SIZE_HARD_LIMIT = 250000; // emergency cap
    const int GENERATION_LAG_TOLERANCE = 2;   // keep nodes accessed within last N generations
    // Reachability cache (hash -> estimate)
    unordered_map<uint64_t,int> reach_cache_;

    int cached_estimate_reach(const Board& b, const string& owner){
        uint64_t key = hash_board(b, owner) ^ (owner=="circle"?0xABCD1234ULL:0xDCBA4321ULL);
        auto it = reach_cache_.find(key);
        if(it != reach_cache_.end()) return it->second;
        int v = estimate_m_reachable_in_one(b, owner);
        reach_cache_[key] = v;
        // Basic cache size control
        if(reach_cache_.size() > 50000){ reach_cache_.clear(); }
        return v;
    }

    void perform_gc(MCTSNode* root){
        // If below soft limit, skip
        if(tt.size() <= TT_SIZE_SOFT_LIMIT) return;
        unordered_set<uint64_t> keep;
        if(!root && last_root_hash_){
            auto it = tt.find(last_root_hash_);
            if(it != tt.end()) root = it->second;
        }
        if(root){
            // Mark phase: DFS from current root
            vector<MCTSNode*> stack{root};
            while(!stack.empty()){
                MCTSNode* nd = stack.back(); stack.pop_back();
                if(!nd) continue;
                keep.insert(nd->hash);
                for(auto* ch : nd->children) if(ch && !keep.count(ch->hash)) stack.push_back(ch);
            }
        }
        // Sweep phase: remove nodes not marked & too old or over hard limit pressure
        size_t before = tt.size();
        for(auto it = tt.begin(); it != tt.end(); ){
            MCTSNode* node = it->second;
            bool remove = false;
            if(!keep.count(it->first)){
                if(current_generation_ - node->last_access_generation > GENERATION_LAG_TOLERANCE) remove = true;
            }
            if(!remove && tt.size() > TT_SIZE_HARD_LIMIT) remove = true; // emergency pressure
            if(remove){ delete node; it = tt.erase(it); }
            else { ++it; }
        }
    }
};

// -----------------------------
// Added private members & GC helpers (appended after class for simplicity if not placed inside)
// -----------------------------
// Extend MCTSNode with generation tracking

// ------------------------------------------------------------
// PyBind11 bindings
// ------------------------------------------------------------
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
