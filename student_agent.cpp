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
using namespace std;

namespace py = pybind11;

// ------------------------------------------------------------
// Types
// ------------------------------------------------------------
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
static bool G_FAST_EVAL = false; // enable lightweight leaf evaluation under tight time

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

// Manhattan distance to the owner's scoring area: row distance + closest scoring column distance
static inline int min_distance_from_scoring_area(const string& owner,int row,int col){
    int target_row = (owner=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
    int best = 1000000;
    for(int sc : G_SCORE_COLS){ best = min(best, abs(row - target_row) + abs(col - sc)); }
    return best==1000000?100:best;
}

// Compute, for a stone at (r,c) owned by 'who', the best distance to scoring
// reachable in ONE legal move (adjacent step or a single river ride along the
// adjacent river's orientation). This is an approximation of move potential
// used inside evaluation; we keep it cheap and non-recursive.
static int best_next_distance_to_score(const vector<vector<map<string,string>>>& board,
                                       const string& who, int r, int c){
    const int R = (int)board.size();
    const int C = (int)board[0].size();
    auto inside = [&](int x,int y){ return x>=0 && x<R && y>=0 && y<C; };

    int best = 100;
    const pair<int,int> dirs[4]={{1,0},{-1,0},{0,1},{0,-1}};

    for(auto [dx,dy]:dirs){
        int nx=r+dx, ny=c+dy;
        if(!inside(nx,ny)) continue;
        // Adjacent empty step
        if(board[nx][ny].empty()){
            if(!is_opponent_score_cell_dyn(nx,ny,who)){
                best = min(best, min_distance_from_scoring_area(who, nx, ny));
            }
            continue;
        }
        // Adjacent river ride (single-segment, along this neighbor's orientation)
        if(board[nx][ny].at("side")=="river"){
            string ori = board[nx][ny].at("orientation");
            vector<pair<int,int>> rays = (ori=="horizontal") ? vector<pair<int,int>>{{0,1},{0,-1}}
                                                               : vector<pair<int,int>>{{1,0},{-1,0}};
            for(auto [rx,ry]:rays){
                int tx = nx + rx, ty = ny + ry;
                while(inside(tx,ty)){
                    if(board[tx][ty].empty()){
                        if(!is_opponent_score_cell_dyn(tx,ty,who)){
                            best = min(best, min_distance_from_scoring_area(who, tx, ty));
                        }
                    } else {
                        // stop at first non-empty (stone or another river)
                        break;
                    }
                    tx += rx; ty += ry;
                }
            }
        }
    }
    return best;
}

// Lightweight shape value and threat counts for priors/scoring: 
// V_shape = (n_self - n_opp) + 0.1*(can_next_self - can_next_opp)
// Also returns can_next_self/can_next_opp for defensive gating
struct ShapeInfo {
    double vshape;
    int can_next_self;
    int can_next_opp;
};

// Forward declaration from below
static int count_in_score_area(const vector<vector<map<string, string>>>& board,string owner);

// Count friendly rivers in scoring columns (vertical favored)
static int count_scoring_col_rivers_for(const vector<vector<map<string,string>>>& b, const string& who){
    int R=b.size(), C=b[0].size(); (void)C;
    int cnt=0;
    for(int i=0;i<R;i++){
        for(int j=0;j<(int)b[0].size();j++){
            if(b[i][j].empty()) continue;
            auto itO = b[i][j].find("owner");
            auto itS = b[i][j].find("side");
            if(itS==b[i][j].end() || itO==b[i][j].end()) continue;
            if(itS->second!="river" || itO->second!=who) continue;
            if(!col_in_score(j)) continue;
            string ori = "";
            auto itOri = b[i][j].find("orientation"); if(itOri!=b[i][j].end()) ori=itOri->second;
            cnt += (ori=="vertical"? 2 : 1);
        }
    }
    return cnt;
}

static ShapeInfo shape_value(const vector<vector<map<string,string>>>& b, const string& owner){
    const string opp = (owner=="circle")?"square":"circle";
    int n_self = count_in_score_area(b, owner);
    int n_opp  = count_in_score_area(b, opp);
    int can_self=0, can_opp=0;
    int R=b.size(), C=b[0].size();
    for(int i=0;i<R;i++){
        for(int j=0;j<C;j++){
            if(b[i][j].empty() || b[i][j].at("side")!="stone") continue;
            const string& who = b[i][j].at("owner");
            int d1 = best_next_distance_to_score(b, who, i, j);
            if(d1==0){ if(who==owner) ++can_self; else ++can_opp; }
        }
    }
    // Offense-forward shape: emphasize increasing our finish-in-one potential
    double v = (double)(n_self - n_opp) + 0.3 * (double)(can_self - can_opp);
    return ShapeInfo{v, can_self, can_opp};
}

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------
static bool allows_flow_into_opponent_score_after_orientation(
    const vector<vector<map<string,string>>>& board,
    int r, int c,
    const string& owner,
    const string& orientation)
{
    int n = board.size();
    int m = board[0].size();
    auto inb = [&](int x,int y){ return x>=0 && x<n && y>=0 && y<m; };

    queue<pair<int,int>> q;
    set<pair<int,int>> vis;
    q.push({r,c}); vis.insert({r,c});

    while(!q.empty()){
        auto [x,y] = q.front(); q.pop();
        string ori;
        if (x==r && y==c) ori = orientation; else {
            if (board[x][y].empty()) continue;
            if (board[x][y].at("side") != string("river")) continue;
            ori = board[x][y].at("orientation");
        }
        vector<pair<int,int>> dirs = (ori=="horizontal") ? vector<pair<int,int>>{{0,1},{0,-1}}
                                                           : vector<pair<int,int>>{{1,0},{-1,0}};
        for (auto [dx,dy] : dirs){
            int nx = x + dx, ny = y + dy;
            while(inb(nx,ny)){
                if (board[nx][ny].empty()){
                    if (is_opponent_score_cell_dyn(nx, ny, owner)) return true;
                } else if (board[nx][ny].at("side")=="river"){
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

// Explore river for moves (stone or river rides)
static void explore_river_move(const vector<vector<map<string, string>>>& board,int x, int y, vector<Move>& moves,
                               int rows, int cols, int from_row, int from_col, set<pair<int,int>>& visited){
    if (visited.count({x, y})) return;
    visited.insert({x, y});
    if (board[x][y].at("side") != "river") return;
    string orientation = board[x][y].at("orientation");
    vector<pair<int,int>> directions = (orientation == "horizontal") ? vector<pair<int,int>>{{0,1},{0,-1}}
                                                                      : vector<pair<int,int>>{{1,0},{-1,0}};
    string owner = board[from_row][from_col].at("owner");
    for(auto [dx, dy] : directions){
        int nx = x + dx, ny = y + dy;
        while (nx >= 0 && nx < rows && ny >= 0 && ny < cols){
            if (nx == from_row && ny == from_col) { nx += dx; ny += dy; continue; }
            if (board[nx][ny].empty()){
                if (!is_opponent_score_cell_dyn(nx, ny, owner)) moves.push_back({"move", {from_row, from_col}, {nx, ny}, {}, ""});
            } else if (board[nx][ny].at("side") == "river"){
                explore_river_move(board, nx, ny, moves, rows, cols, from_row, from_col, visited);
                break;
            } else break;
            nx += dx; ny += dy;
        }
    }
}

// Explore river for pushes
static void explore_river_push(const vector<vector<map<string, string>>>& board,int push_row, int push_col, vector<Move>& moves, int rows, int cols,int from_row, int from_col,
                               set<pair<int,int>>& visited,int or_push_row,int or_push_col, int or_from_row, int or_from_col){
    if (visited.count({push_row, push_col})) return;
    visited.insert({push_row, push_col});
    if (board[from_row][from_col].at("side") != "river") return;
    string orientation = board[from_row][from_col].at("orientation");
    vector<pair<int,int>> directions = (orientation == "horizontal") ? vector<pair<int,int>>{{0,1},{0,-1}}
                                                                      : vector<pair<int,int>>{{1,0},{-1,0}};
    string owner_to_be_pushed = board[or_push_row][or_push_col].at("owner");
    for(auto [dx, dy] : directions){
        int nx = push_row + dx, ny = push_col + dy;
        while (nx >= 0 && nx < rows && ny >= 0 && ny < cols) {
            if(nx == from_row && ny == from_col){ nx += dx; ny += dy; continue; }
            if(board[nx][ny].empty()){
                if(!is_opponent_score_cell_dyn(nx, ny, owner_to_be_pushed)) moves.push_back({"push",{or_from_row, or_from_col},{or_push_row, or_push_col},{nx, ny},""});
                else break;
            } else if(board[nx][ny].at("side") == "river"){
                explore_river_push(board, nx, ny, moves, rows, cols, nx, ny, visited,or_push_row,or_push_col, or_from_row, or_from_col);
                break;
            } else break;
            nx += dx; ny += dy;
        }
    }
}

// Generate all moves for a piece at (row,col)
static vector<Move> generate_all_moves(const vector<vector<map<string, string>>>& board,int row,int col){
    vector<Move> moves;
    int rows = board.size();
    int cols = board[0].size();
    if(board[row][col].empty()) return moves;
    string owner = board[row][col].at("owner");
    string side = board[row][col].at("side");
    string orientation = (side == "river") ? board[row][col].at("orientation") : "";
    vector<pair<int,int>> dirs={{1,0},{-1,0},{0,1},{0,-1}};
    set<pair<int,int>> visited;
    // MOVE
    for(auto [dx,dy]: dirs){
        int nx = row + dx, ny = col + dy;
        if (nx < 0 || nx >= rows || ny < 0 || ny >= cols) continue;
        if(is_opponent_score_cell_dyn(nx,ny,owner)) continue;
        if(board[nx][ny].empty()) moves.push_back({"move", {row,col}, {nx,ny}, {}, ""});
        else if(board[nx][ny].at("side")=="river"){
            set<pair<int,int>> v2; // reset per-ray to avoid premature pruning
            explore_river_move(board, nx, ny, moves, rows, cols, row, col, v2);
        }
    }
    // PUSH
    for(auto [dx,dy]: dirs){
        int nx=row+dx, ny=col+dy, nx2=row+2*dx, ny2=col+2*dy;
        if(nx<0||nx>=rows||ny<0||ny>=cols) continue;
        if(is_opponent_score_cell_dyn(nx,ny,owner)) continue;
        if(board[nx][ny].empty() || board[nx][ny].at("side")!="stone") continue;
        if(nx2<0||nx2>=rows||ny2<0||ny2>=cols) continue;
        if(is_opponent_score_cell_dyn(nx2,ny2,board[nx][ny].at("owner"))) continue;
        if(!board[nx2][ny2].empty()) continue;
        if(side=="stone") moves.push_back({"push", {row,col}, {nx,ny}, {nx2,ny2}, ""});
        else if(side=="river"){
            set<pair<int,int>> visited_push; explore_river_push(board,nx,ny,moves,rows,cols,row,col,visited_push,nx,ny,row,col);
        }
    }
    // FLIP
    if(side=="stone"){
        if(!allows_flow_into_opponent_score_after_orientation(board,row,col,owner,"horizontal")) moves.push_back({"flip",{row,col},{row,col},{},"horizontal"});
        if(!allows_flow_into_opponent_score_after_orientation(board,row,col,owner,"vertical")) moves.push_back({"flip",{row,col},{row,col},{},"vertical"});
    } else {
        moves.push_back({"flip",{row,col},{row,col},{},""});
    }
    // ROTATE
    if(side=="river"){
        string new_ori = (orientation=="horizontal")?"vertical":"horizontal";
        if(!allows_flow_into_opponent_score_after_orientation(board,row,col,owner,new_ori)) moves.push_back({"rotate",{row,col},{row,col},{},new_ori});
    }
    return moves;
}

// Apply a move to return a new board
static vector<vector<map<string, string>>> update_board(const vector<vector<map<string, string>>>& board, const Move& move){
    auto nb = board;
    int sy = move.from[0], sx = move.from[1];
    if(nb[sy][sx].empty()) return nb;
    if(move.action=="move"){
        int ey = move.to[0], ex = move.to[1]; nb[ey][ex] = nb[sy][sx]; nb[sy][sx].clear();
    } else if(move.action=="push"){
        int py = move.to[0], px = move.to[1]; (void)py; (void)px;
        int dy = move.pushed_to[0], dx = move.pushed_to[1];
        int ny = move.to[0], nx = move.to[1];
        string pusher_side = nb[sy][sx].at("side");
        nb[dy][dx] = nb[ny][nx]; nb[ny][nx].clear();
        nb[ny][nx] = nb[sy][sx];
        if(pusher_side=="river"){ nb[ny][nx]["side"] = "stone"; nb[ny][nx].erase("orientation"); }
        nb[sy][sx].clear();
    } else if(move.action=="flip"){
        string current_side = nb[sy][sx].at("side");
        if(current_side=="stone"){ nb[sy][sx]["side"] = "river"; nb[sy][sx]["orientation"] = move.orientation; }
        else { nb[sy][sx]["side"] = "stone"; nb[sy][sx].erase("orientation"); }
    } else if(move.action=="rotate"){
        nb[sy][sx]["orientation"] = (nb[sy][sx].at("orientation")=="horizontal")?"vertical":"horizontal";
    }
    return nb;
}

// Count stones in own score area
static int count_in_score_area(const vector<vector<map<string, string>>>& board,string owner){
    int r = (owner=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
    int cnt = 0; for(int c : G_SCORE_COLS){ if(r>=0 && r<G_ROWS && c>=0 && c<G_COLS){ const auto& cell = board[r][c]; if(!cell.empty() && cell.at("owner")==owner && cell.at("side")=="stone") cnt++; } }
    return cnt;
}

// BFS over rivers: from a river tile (r,c), can we reach an EMPTY cell in own scoring row/cols via flow?
static bool river_reaches_own_scoring_empty(const vector<vector<map<string,string>>>& board, int r, int c, const string& owner){
    int R=board.size(), C=board[0].size();
    auto inb = [&](int x,int y){ return x>=0 && x<R && y>=0 && y<C; };
    int goal_row = (owner=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
    deque<pair<int,int>> dq; dq.push_back({r,c});
    set<pair<int,int>> vis; vis.insert({r,c});
    while(!dq.empty()){
        auto [x,y]=dq.front(); dq.pop_front();
        if(board[x][y].empty() || board[x][y].at("side")!="river") continue;
        string ori = ""; auto itO=board[x][y].find("orientation"); if(itO!=board[x][y].end()) ori=itO->second;
        vector<pair<int,int>> rays = (ori=="horizontal") ? vector<pair<int,int>>{{0,1},{0,-1}} : vector<pair<int,int>>{{1,0},{-1,0}};
        for(auto [dx,dy]:rays){
            int nx=x+dx, ny=y+dy;
            while(inb(nx,ny)){
                if(board[nx][ny].empty()){
                    if(nx==goal_row && col_in_score(ny)) return true; // empty scoring cell reachable
                    nx+=dx; ny+=dy; continue;
                }
                if(board[nx][ny].at("side")=="river"){
                    if(!vis.count({nx,ny})){ vis.insert({nx,ny}); dq.push_back({nx,ny}); }
                    break; // hit another river tile
                }
                break; // stone blocks
            }
        }
    }
    return false;
}

// Is a stone in scoring cell actually CONNECTED to scoring via rivers (so it's convertible/valid)?
static bool is_stone_connected_via_rivers(const vector<vector<map<string,string>>>& board, int sr, int sc, const string& owner){
    if(!(sr>=0 && sr<(int)board.size() && sc>=0 && sc<(int)board[0].size())) return false;
    if(board[sr][sc].empty() || board[sr][sc].at("side")!="stone" || board[sr][sc].at("owner")!=owner) return false;
    int goal_row = (owner=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
    if(!(sr==goal_row && col_in_score(sc))) return false; // must be in scoring cell first
    // Check adjacent rivers and whether they can reach an empty scoring cell (flow continuity)
    const pair<int,int> dirs[4]={{1,0},{-1,0},{0,1},{0,-1}};
    for(auto [dx,dy]:dirs){
        int nx=sr+dx, ny=sc+dy;
        if(nx<0||nx>=(int)board.size()||ny<0||ny>=(int)board[0].size()) continue;
        if(!board[nx][ny].empty() && board[nx][ny].at("side")=="river"){
            if(river_reaches_own_scoring_empty(board, nx, ny, owner)) return true;
        }
    }
    return false;
}

// Count stones in scoring cells that are CONNECTED via rivers to scoring (stricter than occupancy)
static int count_connected_scored(const vector<vector<map<string,string>>>& board, const string& owner){
    int r = (owner=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
    int cnt=0; for(int c: G_SCORE_COLS){ if(r>=0 && r<(int)board.size() && c>=0 && c<(int)board[0].size()){
        if(!board[r][c].empty() && board[r][c].at("owner")==owner && board[r][c].at("side")=="stone"){
            if(is_stone_connected_via_rivers(board, r, c, owner)) cnt++;
        }
    }}
    return cnt;
}

// Check if scoring zone is blocked by own stones (all scoring cells occupied)
static bool is_scoring_zone_blocked(const vector<vector<map<string,string>>>& board, const string& owner){
    int goal_row = (owner=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
    if(goal_row < 0 || goal_row >= (int)board.size()) return false;

    int occupied_count = 0;
    for(int c : G_SCORE_COLS){
        if(c >= 0 && c < (int)board[0].size()){
            if(!board[goal_row][c].empty() &&
               board[goal_row][c].at("owner")==owner &&
               board[goal_row][c].at("side")=="stone"){
                occupied_count++;
            }
        }
    }
    // If all (or most) scoring cells are occupied by own stones, zone is blocked
    return occupied_count >= (int)G_SCORE_COLS.size();
}

// Count how many empty cells are in scoring zone (more empty = better for scoring later)
static int count_empty_in_scoring_zone(const vector<vector<map<string,string>>>& board, const string& owner){
    int goal_row = (owner=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
    if(goal_row < 0 || goal_row >= (int)board.size()) return 0;

    int empty_count = 0;
    for(int c : G_SCORE_COLS){
        if(c >= 0 && c < (int)board[0].size()){
            if(board[goal_row][c].empty()){
                empty_count++;
            }
        }
    }
    return empty_count;
}

// Terminal test
static bool is_terminal(const vector<vector<map<string, string>>>& board){
    int need = (G_COLS<=12)?4:((G_COLS<=14)?5:6);
    return count_in_score_area(board,"circle")>=need || count_in_score_area(board,"square")>=need;
}

// Simple evaluation (Phase 1): emphasize scoring stones
static int evaluate_function(const vector<vector<map<string, string>>>& board, const string& owner){
    // Stronger but still cheap heuristic for MCTS leafs
    const string opp = (owner=="circle")? "square" : "circle";
    const int R = (int)board.size();
    const int C = (int)board[0].size();
    auto inside = [&](int x,int y){ return x>=0 && x<R && y>=0 && y<C; };

    // Features
    // Engine-scoring: occupancy in scoring cells
    int scored_self = count_in_score_area(board, owner);
    int scored_opp  = count_in_score_area(board, opp);
    int sum_dist_self = 0, sum_dist_opp = 0;
    int near1_self = 0, near2_self = 0, near1_opp = 0, near2_opp = 0;
    int mobility_self = 0, mobility_opp = 0;       // adj empties
    int synergy_self = 0, synergy_opp = 0;         // adj friendly rivers (oriented)
    int can_next_self = 0, can_next_opp = 0;       // can finish in one move (approx)
    int path_partial_self = 0, path_partial_opp = 0; // within two moves heuristic
    int ride_gain_self = 0, ride_gain_opp = 0;     // distance reduction in one move

    int row_self = (owner=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
    int row_opp  = (owner=="circle") ? bottom_score_row_dynamic() : top_score_row_dynamic();
    const pair<int,int> dirs[4]={{1,0},{-1,0},{0,1},{0,-1}};

    for(int i=0;i<R;i++){
        for(int j=0;j<C;j++){
            if(board[i][j].empty()) continue;
            const string& who  = board[i][j].at("owner");
            const string& side = board[i][j].at("side");
            if(side!="stone") continue;

            // Scored stones handled by scored_self/scored_opp above

            // Current distance and near-term threats
            int d = min_distance_from_scoring_area(who, i, j);
            if(who==owner){
                sum_dist_self += d;
                if(d==1) near1_self++; else if(d==2) near2_self++;
            } else {
                sum_dist_opp += d;
                if(d==1) near1_opp++; else if(d==2) near2_opp++;
            }

            // Mobility and synergy
            int mob_inc = 0, syn = 0;
            for(auto [dx,dy]:dirs){
                int nx=i+dx, ny=j+dy; if(!inside(nx,ny)) continue;
                if(board[nx][ny].empty()){
                    if(!is_opponent_score_cell_dyn(nx, ny, who)) mob_inc++;
                } else if(board[nx][ny].at("side")=="river" && board[nx][ny].at("owner")==who){
                    syn += (board[nx][ny].at("orientation")=="vertical"?2:1);
                }
            }
            if(who==owner){ mobility_self += mob_inc; synergy_self += syn; }
            else          { mobility_opp  += mob_inc; synergy_opp  += syn; }

            // One-move potential: finish now or reduce distance by riding a river/step
            int best_next = best_next_distance_to_score(board, who, i, j);
            if(best_next==0){ if(who==owner) ++can_next_self; else ++can_next_opp; }
            else if(best_next<=2){ if(who==owner) ++path_partial_self; else ++path_partial_opp; }
            if(best_next<100){
                int gain = max(0, d - best_next);
                if(who==owner) ride_gain_self += gain; else ride_gain_opp += gain;
            }
        }
    }

    // Side-pull toward scoring columns when within 2 rows (column alignment)
    auto row_gap_to_goal = [&](const string& pov, int r){ return abs(r - ((pov=="circle")? top_score_row_dynamic() : bottom_score_row_dynamic())); };
    auto nearest_scoring_col_dist = [&](int c){ int best=100; for(int sc:G_SCORE_COLS) best=min(best, abs(sc-c)); return best; };
    int side_pull_self=0, side_pull_opp=0;
    int cap1 = (int)G_SCORE_COLS.size(); int cap2 = max(1,(int)G_SCORE_COLS.size()/2);
    for(int i=0;i<R;i++){
        for(int j=0;j<C;j++){
            if(board[i][j].empty() || board[i][j].at("side")!="stone") continue;
            const string& who = board[i][j].at("owner");
            int rd = row_gap_to_goal(who, i);
            if(rd<=2){
                int cd = nearest_scoring_col_dist(j);
                int pull = (rd==1 ? (cap1 - cd) : (cap2 - cd));
                if(who==owner) side_pull_self += pull; else side_pull_opp += pull;
            }
        }
    }

    // River network value: coverage and goal reach from river components
    auto river_network_value = [&](const string& pov){
        int Rn = (int)board.size(), Cn = (int)board[0].size();
        vector<vector<char>> seen(Rn, vector<char>(Cn, 0));
        auto inb = [&](int x,int y){ return x>=0 && x<Rn && y>=0 && y<Cn; };
        int goal_row = (pov=="circle") ? top_score_row_dynamic() : bottom_score_row_dynamic();
        long long cov_sum = 0; long long goal_sum = 0;
        const pair<int,int> dirs4[4]={{1,0},{-1,0},{0,1},{0,-1}};
        for(int r=0;r<Rn;r++){
            for(int c=0;c<Cn;c++){
                if(board[r][c].empty() || board[r][c].at("side")!="river" || seen[r][c]) continue;
                // BFS over connected rivers (any owner) via flow adjacencies
                vector<pair<int,int>> comp_rivers;
                int comp_cov = 0; int comp_goal = 0;
                deque<pair<int,int>> dq; dq.push_back({r,c}); seen[r][c]=1;
                while(!dq.empty()){
                    auto [x,y]=dq.front(); dq.pop_front();
                    comp_rivers.push_back({x,y});
                    string ori = board[x][y].at("orientation");
                    vector<pair<int,int>> rays = (ori=="horizontal") ? vector<pair<int,int>>{{0,1},{0,-1}}
                                                                     : vector<pair<int,int>>{{1,0},{-1,0}};
                    for(auto [dx,dy]:rays){
                        int nx=x+dx, ny=y+dy;
                        // Slide along ray; note empty cells are landings, rivers continue BFS
                        while(inb(nx,ny)){
                            if(board[nx][ny].empty()){
                                if(!is_opponent_score_cell_dyn(nx,ny,pov)){
                                    // landing: count coverage and goal proximity (component-local)
                                    comp_cov++;
                                    if(col_in_score(ny) && std::abs(nx - goal_row) <= 2) comp_goal++;
                                }
                                nx+=dx; ny+=dy; continue;
                            }
                            if(board[nx][ny].at("side")=="river"){
                                if(!seen[nx][ny]){ seen[nx][ny]=1; dq.push_back({nx,ny}); }
                                break; // stop this ray at first river
                            }
                            break; // stone stops
                        }
                    }
                }
                // Count friendly stones adjacent to this component's rivers
                int adj_stones = 0;
                std::set<pair<int,int>> uniq;
                for(auto &rc : comp_rivers){
                    for(auto [dx,dy]:dirs4){ int sx=rc.first+dx, sy=rc.second+dy; if(!inb(sx,sy)) continue; if(board[sx][sy].empty()) continue; if(board[sx][sy].at("side")=="stone" && board[sx][sy].at("owner")==pov){ uniq.insert({sx,sy}); } }
                }
                adj_stones = (int)uniq.size();

                // Cap component contributions and accumulate weighted by adjacent stones
                comp_cov = std::min(comp_cov, 60);
                comp_goal = std::min(comp_goal, 20);
                cov_sum += (long long)adj_stones * comp_cov;
                goal_sum += (long long)adj_stones * comp_goal;
            }
        }
        // Clamp totals to avoid overflow while preserving sign
        if(cov_sum > 2000000000LL) cov_sum = 2000000000LL;
        if(goal_sum > 2000000000LL) goal_sum = 2000000000LL;
        return pair<int,int>{(int)cov_sum, (int)goal_sum};
    };
    auto rn_self = river_network_value(owner);
    auto rn_opp  = river_network_value(opp);

    // Weights (tuned to keep eval in ~[-1e4,1e4] typical range)
    const int W_SCORED        = 2300;
    const int W_DIST_DIFF     =  -80;
    const int W_NEAR1_SELF    =  600;
    const int W_NEAR2_SELF    =  280;
    const int W_NEAR1_OPP     =  720;
    const int W_NEAR2_OPP     =  340;
    const int W_MOBILITY      =   25;
    const int W_SYNERGY       =  200;  // stronger stone-river adjacency
    const int W_CANNEXT_SELF  = 2200;
    const int W_CANNEXT_OPP   = 1800;
    const int W_RIDE_SELF     =   40;
    const int W_RIDE_OPP      =   18;
    const int W_SIDE_PULL     =  140;
    const int W_RNET_COV      =    6;  // river network coverage
    const int W_RNET_GOAL     =   36;  // network landings near scoring
    const int W_SCORECOL_RIV  =  800;  // rivers in scoring columns
    const int W_PATH_COMPLETE = 1400;  // additional boost for finish-in-1
    const int W_PATH_PARTIAL  =  600;  // heuristic two-move connectivity
    const int W_BLOCK_OPP     = 1500;  // block opponent's path

    // NEW: Scoring zone clearance weights
    const int W_ZONE_BLOCKED  = -4000; // Heavy penalty for having scoring zone blocked
    const int W_ZONE_CLEAR    =  4000; // Huge bonus for having scoring zone clear
    const int W_EMPTY_SLOTS   =   800; // Reward empty scoring slots (room to score)

    long long eval = 0;

    // NEW: Check scoring zone blockage status (early game focus)
    bool zone_self_blocked = is_scoring_zone_blocked(board, owner);
    bool zone_opp_blocked  = is_scoring_zone_blocked(board, opp);
    int empty_self         = count_empty_in_scoring_zone(board, owner);
    int empty_opp          = count_empty_in_scoring_zone(board, opp);

    // Strategic evaluation of scoring zones
    if(zone_self_blocked && !zone_opp_blocked){
        eval += W_ZONE_BLOCKED;  // We're blocked, opponent isn't - BAD!
    } else if(!zone_self_blocked && zone_opp_blocked){
        eval += W_ZONE_CLEAR;    // We cleared ours, they're blocked - GOOD!
    }

    // Reward having empty slots in scoring zone (means we can score more stones)
    eval += W_EMPTY_SLOTS * (empty_self - empty_opp);

    // Count scored stones, but with reduced weight if zone is blocked
    // (because blocked stones aren't making progress toward winning)
    if(zone_self_blocked){
        // Scoring zone blocked: these stones are BLOCKING, not scoring
        // Reduce their value significantly
        eval += (W_SCORED / 4) * (scored_self - scored_opp);
    } else {
        // Scoring zone clear: these are legitimate scoring stones
        eval += W_SCORED * (scored_self - scored_opp);
    }
    eval += W_DIST_DIFF * (sum_dist_self - sum_dist_opp);
    eval += W_NEAR1_SELF * near1_self + W_NEAR2_SELF * near2_self;
    eval -= W_NEAR1_OPP  * near1_opp  + W_NEAR2_OPP  * near2_opp;
    eval += W_MOBILITY * (mobility_self - mobility_opp);
    eval += W_SYNERGY  * (synergy_self - synergy_opp);
    eval += W_CANNEXT_SELF * can_next_self;
    eval -= W_CANNEXT_OPP  * can_next_opp;
    eval += W_PATH_COMPLETE * can_next_self;                 // extra boost for complete path
    eval += W_PATH_PARTIAL  * path_partial_self;
    eval -= W_BLOCK_OPP     * (can_next_opp + path_partial_opp);
    eval += W_RIDE_SELF * ride_gain_self;
    eval -= W_RIDE_OPP  * ride_gain_opp;
    eval += W_SIDE_PULL * (side_pull_self - side_pull_opp);
    eval += W_RNET_COV  * (rn_self.first - rn_opp.first);
    eval += W_RNET_GOAL * (rn_self.second - rn_opp.second);
    // Scoring column rivers (vertical favored)
    int scr_self = count_scoring_col_rivers_for(board, owner);
    int scr_opp  = count_scoring_col_rivers_for(board, opp);
    eval += W_SCORECOL_RIV * (scr_self - scr_opp);

    // Scoring-zone blockage penalty: if we're near full occupancy but cannot finish next, encourage clearing
    int need = (G_COLS<=12)?4:((G_COLS<=14)?5:6);
    bool self_blocked = (scored_self >= need-1) && (can_next_self==0);
    bool opp_blocked  = (scored_opp  >= need-1) && (can_next_opp==0);
    const int BLOCKING_PENALTY = 3000;
    if(self_blocked && !opp_blocked) eval -= BLOCKING_PENALTY;
    else if(!self_blocked && opp_blocked) eval += BLOCKING_PENALTY;

    if (eval > 2000000000LL) eval = 2000000000LL;
    if (eval < -2000000000LL) eval = -2000000000LL;
    return (int)eval;
}

// Instant win detection and blocking
static Move check_for_instant_win(const vector<vector<map<string, string>>>& board,string owner,const map<pair<int,int>,vector<Move>>& moves){
    int need = (G_COLS<=12)?4:((G_COLS<=14)?5:6);
    int n=board.size(), m=board[0].size(); (void)m;
    for(int i=0;i<n;i++) {
        for(int j=0;j<(int)board[0].size();j++){
            if(!board[i][j].empty() && board[i][j].at("owner")!=owner) continue;
            auto it = moves.find({i,j}); if(it==moves.end()) continue;
            for(const auto& mv : it->second){ auto nb = update_board(board,mv); if(count_in_score_area(nb,owner)>=need) return mv; }
        }
    }
    return {"",{}, {}, {}, ""};
}

static vector<pair<int,int>> trace_winning_path(const vector<vector<map<string, string>>>& board,const Move& win_move,const string& opponent);
static bool simulate_move_path(const vector<vector<map<string, string>>>& board,int current_row,int current_col,int target_row,int target_col,vector<pair<int,int>>& path_stack,set<pair<int,int>>& visited,int dx1,int dy1);
static Move find_path_blocking_moves(const vector<vector<map<string,string>>>& board,const vector<pair<int,int>>& path,const string& owner,const map<pair<int,int>,vector<Move>>& moves);

static Move find_blocking_move(const vector<vector<map<string,string>>>& board,Move win_move,string owner,const map<pair<int,int>,vector<Move>>& moves){
    string opponent = (owner=="circle")?"square":"circle";
    auto path = trace_winning_path(board, win_move, opponent);
    if(path.empty()) return {"",{},{},{},""};
    return find_path_blocking_moves(board, path, owner, moves);
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
    } else if(!board[row][col].empty() && board[row][col].at("side")=="river"){ if(board[row][col].at("owner")==owner){
            // Instead of fabricating a flip, search a legal flip from generated moves at (row,col)
            auto it = moves.find({row,col});
            if(it!=moves.end()){
                for(const auto& mv : it->second){ if(mv.action=="flip") return mv; }
            }
        } }
    }
    return {"",{},{},{},""};
}

// ------------------------------------------------------------
// StudentAgent with MCTS (PUCT + TT + Zobrist)
// ------------------------------------------------------------
class StudentAgent {
public:
    explicit StudentAgent(string s) : side(std::move(s)), gen(rd()){}

    Move choose(const vector<vector<map<string, string>>>& board,int row,int col,const vector<int>& score_cols,float current_player_time,float opponent_time){
        (void)opponent_time;
        G_ROWS=row; G_COLS=col; G_SCORE_COLS=score_cols;

        // Precompute our moves first (cheaper), then opponent only if needed
        int n=board.size(), m=board[0].size();
        map<pair<int,int>,vector<Move>> moves_self;
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                if(!board[i][j].empty() && board[i][j].at("owner")==side){
                    moves_self[{i,j}] = generate_all_moves(board,i,j);
                }
            }
        }

        // Tactical guardrails: instant win or block
        Move iw = check_for_instant_win(board, side, moves_self);
        if(iw.action!="") return convert_move_format(iw);
        string opp = (side=="circle")?"square":"circle";
        map<pair<int,int>,vector<Move>> moves_opp;
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                if(!board[i][j].empty() && board[i][j].at("owner")==opp){
                    moves_opp[{i,j}] = generate_all_moves(board,i,j);
                }
            }
        }
        Move ow = check_for_instant_win(board, opp, moves_opp);
        if(ow.action!=""){ Move block = find_blocking_move(board, ow, side, moves_self); if(block.action!="") return convert_move_format(block); }

        // NEW: Greedy zone clearance - if scoring zone is blocked, prioritize clearing it
        if(is_scoring_zone_blocked(board, side)){
            Move best_clear_move = {"",{},{},{},""};
            int best_clear_delta = 0;
            int current_blocked = count_in_score_area(board, side);

            for(const auto& kv : moves_self){
                const auto& v = kv.second;
                for(const auto& mv : v){
                    auto nb = update_board(board, mv);
                    int new_blocked = count_in_score_area(nb, side);
                    int clear_delta = current_blocked - new_blocked;  // Positive if we cleared stones
                    if(clear_delta > best_clear_delta){
                        best_clear_delta = clear_delta;
                        best_clear_move = mv;
                    }
                }
            }
            // If we found a move that clears at least one stone from scoring zone, play it!
            if(best_clear_delta > 0){
                return convert_move_format(best_clear_move);
            }
        }

        // Greedy scoring increase: if any move increases stones in scoring area (engine scoring), do it immediately
        int prev_scored = count_in_score_area(board, side);
        Move best_score_move = {"",{},{},{},""};
        int best_delta = 0;
        for(const auto& kv : moves_self){
            const auto& v = kv.second;
            for(const auto& mv : v){
                auto nb = update_board(board, mv);
                int ns = count_in_score_area(nb, side);
                int delta = ns - prev_scored;
                if(delta > best_delta){ best_delta = delta; best_score_move = mv; }
            }
        }
        if(best_delta > 0){
            return convert_move_format(best_score_move);
        }

        // Phase-based time budget: early -> build rivers fast; mid -> balance; late -> precise
        auto danger_eval = [&](){
            for(int i=0;i<n;i++){
                for(int j=0;j<m;j++){
                    if(board[i][j].empty() || board[i][j].at("side")!="stone") continue;
                    if(board[i][j].at("owner")!=opp) continue;
                    int bestn = best_next_distance_to_score(board, opp, i, j);
                    if(bestn==0) return true; // opponent can finish in one
                }
            }
            return false;
        };
        auto offense_eval = [&](){
            for(int i=0;i<n;i++){
                for(int j=0;j<m;j++){
                    if(board[i][j].empty() || board[i][j].at("side")!="stone") continue;
                    if(board[i][j].at("owner")!=side) continue;
                    if(best_next_distance_to_score(board, side, i, j)==0) return true; // we can finish in one
                }
            }
            return false;
        };
        auto phase_eval = [&](){
            int rivers=0, stones_near=0;
            for(int i=0;i<n;i++) for(int j=0;j<m;j++) if(!board[i][j].empty()){
                if(board[i][j].at("side")=="river") rivers++;
                if(board[i][j].at("side")=="stone" && (min_distance_from_scoring_area(board[i][j].at("owner"), i, j)<=2)) stones_near++;
            }
            if(rivers < 10 && stones_near < 4) return 0;   // early
            if(rivers < 24 && stones_near < 8) return 1;    // mid
            return 2;                                       // late
        };
        bool danger = danger_eval();
        bool offense = offense_eval();
        int phase = phase_eval();
        double base_phase = (phase==0 ? 0.30 : (phase==1 ? 0.50 : 0.80));
        double base = base_phase; // override simple base by phase heuristic
        double bonus = (danger ? 0.25 : 0.0) + (offense ? 0.15 : 0.0);
        double time_limit = std::min(0.9, base + bonus);
        Move best = mcts_decide(board, side, n, m, score_cols, time_limit);
        if(best.action==""){
            // fallback: first legal move we own
            for(int i=0;i<n;i++) for(int j=0;j<m;j++) if(!board[i][j].empty() && board[i][j].at("owner")==side){ auto v=generate_all_moves(board,i,j); if(!v.empty()){ best=v[0]; break; } }
        }
        return convert_move_format(best);
    }

private:
    string side;
    random_device rd; mt19937 gen;

    // Move-format conversion (internal row,col -> external col,row)
    Move convert_move_format(const Move& m){ Move r; r.action=m.action; if(!m.from.empty()) r.from={m.from[1],m.from[0]}; if(!m.to.empty()) r.to={m.to[1],m.to[0]}; if(!m.pushed_to.empty()) r.pushed_to={m.pushed_to[1],m.pushed_to[0]}; r.orientation=m.orientation; return r; }

    struct MCTSNode{
        int N=0; double W=0.0; double Q=0.0;
        uint64_t hash=0ULL; string to_move; vector<vector<map<string,string>>> state;
        vector<Move> moves; vector<double> priors; vector<MCTSNode*> children; vector<int> order_by_prior; int expanded_count=0; bool terminal=false;
    };

    unordered_map<uint64_t,MCTSNode*> tt;
    bool zob_init=false; int zob_r=-1, zob_c=-1; vector<vector<array<uint64_t,6>>> zob_cell; array<uint64_t,2> zob_side{{0ULL,0ULL}};

    void init_zobrist(int r,int c){ if(zob_init && r==zob_r && c==zob_c) return; std::mt19937_64 rng(0xC0FFEE1234ULL ^ (uint64_t)r<<32 ^ (uint64_t)c); zob_cell.assign(r, vector<array<uint64_t,6>>(c)); for(int i=0;i<r;i++) for(int j=0;j<c;j++) for(int k=0;k<6;k++) zob_cell[i][j][k]=rng(); zob_side[0]=rng(); zob_side[1]=rng(); zob_init=true; zob_r=r; zob_c=c; }
    static inline int piece_code(const map<string,string>& cell){ if(cell.empty()) return -1; const string& own=cell.at("owner"); const string& s=cell.at("side"); if(s=="stone") return (own=="circle"?0:1); const string& o=cell.at("orientation"); if(own=="circle") return (o=="horizontal"?2:3); return (o=="horizontal"?4:5); }
    uint64_t hash_board(const vector<vector<map<string,string>>>& b, const string& to_move){ init_zobrist((int)b.size(), (int)b[0].size()); uint64_t h=0ULL; int R=b.size(), C=b[0].size(); for(int i=0;i<R;i++) for(int j=0;j<C;j++){ int pc=piece_code(b[i][j]); if(pc>=0) h^=zob_cell[i][j][pc]; } h ^= (to_move=="circle"?zob_side[0]:zob_side[1]); return h; }
    static inline string opponent_of(const string& s){ return (s=="circle"?"square":"circle"); }

    // Count stones that can be within two moves to score: best_next_distance <= 1
    int count_within_two(const vector<vector<map<string,string>>>& b, const string& who){
        int R=b.size(), C=b[0].size(); int cnt=0;
        for(int i=0;i<R;i++) for(int j=0;j<C;j++){
            if(b[i][j].empty() || b[i][j].at("side")!="stone") continue;
            if(b[i][j].at("owner")!=who) continue;
            int bn = best_next_distance_to_score(b, who, i, j);
            if(bn<=1) cnt++;
        }
        return cnt;
    }

    vector<Move> collect_moves(const vector<vector<map<string,string>>>& b, const string& who){
        int R=b.size();
        vector<Move> mv;
        for(int i=0;i<R;i++){
            for(int j=0;j<(int)b[0].size();j++){
                if(!b[i][j].empty() && b[i][j].at("owner")==who){
                    auto v=generate_all_moves(b,i,j);
                    mv.insert(mv.end(), v.begin(), v.end());
                }
            }
        }
        return mv;
    }
    double leaf_value(const vector<vector<map<string,string>>>& b, const string& root_side){
        int need=(G_COLS<=12)?4:((G_COLS<=14)?5:6);
        if(count_in_score_area(b,root_side)>=need) return 1.0;
        if(count_in_score_area(b,opponent_of(root_side))>=need) return -1.0;
        if(G_FAST_EVAL){
            // Lightweight shape based value under tight time
            ShapeInfo si = shape_value(b, root_side);
            double v = si.vshape; // typically small integer-scale
            // squash to [-1,1]
            v = std::max(-4.0, std::min(4.0, v));
            v /= 4.0;
            if(!std::isfinite(v)) v = 0.0;
            return v;
        } else {
            double v=(double)evaluate_function(b, root_side);
            v/=10000.0;
            if(!std::isfinite(v)) v = 0.0;
            if(v>1.0) v=1.0; if(v<-1.0) v=-1.0;
            return v;
        }
    }

    Move mcts_decide(const vector<vector<map<string,string>>>& board,
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
        } else {
            root = new MCTSNode();
            root->state = board;
            root->to_move = to_move;
            root->hash = h;
            root->terminal = is_terminal(board);
            root->moves = collect_moves(board, to_move);
            int M = (int)root->moves.size();
            // Heuristic policy priors at root using V_shape deltas and defensive gating
            root->priors.assign(M, 0.0);
            if(M>0){
                const double tau = 1.5; // temperature for softmax over V_shape deltas
                double sumw = 0.0;
                // Dynamic cap of priors by time budget for responsiveness
                int max_priors_cap = 64;
                if(time_limit < 0.4) max_priors_cap = 24; else if(time_limit < 0.6) max_priors_cap = 32;
                int max_priors = std::min(M, max_priors_cap);
                const string opp = (root_side=="circle")?"square":"circle";
                ShapeInfo parent_si = shape_value(board, root_side);
                int parent_scr_riv = count_scoring_col_rivers_for(board, root_side);
                int need = (G_COLS<=12)?4:((G_COLS<=14)?5:6);
                int parent_n_opp = count_in_score_area(board, opp);
                int parent_two = count_within_two(board, root_side);
                for(int i=0;i<M;i++){
                    double w;
                    if(i < max_priors){
                        auto nb = update_board(board, root->moves[i]);
                        ShapeInfo child_si = shape_value(nb, root_side);
                        double gain = (child_si.vshape - parent_si.vshape) / tau;
                        // Defensive gating: bias to reduce opponent finish-in-1 without over-randomizing
                        double gate = 0.0;
                        if(parent_si.can_next_opp > 0){
                            int diff = parent_si.can_next_opp - child_si.can_next_opp; // >0 blocks threats
                            double mult = (parent_n_opp >= need-1 ? 1.5 : 1.0); // urgency if opp at 3/4
                            if(diff > 0) gate = mult * 0.6 * diff;  // reward blocking
                            else if(diff == 0) gate = mult * (-0.15); // slight downweight if not helping
                            else gate = mult * (-0.5 * (-diff));      // punish increasing opp threats
                        }
                        // Offense bonus: bias towards increasing our finish-in-1 count
                        int off_diff = child_si.can_next_self - parent_si.can_next_self;
                        double off_bonus = (off_diff > 0) ? (0.4 * off_diff) : 0.0;
                        // Path completion bias: more rivers in scoring columns
                        int child_scr_riv = count_scoring_col_rivers_for(nb, root_side);
                        double path_bonus = 0.3 * (double)(child_scr_riv - parent_scr_riv);
                        // Planning bonus: increase stones within 2 moves to score
                        int child_two = count_within_two(nb, root_side);
                        double plan_bonus = 0.25 * (double)(child_two - parent_two);
                        w = exp(gain + gate + off_bonus + path_bonus + plan_bonus);
                    } else {
                        w = 1.0; // fallback uniform weight for tail
                    }
                    if(!isfinite(w) || w<=0.0) w = 1e-6;
                    root->priors[i] = w; sumw += w;
                }
                if(sumw>0){ for(int i=0;i<M;i++) root->priors[i] /= sumw; }
                else { for(int i=0;i<M;i++) root->priors[i] = 1.0/(double)M; }
                // Add Dirichlet noise to encourage root exploration when moves look similar
                std::gamma_distribution<double> gamma(0.15, 1.0);
                vector<double> noise(M); double sumn=0.0;
                for(int i=0;i<M;i++){ noise[i]=gamma(gen); sumn+=noise[i]; }
                if(sumn>0){ for(int i=0;i<M;i++) noise[i]/=sumn; }
                double eps=0.05; double unif = 1.0/(double)M;
                for(int i=0;i<M;i++){
                    double n = noise[i];
                    double mixed = (1.0-eps)*root->priors[i] + eps*n;
                    // small smoothing with uniform to prevent zeros
                    root->priors[i] = 0.95*mixed + 0.05*unif;
                }
            }
            root->children.assign(M, nullptr);
            root->order_by_prior.resize(M);
            iota(root->order_by_prior.begin(), root->order_by_prior.end(), 0);
            sort(root->order_by_prior.begin(), root->order_by_prior.end(), [&](int a,int b){ return root->priors[a] > root->priors[b]; });
            tt[h] = root;
        }

        // Enable fast eval for tight budgets
        G_FAST_EVAL = (time_limit <= 0.5);
        const double c_puct = G_FAST_EVAL ? 1.0 : 1.2;
        const int K0 = 6;    // slightly more breadth engagement
        const double K1 = 2.0;

        auto start = chrono::high_resolution_clock::now();
        while(true){
            auto now = chrono::high_resolution_clock::now();
            if(chrono::duration<double>(now - start).count() >= time_limit) break;

            vector<pair<MCTSNode*, int>> path;
            MCTSNode* node = root;
            bool expanded = false;

            while(true){
                if(node->terminal) break;
                int M = (int)node->moves.size();
                if(M == 0) break;

                int K = std::max(K0, (int)floor(K1 * sqrt(std::max(1, node->N))));
                K = std::min(K, M);

                if(node->expanded_count < K){
                    int idx = node->order_by_prior[node->expanded_count++];
                    if(node->children[idx] == nullptr){
                        auto nb = update_board(node->state, node->moves[idx]);
                        string child_to_move = opponent_of(node->to_move);
                        uint64_t ch = hash_board(nb, child_to_move);
                        MCTSNode* child = nullptr;
                        auto jt = tt.find(ch);
                        if(jt != tt.end()){
                            child = jt->second;
                        } else {
                            child = new MCTSNode();
                            child->state = std::move(nb);
                            child->to_move = child_to_move;
                            child->hash = ch;
                            child->terminal = is_terminal(child->state);
                            if(!child->terminal){
                                child->moves = collect_moves(child->state, child->to_move);
                                int CM = (int)child->moves.size();
                                child->priors.assign(CM, CM>0 ? 1.0/(double)CM : 0.0);
                                child->children.assign(CM, nullptr);
                                child->order_by_prior.resize(CM);
                                iota(child->order_by_prior.begin(), child->order_by_prior.end(), 0);
                                sort(child->order_by_prior.begin(), child->order_by_prior.end(), [&](int a,int b){ return child->priors[a] > child->priors[b]; });
                            }
                            tt[ch] = child;
                        }
                        node->children[idx] = child;
                        double v = leaf_value(child->state, root_side);
                        path.push_back({node, idx});
                        for(auto &pe : path){
                            MCTSNode* pn = pe.first;
                            double add = (pn->to_move == root_side) ? v : -v; // sign by side-to-move
                            pn->N += 1;
                            pn->W += add;
                            pn->Q = pn->W / std::max(1, pn->N);
                        }
                        expanded = true;
                    }
                    break;
                }

                int best_i = -1; double best_score = -1e100;
                double sqrtN = sqrt((double)std::max(1, node->N));
                for(int k=0;k<node->expanded_count;k++){
                    int idx = node->order_by_prior[k];
                    MCTSNode* ch = node->children[idx];
                    if(!ch) continue;
                    double prior = node->priors[idx];
                    double q = ch->Q;
                    if(node->to_move != root_side) q = -q; // opponent minimizes root POV
                    double u = c_puct * prior * sqrtN / (1.0 + (double)ch->N);
                    double score = q + u;
                    if(score > best_score){ best_score = score; best_i = idx; }
                }
                if(best_i < 0) break;
                path.push_back({node, best_i});
                node = node->children[best_i];
            }

            if(!expanded){
                double v = leaf_value(node->state, root_side);
                for(auto &pe : path){
                    MCTSNode* pn = pe.first;
                    double add = (pn->to_move == root_side) ? v : -v;
                    pn->N += 1;
                    pn->W += add;
                    pn->Q = pn->W / std::max(1, pn->N);
                }
            }
        }

        int M = (int)root->moves.size();
        if(M == 0) return {"",{},{},{},""};
        int best_idx = -1; int bestN = -1; double bestQ = -1e18;
        int cap = root->expanded_count;
        bool debug = (std::getenv("DEBUG_STUDENT")!=nullptr);
        int print_top = std::min(cap, 5);
        struct Cand { int idx; int n; double q; double p; }; 
        vector<Cand> cands;
        for(int k=0;k<cap;k++){
            int idx = root->order_by_prior[k];
            MCTSNode* ch = root->children[idx];
            if(!ch) continue;
            int n = ch->N; double q = ch->Q;
            if(debug && k<print_top){ cands.push_back({idx, n, q, root->priors[idx]}); }
            if(n > bestN || (n==bestN && q>bestQ)){
                bestN = n; bestQ = q; best_idx = idx;
            }
        }
        if(best_idx < 0) best_idx = 0;
        if(debug){
            fprintf(stderr, "[student_cpp] root candidates (top %d by prior):\n", print_top);
            for(auto &c : cands){
                auto m = root->moves[c.idx];
                fprintf(stderr, "  idx=%d N=%d Q=%.3f P=%.3f action=%s from=[%d,%d] to=[%d,%d]\n",
                        c.idx, c.n, c.q, c.p,
                        m.action.c_str(),
                        (int)(m.from.size()?m.from[0]:-1), (int)(m.from.size()?m.from[1]:-1),
                        (int)(m.to.size()?m.to[0]:-1), (int)(m.to.size()?m.to[1]:-1));
            }
            auto bm = root->moves[best_idx];
            fprintf(stderr, "[student_cpp] choose idx=%d N=%d Q=%.3f action=%s\n", best_idx, bestN, bestQ, bm.action.c_str());
        }
        return root->moves[best_idx];
    }
};

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
