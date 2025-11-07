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
        else if(board[nx][ny].at("side")=="river") explore_river_move(board, nx, ny, moves, rows, cols, row, col, visited);
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

// Terminal test
static bool is_terminal(const vector<vector<map<string, string>>>& board){
    int need = (G_COLS<=12)?4:((G_COLS<=14)?5:6);
    return count_in_score_area(board,"circle")>=need || count_in_score_area(board,"square")>=need;
}

// Simple evaluation (Phase 1): emphasize scoring stones
static int evaluate_function(const vector<vector<map<string, string>>>& board, const string& owner){
    int self_n = count_in_score_area(board, owner);
    int opp_n  = count_in_score_area(board, (owner=="circle"?"square":"circle"));
    return 3000*(self_n - opp_n);
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
    } else if(!board[row][col].empty() && board[row][col].at("side")=="river"){ if(board[row][col].at("owner")==owner) return {"flip", {row, col}, {row, col}, {}, {}}; } }
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

        // Precompute all moves per piece for fast checks
        map<pair<int,int>,vector<Move>> moves_for_current;
        int n=board.size(), m=board[0].size();
        for(int i=0;i<n;i++) for(int j=0;j<m;j++) if(!board[i][j].empty()) moves_for_current[{i,j}] = generate_all_moves(board,i,j);

        // Tactical guardrails: instant win or block
        Move iw = check_for_instant_win(board, side, moves_for_current); if(iw.action!="") return convert_move_format(iw);
        string opp = (side=="circle")?"square":"circle";
        Move ow = check_for_instant_win(board, opp, moves_for_current);
        if(ow.action!=""){ Move block = find_blocking_move(board, ow, side, moves_for_current); if(block.action!="") return convert_move_format(block); }

        // Time budget (0.5–0.8s)
        double time_limit = 0.5 + 0.3 * std::min(1.0, std::max(0.0, (double)current_player_time / 20.0));
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

    vector<Move> collect_moves(const vector<vector<map<string,string>>>& b, const string& who){ int R=b.size(), C=b[0].size(); (void)C; vector<Move> mv; for(int i=0;i<R;i++) for(int j=0;j<(int)b[0].size();j++){ if(!b[i][j].empty() && b[i][j].at("owner")==who){ auto v=generate_all_moves(b,i,j); for(const auto& m:v){ auto nb=update_board(b,m); if(nb!=b) mv.push_back(m);} } } return mv; }
    double leaf_value(const vector<vector<map<string,string>>>& b, const string& root_side){ int need=(G_COLS<=12)?4:((G_COLS<=14)?5:6); if(count_in_score_area(b,root_side)>=need) return 1.0; if(count_in_score_area(b,opponent_of(root_side))>=need) return -1.0; double v=(double)evaluate_function(b, root_side); v/=10000.0; if(v>1.0) v=1.0; if(v<-1.0) v=-1.0; return v; }

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
            root->priors.assign(M, M>0 ? 1.0 / (double)M : 0.0);
            root->children.assign(M, nullptr);
            root->order_by_prior.resize(M);
            iota(root->order_by_prior.begin(), root->order_by_prior.end(), 0);
            sort(root->order_by_prior.begin(), root->order_by_prior.end(), [&](int a,int b){ return root->priors[a] > root->priors[b]; });
            tt[h] = root;
        }

        const double c_puct = 1.2;
        const int K0 = 6;
        const double K1 = 1.5;

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
                            pn->N += 1;
                            pn->W += v;
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
                    if(node->to_move != root_side) q = -q;
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
                    pn->N += 1;
                    pn->W += v;
                    pn->Q = pn->W / std::max(1, pn->N);
                }
            }
        }

        int M = (int)root->moves.size();
        if(M == 0) return {"",{},{},{},""};
        int best_idx = -1; int bestN = -1; double bestQ = -1e18;
        int cap = root->expanded_count;
        for(int k=0;k<cap;k++){
            int idx = root->order_by_prior[k];
            MCTSNode* ch = root->children[idx];
            if(!ch) continue;
            int n = ch->N; double q = ch->Q;
            if(n > bestN || (n==bestN && q>bestQ)){
                bestN = n; bestQ = q; best_idx = idx;
            }
        }
        if(best_idx < 0) best_idx = 0;
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
