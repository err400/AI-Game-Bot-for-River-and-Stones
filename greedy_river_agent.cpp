#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <algorithm>
#include <random>
#include <chrono>

namespace py = pybind11;
using namespace std;

// Aggressive greedy agent that exploits river networks
struct Move {
    string action;
    vector<int> from;
    vector<int> to;
    vector<int> pushed_to;
    string orientation;
};

static int G_ROWS = 13;
static int G_COLS = 12;
static vector<int> G_SCORE_COLS;

static inline int score_row(const string& owner) {
    return (owner == "circle") ? 2 : (G_ROWS - 3);
}

static inline bool is_scoring_col(int c) {
    return find(G_SCORE_COLS.begin(), G_SCORE_COLS.end(), c) != G_SCORE_COLS.end();
}

// BFS to find all reachable cells via river network
static set<pair<int,int>> get_river_reachable(
    const vector<vector<map<string,string>>>& board,
    int sr, int sc,
    const string& owner)
{
    set<pair<int,int>> reachable;
    queue<pair<int,int>> q;
    q.push({sr, sc});
    reachable.insert({sr, sc});

    while (!q.empty()) {
        auto [r, c] = q.front();
        q.pop();

        // Check all 4 directions
        vector<pair<int,int>> dirs = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        for (auto [dr, dc] : dirs) {
            int nr = r + dr, nc = c + dc;
            if (nr < 0 || nr >= (int)board.size() || nc < 0 || nc >= (int)board[0].size())
                continue;
            if (reachable.count({nr, nc}))
                continue;

            const auto& cell = board[nr][nc];

            // Can move through friendly rivers
            if (!cell.empty() && cell.at("owner") == owner && cell.at("side") == "river") {
                string ori = cell.at("orientation");
                // Check if river orientation allows passage from (r,c) to (nr,nc)
                if (ori == "horizontal" && dr == 0) {
                    reachable.insert({nr, nc});
                    q.push({nr, nc});
                } else if (ori == "vertical" && dc == 0) {
                    reachable.insert({nr, nc});
                    q.push({nr, nc});
                }
            }
            // Can move to empty cells
            else if (cell.empty()) {
                reachable.insert({nr, nc});
            }
        }
    }
    return reachable;
}

// Distance to scoring row
static int dist_to_score(int r, const string& owner) {
    int target = score_row(owner);
    return abs(r - target);
}

// Count stones in scoring area
static int count_scored(const vector<vector<map<string,string>>>& board, const string& owner) {
    int row = score_row(owner);
    int count = 0;
    for (int c : G_SCORE_COLS) {
        if (row >= 0 && row < (int)board.size() && c >= 0 && c < (int)board[0].size()) {
            const auto& cell = board[row][c];
            if (!cell.empty() && cell.at("side") == "stone" && cell.at("owner") == owner)
                count++;
        }
    }
    return count;
}

// Generate all legal moves for a piece
static vector<Move> gen_piece_moves(
    const vector<vector<map<string,string>>>& board,
    int r, int c,
    const string& owner)
{
    vector<Move> moves;
    const auto& cell = board[r][c];
    if (cell.empty() || cell.at("owner") != owner)
        return moves;

    string side = cell.at("side");

    // FLIP
    if (side == "stone") {
        moves.push_back({"flip", {r,c}, {r,c}, {}, "horizontal"});
        moves.push_back({"flip", {r,c}, {r,c}, {}, "vertical"});
    } else if (side == "river") {
        moves.push_back({"flip", {r,c}, {r,c}, {}, ""});
    }

    // ROTATE (for rivers)
    if (side == "river") {
        moves.push_back({"rotate", {r,c}, {r,c}, {}, ""});
    }

    // MOVE - use river network
    if (side == "stone") {
        auto reachable = get_river_reachable(board, r, c, owner);
        for (auto [tr, tc] : reachable) {
            if (tr == r && tc == c)
                continue;
            const auto& target = board[tr][tc];
            if (target.empty()) {
                moves.push_back({"move", {r,c}, {tr,tc}, {}, ""});
            }
        }
    }

    // PUSH - only adjacent
    if (side == "stone" || side == "river") {
        vector<pair<int,int>> dirs = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        for (auto [dr, dc] : dirs) {
            int nr = r + dr, nc = c + dc;
            if (nr < 0 || nr >= (int)board.size() || nc < 0 || nc >= (int)board[0].size())
                continue;
            const auto& adj = board[nr][nc];
            if (adj.empty() || adj.at("owner") == owner)
                continue;

            // Check if we can push (destination must be empty)
            int pr = nr + dr, pc = nc + dc;
            if (pr < 0 || pr >= (int)board.size() || pc < 0 || pc >= (int)board[0].size())
                continue;
            if (!board[pr][pc].empty())
                continue;

            moves.push_back({"push", {r,c}, {nr,nc}, {pr,pc}, ""});
        }
    }

    return moves;
}

// Apply move to board
static vector<vector<map<string,string>>> apply_move(
    const vector<vector<map<string,string>>>& board,
    const Move& m)
{
    auto nb = board;
    if (m.from.size() < 2)
        return nb;

    int sr = m.from[0], sc = m.from[1];

    if (m.action == "flip") {
        auto& cell = nb[sr][sc];
        if (cell.at("side") == "stone") {
            cell["side"] = "river";
            cell["orientation"] = m.orientation.empty() ? "horizontal" : m.orientation;
        } else {
            cell["side"] = "stone";
            cell.erase("orientation");
        }
    }
    else if (m.action == "rotate") {
        auto& cell = nb[sr][sc];
        if (cell.at("side") == "river") {
            string cur = cell.at("orientation");
            cell["orientation"] = (cur == "horizontal") ? "vertical" : "horizontal";
        }
    }
    else if (m.action == "move" && m.to.size() >= 2) {
        int tr = m.to[0], tc = m.to[1];
        nb[tr][tc] = nb[sr][sc];
        nb[sr][sc].clear();
    }
    else if (m.action == "push" && m.to.size() >= 2 && m.pushed_to.size() >= 2) {
        int nr = m.to[0], nc = m.to[1];
        int pr = m.pushed_to[0], pc = m.pushed_to[1];

        auto pusher = nb[sr][sc];
        auto pushed = nb[nr][nc];

        nb[pr][pc] = pushed;
        nb[nr][nc] = pusher;
        if (pusher.at("side") == "river") {
            nb[nr][nc]["side"] = "stone";
            nb[nr][nc].erase("orientation");
        }
        nb[sr][sc].clear();
    }

    return nb;
}

// Greedy scoring heuristic
static double score_move(
    const vector<vector<map<string,string>>>& board,
    const Move& m,
    const string& owner)
{
    auto nb = apply_move(board, m);

    double score = 0.0;

    // Priority 1: Immediate scoring
    int before = count_scored(board, owner);
    int after = count_scored(nb, owner);
    if (after > before)
        score += 1000.0 * (after - before);

    // Priority 2: Block opponent scoring
    string opp = (owner == "circle") ? "square" : "circle";
    int opp_before = count_scored(board, opp);
    int opp_after = count_scored(nb, opp);
    if (opp_after < opp_before)
        score += 500.0;

    // Priority 3: River network building (early game, stronger weight when few rivers)
    if (m.action == "flip") {
        auto count_rivers = [&](const vector<vector<map<string,string>>>& B){
            int rc = 0;
            for (int i = 0; i < (int)B.size(); ++i)
                for (int j = 0; j < (int)B[0].size(); ++j)
                    if (!B[i][j].empty() && B[i][j].at("owner") == owner && B[i][j].at("side") == "river")
                        rc++;
            return rc;
        };
        int before_r = count_rivers(board);
        int after_r  = count_rivers(nb);
        int delta_r  = after_r - before_r;
        if (delta_r > 0) {
            double base = (before_r < 4) ? 60.0 : 25.0; // encourage early river building
            score += base * delta_r;
        } else {
            score += 5.0 * after_r; // mild preference to maintain networks
        }
        // Favor orientation aligned toward scoring columns slightly
        if (!m.orientation.empty()) {
            if (m.orientation == "horizontal") score += 3.0; else score += 2.0;
        }
    }

    // Priority 4: Forward progress toward scoring row
    if (m.action == "move" && m.to.size() >= 2) {
        int sr = m.from[0];
        int tr = m.to[0];
        int dist_before = dist_to_score(sr, owner);
        int dist_after = dist_to_score(tr, owner);
        int improvement = dist_before - dist_after;

        score += 50.0 * improvement;

        // Bonus for reaching scoring columns
        if (is_scoring_col(m.to[1]))
            score += 20.0;

        // Huge bonus for river-assisted long moves
        int manhattan = abs(tr - sr) + abs(m.to[1] - m.from[1]);
        if (manhattan > 2)
            score += 30.0 * manhattan;
    }

    // Priority 5: Push opponent pieces away
    if (m.action == "push") {
        score += 15.0;
        // Bonus if pushing away from their scoring row
        if (m.pushed_to.size() >= 2) {
            int opp_row = score_row(opp);
            int dist_after_push = abs(m.pushed_to[0] - opp_row);
            score += 5.0 * dist_after_push;
        }
    }

    return score;
}

class GreedyRiverAgent {
public:
    string side;
    mt19937 rng;

    GreedyRiverAgent(string s) : side(s), rng(chrono::steady_clock::now().time_since_epoch().count()) {}

    map<string, py::object> choose(
        const vector<vector<map<string,string>>>& board,
        int rows, int cols,
        const vector<int>& score_cols,
        float time_left, float opp_time)
    {
        G_ROWS = rows;
        G_COLS = cols;
        G_SCORE_COLS = score_cols;

        int need = (cols <= 12) ? 4 : ((cols <= 14) ? 5 : 6);

        // Check for instant win
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (board[i][j].empty() || board[i][j].at("owner") != side)
                    continue;

                auto moves = gen_piece_moves(board, i, j, side);
                for (const auto& m : moves) {
                    auto nb = apply_move(board, m);
                    if (count_scored(nb, side) >= need) {
                        return convert_move(m);
                    }
                }
            }
        }

        // Greedy: pick best scoring move
        Move best = {"", {}, {}, {}, ""};
        double best_score = -1e9;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (board[i][j].empty() || board[i][j].at("owner") != side)
                    continue;

                auto moves = gen_piece_moves(board, i, j, side);
                for (const auto& m : moves) {
                    double s = score_move(board, m, side);

                    // Add small random tiebreaker
                    s += uniform_real_distribution<double>(0.0, 0.1)(rng);

                    if (s > best_score) {
                        best_score = s;
                        best = m;
                    }
                }
            }
        }

        if (best.action.empty()) {
            // Fallback: first legal move
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (board[i][j].empty() || board[i][j].at("owner") != side)
                        continue;
                    auto moves = gen_piece_moves(board, i, j, side);
                    if (!moves.empty()) {
                        best = moves[0];
                        break;
                    }
                }
                if (!best.action.empty())
                    break;
            }
        }

        return convert_move(best);
    }

private:
    map<string, py::object> convert_move(const Move& m) {
        // Convert internal row,col vectors to external [col,row]
        auto swap_rc = [](const vector<int>& v){
            if(v.size() >= 2) return vector<int>{v[1], v[0]};
            return vector<int>{};
        };

        map<string, py::object> result;
        result["action"] = py::cast(m.action);
        result["from"] = py::cast(swap_rc(m.from));
        result["to"] = py::cast(swap_rc(m.to));
        result["pushed_to"] = py::cast(swap_rc(m.pushed_to));
        result["orientation"] = py::cast(m.orientation);
        return result;
    }
};

PYBIND11_MODULE(greedy_river_module, m) {
    py::class_<GreedyRiverAgent>(m, "GreedyRiverAgent")
        .def(py::init<string>())
        .def("choose", &GreedyRiverAgent::choose);
}
