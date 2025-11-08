#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include<iostream>
#include <map>
#include <unordered_map>
#include <random>
#include <set>
#include <array>
#include <chrono>
#include <algorithm>
#include <chrono> // For timing iterative deepening
#include<queue>
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

// Helper functions for path blocking
vector<pair<int,int>> trace_winning_path(const vector<vector<map<string, string>>>& board, const Move& win_move, const string& opponent);
bool simulate_move_path(const vector<vector<map<string, string>>>& board, int current_row, int current_col, int target_row, int target_col, vector<pair<int,int>>& path_stack, set<pair<int,int>>& visited, int dx, int dy);
Move find_path_blocking_moves(const vector<vector<map<string, string>>>& board, const vector<pair<int,int>>& path, const string& owner, const map<pair<int,int>,vector<Move>>& moves);

//this function determines teh minimum distnace from the score area
int min_distance_from_scoring_area(const string& owner,int x,int y){
    int owner_x;
    vector<int> v = {4,5,6,7};
    if(owner == "circle"){
        owner_x = 2;
    }
    else{
        owner_x = 10;
    }
    int min_dist = 100;
    for(auto owner_y:v){
        int dist = abs(x - owner_x) + abs(y - owner_y);
        min_dist = min(min_dist, dist);
    }
    return min_dist;
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
    const int circle_target = 2, square_target = 10;

    auto goal_row = [&](const string& who){ return (who=="circle") ? circle_target : square_target;};
    
    //calculating manhattan distance
    auto MD = [&](const string& who,int i,int j){
        int g = goal_row(who),best = 100;
        for(int sc=4; sc<=7; ++sc)best = min(best,abs(i-g) + abs(j-sc));
        return best;
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
    //defines acore area 
    vector<pair<int, int>> circle_score_area = { {2,4}, {2,5}, {2,6}, {2,7} }; 
    vector<pair<int, int>> square_score_area = { {10,4}, {10,5}, {10,6}, {10,7} }; 
   //checking presence in their respective areas
    if(owner=="circle"){
        int count_circle = 0;
        for(const auto& pos : circle_score_area){
            if(!board[pos.first][pos.second].empty() && board[pos.first][pos.second].at("side")=="stone"&& board[pos.first][pos.second].at("owner") == "circle"){
                count_circle++;
            }
        }
        return count_circle;
    }
    else{
        int count_square = 0;
        for(const auto& pos : square_score_area){
            if(!board[pos.first][pos.second].empty() && board[pos.first][pos.second].at("side")=="stone" && board[pos.first][pos.second].at("owner")=="square"){
                count_square++;
            }
        }
        return count_square;
    }
}

//for checking if the stone was going to enter the scorign area of opposition  
bool is_in_score_area(int row,int col,string owner){

    vector<pair<int, int>> circle_score_area = { {2,4}, {2,5}, {2,6}, {2,7} }; 
    vector<pair<int, int>> square_score_area = { {10,4}, {10,5}, {10,6}, {10,7} }; 
   
    if(owner=="circle"){
        for(const auto& pos : square_score_area){
            if(row == pos.first && col == pos.second){
                return true;
            }
        }
    }
    else{
        for(const auto& pos : circle_score_area){
            if(row ==pos.first && col==pos.second){
                return true;
            }
        }
    }
    return false;
}


//Explore river paths for MOVE actions
//here is where using recursive movements we tarverse from river to river and get to all the possible move for that particualr stone or river
void explore_river_move(const vector<vector<map<string, string>>>& board,int x, int y, vector<Move>& moves, int rows, int cols,int from_row, int from_col, set<pair<int,int>>& visited){
    //not visiting agin already visited
    if (visited.count({x, y})) return;
    visited.insert({x, y});

    //river for new direction movement
    if (board[x][y].at("side") != "river") return;
    string orientation = board[x][y].at("orientation");

    vector<pair<int,int>> directions;
    if (orientation == "horizontal") directions = {{0,1},{0,-1}};
    else directions = {{1,0},{-1,0}};

    string owner = board[from_row][from_col].at("owner");

    for(auto [dx, dy] : directions){
        int nx = x + dx, ny = y + dy;
        while (nx >= 0 && nx < rows && ny >= 0 && ny < cols){
            if (nx == from_row && ny == from_col) { nx += dx; ny += dy; continue;}

            if (board[nx][ny].empty()){
                if (!is_in_score_area(nx, ny, owner)) {
                    moves.push_back({"move", {from_row, from_col}, {nx, ny}, {}, ""});
                }
            } 
            else if (board[nx][ny].at("side") == "river"){
                //river found now check for new contiuos movement direction and thus recursive call
                explore_river_move(board, nx, ny, moves, rows, cols, from_row, from_col, visited);
                break;
            } 
            else break;
            nx += dx; ny += dy;
        }
    }
}

//Explore river paths for PUSH actions
//here is where using recursive movements we tarverse from river to river and get to all the possible move for that particualr stone or river
void explore_river_push(const vector<vector<map<string, string>>>& board,int push_row, int push_col, vector<Move>& moves, int rows, int cols,int from_row, int from_col,
                         set<pair<int,int>>& visited,int or_push_row,int or_push_col, int or_from_row, int or_from_col){

    //not visiting agin the visited
    if (visited.count({push_row, push_col})) return;
    visited.insert({push_row, push_col});

    //getting direction
    if (board[from_row][from_col].at("side") != "river") return;
    string orientation = board[from_row][from_col].at("orientation");

    vector<pair<int,int>> directions;
    if (orientation == "horizontal") directions = {{0,1},{0,-1}};
    else directions = {{1,0},{-1,0}};

    
    string owner_to_be_pushed = board[or_push_row][or_push_col].at("owner");

    for(auto [dx, dy] : directions){
        int nx = push_row + dx;
        int ny = push_col + dy;
        while (nx >= 0 && nx < rows && ny >= 0 && ny < cols) {
            if(nx == from_row && ny == from_col){ 
                nx += dx; 
                ny += dy; 
                continue; 
            }

            if(board[nx][ny].empty()){
                if(!is_in_score_area(nx, ny, owner_to_be_pushed)){
                 // empty thus a possible move position
                    moves.push_back({"push",{or_from_row, or_from_col},{or_push_row, or_push_col},{nx, ny},""});
                }
                else{
                    break;
                }
            } 
            else if(board[nx][ny].at("side") == "river"){
                //got river in path now need to tarce path accoirding to that
                explore_river_push(board, nx, ny, moves, rows, cols, nx, ny, visited,or_push_row,or_push_col, or_from_row, or_from_col);
                break;
            }
            //stone found so break stop
            else break;

            nx += dx; 
            ny += dy;
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
        if(board[nx][ny].empty()){
            moves.push_back({"move", {row,col}, {nx,ny}, {}, ""});
        }
        else if(!board[nx][ny].empty()&& board[nx][ny].at("side")== "river"){
            //exploring moves via rivers
           explore_river_move(board, nx, ny, moves, rows, cols, row, col, visited);
        }
    }
    

    // --- PUSH ---
    for(auto [dx,dy] : dirs){
        int nx = row + dx;
        int ny = col + dy;
        int nx2 = row + 2*dx;
        int ny2 = col + 2*dy;
        if(nx < 0 || nx >= rows || ny < 0 || ny >= cols) continue;
        if(is_in_score_area(nx,ny,owner)) continue;
        if(board[nx][ny].empty()) continue;
        if(board[nx][ny].at("side")!="stone") continue;

        if(!board[nx][ny].empty() && board[nx][ny].at("side")=="stone"){  
            string owner_to_be_pushed = board[nx][ny].at("owner");

            // Skip invalid destination
            if(nx2 < 0 || nx2 >= rows || ny2 < 0 || ny2 >= cols)continue;
            if(is_in_score_area(nx2, ny2, owner_to_be_pushed))continue;
            if(!board[nx2][ny2].empty())continue;

            //  stone push
            if(board[row][col].at("side") == "stone"){
                moves.push_back({"push", {row, col}, {nx, ny}, {nx2, ny2}, ""});
            }
            // River pushing a stone
            else if(board[row][col].at("side") == "river"){
                set<pair<int,int>> visited_push;
                explore_river_push(board,nx,ny,moves,rows,cols,row,col,visited_push,nx,ny,row,col);
            }
        }
    }


    // --- FLIP ---
    //stone to river and river to stone
    if(side == "stone"){
        moves.push_back({"flip",{row,col},{row,col},{},"horizontal"});
        moves.push_back({"flip",{row,col},{row,col},{},"vertical"});
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

    return moves;
}


//checking if any one of the palyer has one and thus recognizing the terminal node
bool is_terminal(const vector<vector<map<string, string>>>& board){
   
    vector<pair<int, int>>circle_score_area = {{2,4},{2,5},{2,6},{2,7}};
    vector<pair<int, int>>square_score_area = {{10,4},{10,5},{10,6},{10,7}};

    int circle_stones = 0,square_stones = 0;

    //counting the number of circles stones in their area
    for(const auto& pos : circle_score_area){
        const auto& cell = board[pos.first][pos.second];
        if(!cell.empty() && cell.at("owner") == "circle" && cell.at("side") == "stone"){
                circle_stones++;
        }
    }
    //counting the number of square stones in their area
    for(const auto& pos:square_score_area){
        const auto& cell = board[pos.first][pos.second];
        if(!cell.empty() && cell.at("owner")=="square" && cell.at("side") == "stone"){
                square_stones++;
        }
    }

    // Terminal if either player has all 4 stones in their score area
    if(circle_stones == 4 || square_stones == 4)return true;

    return false;
}

//applying the minimax function
int min_value(const vector<vector<map<string, string>>>& board,string owner,int max_depth,int current_depth,int alpha,int beta){
    //checking if reached teh termibal node or the cutoff depth
    if(is_terminal(board) || current_depth == max_depth){
        return evaluate_function(board, owner);
    }
    
    int num_rows = board.size();
    int num_cols = board[0].size();
    
    //getting all oponents moves
    vector<pair<Move, vector<vector<map<string, string>>>>> all_moves;
    string opponent = (owner == "circle") ? "square" : "circle";
    
    for(int row = 0; row < num_rows; row++){
        for(int col = 0; col < num_cols; col++){
            if(!board[row][col].empty() && board[row][col].at("owner") == opponent){
                vector<Move> current_moves = generate_all_moves(board, row, col);
                if(current_moves.empty()) continue;
                
                for(const auto& move : current_moves){
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
        int eval_score = evaluate_function(next_board, owner); 
        ordered_moves.push_back({eval_score,{move, next_board}});
    }
    sort(ordered_moves.begin(),ordered_moves.end(),[](const auto& a,const auto& b){
        return a.first < b.first;}); 
    
    // Standard alpha-beta search with ordered moves
    int value = 1e9;
    for(const auto& [score, move_and_board] : ordered_moves){
        const auto& move = move_and_board.first;
        const auto& next_board = move_and_board.second;
        
        int new_value = max_value(next_board, owner, max_depth, current_depth + 1, alpha, beta);
        if(new_value < value){
            value = new_value;
        }
        
        beta = min(beta, value);
        if(value <= alpha){
            return value; // Alpha-beta pruning
        }
    }
    
    return value;
}

int max_value(const vector<vector<map<string, string>>>& board,string owner,int max_depth,int current_depth,int alpha,int beta){
    // Terminal test
    if(is_terminal(board) || current_depth == max_depth){
        return evaluate_function(board, owner);
    }
    
    int num_rows = board.size();
    int num_cols = board[0].size();
    
    // Collect all our moves
    vector<pair<Move, vector<vector<map<string, string>>>>>all_moves;
    
    for(int row = 0; row < num_rows; row++) {
        for(int col = 0; col < num_cols; col++) {
            if(!board[row][col].empty() && board[row][col].at("owner") == owner) {
                vector<Move> current_moves = generate_all_moves(board, row, col);
                if(current_moves.empty()) continue;
                
                for(const auto& move : current_moves) {
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
        // Score from our perspective (we want to maximize our score)
        int eval_score = evaluate_function(next_board, owner);
        ordered_moves.push_back({eval_score,{move, next_board}});
    }
    
    // Sort by score (descending - we want to maximize our score)
    sort(ordered_moves.begin(), ordered_moves.end(), 
         [](const auto& a,const auto& b){return a.first > b.first;});
    
    // Standard alpha-beta search with ordered moves
    int value = -1e9;
    for(const auto& [score, move_and_board] : ordered_moves){
        const auto& move = move_and_board.first;
        const auto& next_board = move_and_board.second;
        
        int new_value = min_value(next_board,owner,max_depth,current_depth + 1,alpha,beta);
        if(new_value > value){
            value = new_value;
        }
        
        alpha = max(alpha, value);
        if(value >= beta){
            return value; // Alpha-beta pruning
        }
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

    // Score and order moves once in decreasing order this would help in alpha beta pruning 
    vector<pair<int,int>> scores; scores.reserve(all_moves.size());
    for(int i=0;i<(int)all_moves.size();++i){
        int s = evaluate_function(all_moves[i].second, owner);
        scores.emplace_back(s, i);
    }
    sort(scores.begin(), scores.end(),[](const auto &a,const auto &b){
        return a.first > b.first;});

    for(auto &p : scores){
        const Move &mv = all_moves[p.second].first;
        const auto &next_board = all_moves[p.second].second;

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
                if(count_in_score_area(updated,owner)==4){
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

        //cehck for my instant win and returning that move
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
        
        // Iterative deepening with time control
        auto start_time = chrono::high_resolution_clock::now();
        double time_limit = 0.1; 
        
        // Adjust time limit based on remaining time
        if(current_player_time < 10){
            time_limit = 0.05; 
        }else if(current_player_time < 30){
            time_limit = 0.075;
        }
        
        // Start with minimal depth
        int depth = 1;
        Move best_move = {"move",{0,0},{0,0},{},""};
        
        // Iterative deepening loop
        while(true){
            // Get best move at current depth
            Move current_best = minimax_decision(board,side,depth);
            
            // Update our best move
            if(current_best.action != ""){
                best_move = current_best;
            }
            
            // Check if we have time for another iteration
            auto current_time = chrono::high_resolution_clock::now();
            chrono::duration<double>elapsed = current_time - start_time;
            
            // If we're getting close to our time limit, or reached a reasonable depth, stop
            if(elapsed.count() > time_limit || depth >= 4){
                break;
            }
            
            // Otherwise, try a deeper search
            depth++;
        }
        
        //here is where we check for three cosecutive same moves and moev the first move not equal to it found 
        SimpleMove current_simple = to_simple(best_move);
        if(last_move_simple.valid && same_simple(last_move_simple, current_simple)){
            consecutive_repeat_count++;
        } else {
            consecutive_repeat_count = 0;
        }
        if(consecutive_repeat_count >= 3){
            // picking different moves
            vector<Move> alt_moves;
            for(int i=0;i<row;i++){
                for(int j=0;j<col;j++){
                    if(!board[i][j].empty() && board[i][j].count("owner") && board[i][j].at("owner")==side){
                        auto gm = generate_all_moves(board,i,j);
                        for(auto &mv : gm)alt_moves.push_back(mv);
                    }
                }
            }
            //picking first move that is not simple , can improve here to think and return some valid move 
            for(const auto &mv : alt_moves){
                if(!same_simple(to_simple(mv),current_simple)){ 
                    best_move = mv; 
                    current_simple = to_simple(best_move); 
                    break; 
                }
            }
            consecutive_repeat_count = 0; // reset after attempt
        }
        last_move_simple = current_simple;
        recent_moves.push_back(current_simple);
        if(recent_moves.size()>20) recent_moves.erase(recent_moves.begin());

        return convert_move_format(best_move);
    }

private:
    string side;
    random_device rd;
    mt19937 gen;


    SimpleMove last_move_simple;          // last move we executed
    int consecutive_repeat_count = 0;     // repeats of same move
    vector<SimpleMove>recent_moves;      // history window (optional future use)

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
    int n = board.size();
    int m = board[0].size();
    string opponent = (owner == "circle") ? "square" : "circle";
    int num_circle = 0;
    int num_square = 0;
    std::vector<std::pair<int, int>> circle_score_area = { {2,4}, {2,5}, {2,6}, {2,7} };
    std::vector<std::pair<int, int>> square_score_area = { {10,4}, {10,5}, {10,6}, {10,7} };
    
    for(const auto& pos : circle_score_area){
        const auto& cell = board[pos.first][pos.second];
        if(!cell.empty() && cell.at("owner") == "circle" && cell.at("side") == "stone"){
            num_circle++;
        }
    }
    
    for(const auto& pos : square_score_area){
        const auto& cell = board[pos.first][pos.second];
        if(!cell.empty() && cell.at("owner") == "square" && cell.at("side") == "stone"){
            num_square++;
        }
    }
    
    // Determine own vs opponent score count
    int n_self = (owner == "circle") ? num_circle : num_square;
    int n_opp = (owner == "circle") ? num_square : num_circle;

    return make_pair(n_self, n_opp);
}

//computing m self  where i get the improvemnt after one move 
//that is the number of stones in score area increases
int compute_m_self(const std::vector<std::vector<std::map<std::string, std::string>>>& board,const string& owner,const map<pair<int,int>,vector<Move>>& moves){


    std::vector<std::pair<int, int>>circle_score_area = { {2,4},{2,5}, {2,6}, {2,7}};
    std::vector<std::pair<int, int>>square_score_area = { {10,4},{10,5},{10,6}, {10,7}};
    //getting which score area to chcek 
    auto& score_area = (owner == "circle") ? circle_score_area : square_score_area;
    
    int n = board.size();
    int m = board[0].size();
    
    // getting the stones now in score area
    int current_stones_in_score = 0;
    for(const auto& pos : score_area){
        if(!board[pos.first][pos.second].empty() && board[pos.first][pos.second].at("owner") == owner && board[pos.first][pos.second].at("side") == "stone"){
            current_stones_in_score++;
        }
    }
   
    //adding score bonuses based on waht state we already had 
    int score_bonus = 0;
    if(current_stones_in_score == 1){
        score_bonus = 1000;
    }
    else if(current_stones_in_score == 2){
        score_bonus = 3000;
    }
    else if(current_stones_in_score == 3){
        score_bonus = 8000;
    }
    else if(current_stones_in_score == 4){
        score_bonus = 1000000;
    }

    
    // Count potential moves that would increase stones in score area
    int potential_moves = 0;

    for(int row = 0; row < n; row++){
        for(int col = 0; col < m; col++){
           if(!board[row][col].empty() && board[row][col].at("owner") == owner){
                auto it = moves.find(make_pair(row, col));
                if (it == moves.end()) continue;
                
                const vector<Move>& current_moves = it->second;
                for(const auto& move : current_moves){
                    int change = 0;
                    
                    // checking for aech moev if its reached positon was in the score area it simply means that we have incraesed numbe rof stones ins core area
                    if(board[row][col].at("side") == "stone" && move.action == "move"){
                        pair<int,int> old_pos = make_pair(move.from[0], move.from[1]);
                        pair<int,int> new_pos = make_pair(move.to[0], move.to[1]);
                        
                        // Check if moving from/to score area
                        for(int k = 0;k < 4;k++){
                            if(old_pos == score_area[k]){
                                change--;
                            }
                            if(new_pos == score_area[k]){
                                change++;
                            }
                        }
                    }
                    // River flipping to stone
                    else if(board[row][col].at("side") == "river" && move.action == "flip"){
                        pair<int,int>new_pos = make_pair(move.to[0],move.to[1]);
                        
                        // Check if new stone position is in score area
                        for(int k = 0; k < 4; k++){
                            if(new_pos == score_area[k]){
                                change++;
                            }
                        }
                    }
                    // Push move
                    else if(move.action == "push"){
                        pair<int,int> old_pos = make_pair(move.from[0], move.from[1]);
                        pair<int,int> new_pos = make_pair(move.to[0], move.to[1]);
                        pair<int,int> pushed_pos = make_pair(move.pushed_to[0], move.pushed_to[1]);
                        
                        bool is_pushing_stone = !board[move.to[0]][move.to[1]].empty() &&  board[move.to[0]][move.to[1]].at("side") == "stone";
                                               
                        for(int k = 0;k < 4;k++){
                            // If our stone is moving from score area
                            if(old_pos == score_area[k] && board[move.from[0]][move.from[1]].at("side") == "stone"){
                                change--;
                            }
                            
                            // If our stone is moving to score area
                            if(new_pos == score_area[k] && board[move.from[0]][move.from[1]].at("side") == "stone"){
                                change++;
                            }
                            
                            // If we're pushing our own stone
                            if(is_pushing_stone && 
                               board[move.to[0]][move.to[1]].at("owner") == owner){
                                // If it's moving from score area
                                if(new_pos == score_area[k]){
                                    change--;
                                }
                                // If it's moving to score area
                                if(pushed_pos == score_area[k]){
                                    change++;
                                }
                            }
                        }
                    }

                    // Count moves that increase stones in score area
                    if(change > 0){
                        potential_moves++;
                    }
                }
           }
        }
    }

    // Return score that heavily prioritizes existing stones in scoring area
    // and also rewards potential to add more stones
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
    int weight_river_syn  = 20;     
    int weight_m_score    = 1100;   
    int weight_oppbest    = 150; 

    auto in_own_score = [&](const string& who, int x, int y){
        if (who=="circle") return (x==2  && y>=4 && y<=7);
        else               return (x==10 && y>=4 && y<=7);
    };

    // Optional flip penalty: discourage flipping unless it creates value.
    int friendly_rivers=0, friendly_stones=0;
    for(int i=0;i<n;i++) for(int j=0;j<m;j++) if(!board[i][j].empty() && board[i][j].at("owner")==owner){
        if(board[i][j].at("side")=="river") friendly_rivers++; else friendly_stones++; }
    int excess = max(0, friendly_rivers - friendly_stones); 
    int flip_penalty = excess * 150;

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
                for (int sc=4; sc<=7; ++sc) best = min(best, abs(i-grow) + abs(j-sc));
                return best;
            };
            int goal_row = (opponent=="circle") ? 2 : 10;
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
    
    for(int c=0; c<m; c++){
        auto [blockers, rivers] = col_analysis(c);
        bool is_wing = (c <= 2 || c >= 9);
        
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
                if(j >= 3 && j <= 8) center_pieces++;
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
        for(int c=3; c<=8; c++){
            if(!board[i][c].empty()) center_congestion++;
        }
    }
    
    // Progressive penalty: light for normal use, heavy for overcrowding
    int corridor_penalty = 0;
    if(center_congestion > 8) { // Heavy congestion
        corridor_penalty = (center_congestion - 8) * (center_congestion - 8) * 50;
    }
    
    // Additional penalty for having too many of our own pieces in center without support
    int our_center_overuse = max(0, center_pieces - 4); // Allow up to 4 center pieces
    if(our_center_overuse > 0 && best_center_supported > 2) { // Only if center is poorly supported
        corridor_penalty += our_center_overuse * our_center_overuse * 100;
    }

    int bridge_hits = 0, enemy_bridge_hits = 0, wing_bridge = 0;
    auto in_wing_col = [&](int y){ return (y<=2 || y>=9); };

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

    // ---- BALANCED WEIGHTS ----
    int w_lane_pref_att   = 1;     // Adaptive lane preference
    int w_bridge_att      = 200;   
    int w_enemybridge_att = 100;    
    int w_wingbridge_att  = 180;   
    int w_corridor_pen_att = 1;    // Adaptive corridor penalty

    int w_lane_pref_def   = 1;
    int w_bridge_def      = 150;
    int w_enemybridge_def = 80;
    int w_wingbridge_def  = 150;
    int w_corridor_pen_def = 1;

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
        + weight_oppbest   * opp_best   
        - flip_penalty;

    eval_total += w_lane_pref   * lane_preference_bonus;    // Intelligent path selection
    eval_total += w_bridge      * bridge_hits;
    eval_total += w_enemyb      * enemy_bridge_hits;
    eval_total += w_wingbridge  * wing_bridge;
    eval_total -= w_corridor_pen * corridor_penalty;       // Adaptive crowding penalty

    set<int> occupied_wing_cols, occupied_center_cols;
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            if(!board[i][j].empty() && board[i][j].at("owner")==owner && board[i][j].at("side")=="stone"){
                if(j <= 2 || j >= 9) {
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
                // Manhattan to *your* scoring row (you already have this helper)
                int dist = min_distance_from_scoring_area(owner, r, c);
                if (is_in_score_area(r, c, owner)) ++in_score_self;
                else if (dist == 1) ++near1_self;
                else if (dist <= 2) ++near2_self;
                else if (dist <= 4) ++near4_self;

                // Can enter scoring area in ONE move (plain move or river ride)
                int d1 = find_next_river_distance(board, owner, r, c, all_moves);
                if (d1 == 0) ++can_score_next_self;
            } 
            else { // opponent stone
                int dist = min_distance_from_scoring_area(opp, r, c);
                if (is_in_score_area(r, c, opp)) ++in_score_opp;
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
    auto nearest_scoring_col_dist = [&](const string& pov, int /*r*/, int c){
        int best = 100;
        for(int sc=4; sc<=7; ++sc) best = min(best, abs(sc - c));
        return best; // 0 means already in a scoring column
    };
    auto row_gap_to_goal = [&](const string& pov, int r){
        return abs(r - ((pov=="circle")? 2 : 10));
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
                    if (who==owner && is_in_score_area(mv.to[0], mv.to[1], owner)){
                        ++finish_now_self; break;
                    }
                    if (who!=owner){
                        string oppo = (owner=="circle")? "square":"circle";
                        if (is_in_score_area(mv.to[0], mv.to[1], oppo)){
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