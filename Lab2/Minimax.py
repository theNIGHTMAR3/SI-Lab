import connect4
import sys
import copy

def heurystyka(game):
    score=0
    mid_column=game.center_column()
    amount=mid_column.count('x')
    
    score+=3*amount

    score+=game.check_three_and_two()
    return score


def minimax(game,depth, maximalization,alpha,beta):
    if game.check_game_over():
        if game.wins=='x':
            return 100,None
        elif game.wins=='o':
            return -100,None
        else:
            return 0,None 

    if depth==0:
        #return 0,None
        return heurystyka(game),None

    if maximalization:
        best_score=-sys.maxsize-1
        best_move=None

        for move in game.possible_drops():
            temp_game=copy.deepcopy(game)
            temp_game.drop_token(move)
            score,_=minimax(temp_game,depth-1,False,alpha,beta)
            if score>best_score:
                best_score=score
                best_move=move

            alpha=max(alpha,score)
            if beta<=alpha:
                break
        return best_score,best_move
    else:
        min_score=sys.maxsize

        for move in game.possible_drops():
            temp_game=copy.deepcopy(game)
            temp_game.drop_token(move)
            score,_=minimax(temp_game,depth-1,True,alpha,beta)
            if score<min_score:
                min_score=score
                best_move=move

            beta=min(beta,score)
            if beta<=alpha:
                break
        return min_score,best_move
        

