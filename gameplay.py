

from abc import ABCMeta, abstractmethod
import random
import numpy as np
import operator
import networkx as nx
import copy
import matplotlib.pyplot as plt

from GameState3 import GameState
from mcts_player import MCTSPolicy
from mcts_player import RandomPolicy


ai_win = []
hu_win = []

def checkWinner(gs):

    win = gs.winner()

    if win == "O":
        ai_win.append(1)
        print("AI Win. " )
    if win == "X":
        hu_win.append(1)
        print("Random Win. ")

    if win:
        print(gs)
        return True
    else:
        return False

def gameController(gs,mc,rp):

    while True:

        best_move = mc.uctsearch(gs)
        #print("<game play> best_move", best_move)
        gs.move(best_move)
        #print(gs.board)

        if checkWinner(gs):
            break
        next_move = rp.move(gs)
        #print("<game play> random play", next_move)
        gs.move(next_move)
        #print(gs.board)
        if checkWinner(gs):
            break


def main():


    ai_win_rate = []
    hu_win_rate = []


    for i in range(300):
        gs = GameState()
        mc = MCTSPolicy("O")
        rp = RandomPolicy()

        gameController(gs,mc,rp)

        total_ai = np.sum(ai_win)
        total_hu = np.sum(hu_win)

        aiw_r = total_ai / float(i+1)
        huw_r = total_hu / float(i+1)

        ai_win_rate.append(aiw_r)
        hu_win_rate.append(huw_r)

    plt.plot( range(len(ai_win_rate))[10:],ai_win_rate[10:])
    plt.show()





if __name__ == "__main__":
    main()
