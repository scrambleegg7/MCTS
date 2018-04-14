#
import numpy as np
import math
import logging
import hashlib
import matplotlib.pyplot as plt

from node2 import Node

from uctSearch import UCTSEARCH
from GameState2 import GameState2
from Board import Board

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('GameState')

NUM_TURNS = 9



def isWin(bo,le):
    # bo - input board style
    # le - letter on board from player's piece
    return ((bo[7] == le and bo[8] == le and bo[9] == le) or # across the top
    (bo[4] == le and bo[5] == le and bo[6] == le) or # across the middle
    (bo[1] == le and bo[2] == le and bo[3] == le) or # across the bottom
    (bo[7] == le and bo[4] == le and bo[1] == le) or # down the left side
    (bo[8] == le and bo[5] == le and bo[2] == le) or # down the middle
    (bo[9] == le and bo[6] == le and bo[3] == le) or # down the right side
    (bo[7] == le and bo[5] == le and bo[3] == le) or # diagonal
    (bo[9] == le and bo[5] == le and bo[1] == le)) # diagonal


class Player(object):

    def __init__(self, first):
        self.first = first

    def isSpaceFree(self, board):
        b = np.array(board)
        space = np.where( b  == " "  )
        return space[0][1:]

    def isAvailable(self, board, move):

        m = int(move)

        space = self.isSpaceFree(board)
        #logging.debug("move:%s, available space:%s",move,space)

        if m in space:
            return True
        else:
            return False

class huPlayer(Player):

    def __init__(self,first=True):
        super(huPlayer, self).__init__(first)
        self.piece = "O"


    def randomPlay(self,board):

        space = self.isSpaceFree(board)
        r = np.random.choice(space)
        return r

    def getPlayerMove(self, board):

        # Let the player type in their move.
        move = ' '
        while move not in '1 2 3 4 5 6 7 8 9'.split() or not self.isAvailable(board, move):
            print('What is your next move? (1-9)')
            move = input()

        return int(move)



class aiPlayer(Player):

    def __init__(self,first=False):
        super(aiPlayer,self).__init__(first)
        self.piece = "X"

    def playerMove(self):
        pass

    def randomPlay(self,board):

        space = self.isSpaceFree(board)
        r = np.random.choice(space)
        logging.info("available space for ai...%s ",space)
        logging.info("ai selected %d..", r)
        return r

    def simulaton(self,board):

        space = self.isSpaceFree(board)
        NUM_TURNS = len(space)

        for l in range(NUM_TURNS):
            pass


class GameController(object):

    def __init__(self, board = np.array([' '] * 10), next_turn=0):

        self.board = board
        self.turn = 0
        self.next_turn = next_turn
        self.num_moves = 10
        self.moves = []



    def getBoard(self):

        return self.board

    def makeMove(self,move,le):

        self.board[move] = le

    def __repr__(self):

        print('   |   |')
        print(' ' + self.board[7] + ' | ' + self.board[8] + ' | ' + self.board[9])
        print('   |   |')
        print('-----------')
        print('   |   |')
        print(' ' + self.board[4] + ' | ' + self.board[5] + ' | ' + self.board[6])
        print('   |   |')
        print('-----------')
        print('   |   |')
        print(' ' + self.board[1] + ' | ' + self.board[2] + ' | ' + self.board[3])
        print('   |   |')

        s="CurrentState ...: turn %d"%(self.turn)
        return s

    def terminal(self):

        if isWin(self.board,self.aiPlayer.piece):
            print("AI WIN....")
            return 1
        elif isWin(self.board,self.huPlayer.piece):
            print("Human WIN....")
            return -1
        elif self.turn == NUM_TURNS:
            return 0
        else:
            #print("sorry no winner from terminal..")
            return 0

    def aiPlayerAction(self):

        board_array = self.board
        myBoard = Board(board=board_array)

        current_node = Node(GameState2(moves=self.moves, board=myBoard,turn=self.turn, next_turn="a"))

        for l in range(9 - self.turn):
            uctClass = UCTSEARCH(500/(l+1),current_node)
            current_node = uctClass.getCurrent_Node()

            #print("Num Children: %d"%len(current_node.children))
            #for i,c in enumerate(current_node.children):
            #	print(i,c)
            #print("Best Child: %s"%current_node.state)
            #print("--------------------------------")

            best_state = current_node.state
            move = best_state.moves[ self.turn ]

        self.turn += 1
        self.moves.append(move)
        piece = self.aiPlayer.piece
        self.makeMove(move,piece)

        # show board....
        #print(self)

        if isWin(self.board,piece):
            print("AI WIN....")
            return 1

        return 0

    def huPlayerAction(self): # from human player with manual entry with number...
        move = self.huPlayer.randomPlay(self.board)
        self.moves.append(move)
        piece = self.huPlayer.piece
        self.makeMove(move,piece)

        self.turn += 1

        # print Board
        #print(self)

        if isWin(self.board,piece):
            print("Human WIN....")
            return -1

        return 0


    def huPlayerActionManual(self): # from human player with manual entry with number...
        move = self.huPlayer.getPlayerMove(self.board)
        self.moves.append(move)
        piece = self.huPlayer.piece

        self.makeMove(move,piece)

        self.turn += 1

        # print board
        #print(self)

        if isWin(self.board,self.huPlayer.piece):
            print("Human WIN....")
            return -1

        return 0

    def next_state_ai_vs_hurandom(self):

        if self.aiPlayer.first:
            if self.aiPlayerAction() != 0:
                return 1

            #if self.huPlayerActionManual() != 0:
            if self.huPlayerAction() != 0:
                return -1

        else:
            #if self.huPlayerActionManual() != 0:
            if self.huPlayerAction() != 0:
                return -1

            if self.aiPlayerAction() != 0:
                return 1

        return 0


    def next_state_manual(self):

        if self.aiPlayer.first:
            self.aiPlayerAction()
            self.turn += 1
            print(self)
            #print(self.board)
            #print(self.moves)
            if self.terminal():
                return True

            self.huPlayerActionManual()
            self.turn += 1
            print(self)
            #print(self.board)
            #print(self.moves)
            if self.terminal():
                return True

        else:
            self.huPlayerActionManual()
            self.turn += 1
            print(self)
            #print(self.board)
            #print(self.moves)
            if self.terminal():
                return True

            self.aiPlayerAction()
            self.turn += 1
            print(self)
            #print(self.board)
            #print(self.moves)
            if self.terminal():
                return True

        return False

        logging.info("currnet board for ai...%s ",self.board)

    def firstTurn(self):
        c = np.random.choice((0,1))
        if c == 0: # aiPlayer is first
            self.aiPlayer = aiPlayer(first=True)
            self.huPlayer = huPlayer(first=False)
            #print("* AI first turn.....")
            self.next_turn = 1
        else:
            self.aiPlayer = aiPlayer(first=False)
            self.huPlayer = huPlayer(first=True)
            #print("** human first turn.....")
            self.next_turn = 0


def main():

    ai_win = []
    hu_win = []

    ai_win_rate = []
    hu_win_rate = []

    for i in range(100):

        if i % 100 == 0:
            print("Game Loop counter..",i)

        gs = GameController(board = np.array([' '] * 10), next_turn=0)
        gs.firstTurn() # decide which turn place piece first

        for l in range(NUM_TURNS-1):
            logging.info("turn : %d", l)
            win_lose = gs.next_state_ai_vs_hurandom()

            if win_lose == 1:
                ai_win.append(win_lose)
                break
            elif win_lose == -1:
                hu_win.append(abs(win_lose))
                break

            elif gs.turn == NUM_TURNS-1:
                break

        total_ai = np.sum(ai_win)
        total_hu = np.sum(hu_win)

        aiw_r = total_ai / float(i+1)
        huw_r = total_hu / float(i+1)

        ai_win_rate.append(aiw_r)
        hu_win_rate.append(huw_r)

    plt.plot( range(len(ai_win_rate)),ai_win_rate)
    plt.show()

if __name__ == "__main__":
    main()
