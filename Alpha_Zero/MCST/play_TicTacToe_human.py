from TicTacToe import OXOState
import random
import numpy as np


def play_tic_tac_toe():

    Game = OXOState()

    while True:
        print(Game)
        legal_moves = Game.GetMoves()
        if len(legal_moves) == 0:
            if Game.GetResult(Game.playerJustMoved) == 1:
                print("Player {} wins!".format(Game.playerJustMoved))
            elif Game.GetResult(Game.playerJustMoved) == 0:
                print("Player {} wins!".format(3 - Game.playerJustMoved))
            else:
                print("Draw!")
            break
        print('Now plays Player: {}'.format(3 - Game.playerJustMoved))
        move = int(input('Choose a move:'))
        assert move in legal_moves

        Game.DoMove(move)

if __name__ == "__main__":
    play_tic_tac_toe()