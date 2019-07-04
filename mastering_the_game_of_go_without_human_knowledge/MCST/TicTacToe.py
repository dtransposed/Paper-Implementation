import numpy as np


class TicTacToeGame:
    """
    State of our board:

    00|01|02|03
    -----------
    04|05|06|07
    -----------
    08|09|10|11
    -----------
    12|13|14|15

    """

    def __init__(self):
        self.no_fields = 16
        self.player_just_moved = 2
        self.board = [0]*self.no_fields
        self.winning_combinations = [
                                    (0, 1, 2, 3),
                                    (4, 5, 6, 7),
                                    (8, 9, 10, 11),
                                    (12, 13, 14, 15),

                                    (0, 4, 8, 12),
                                    (1, 5, 9, 13),
                                    (2, 6, 10, 14),
                                    (3, 7, 11, 15),

                                    (0, 5, 10, 15),
                                    (3, 6, 9, 12)]

    def clone(self):
        cloned_state = TicTacToeGame()
        cloned_state.player_just_moved = self.player_just_moved
        cloned_state.board = self.board[:]
        return cloned_state

    def do_move(self, move):
        assert move == int(move)
        assert move >= 0
        assert move <= self.no_fields - 1
        assert self.board[move] == 0

        self.player_just_moved = 3 - self.player_just_moved
        self.board[move] = self.player_just_moved

    def get_moves(self):
        return [i for i in range(self.no_fields) if self.board[i] == 0]

    def get_result(self, current_player):
        for (w, x, y, z) in self.winning_combinations:
            if self.board[w] == self.board[x] == self.board[y] == self.board[z]:
                if self.board[w] == current_player:
                    return 1.0
                else:
                    return 0.0

        if self.get_moves() == []: return 0.5
        assert False

    def __repr__(self):
        s = ""
        for i in range(self.no_fields):
            s += ".XO"[self.board[i]]
            if i % int(np.sqrt(self.no_fields)) ==3: s += "\n"
        return s

