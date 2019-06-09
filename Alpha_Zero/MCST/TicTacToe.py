
class OXOState:
    def __init__(self):
        """
        Start the TicTacToe board.
        Player1 starts his move on the 3x3 board.
        Those are the indices of the TicTacToe board:

        0|1|2
        -----
        3|4|5
        -----
        6|7|8

        """
        self.playerJustMoved = 2    # Assuming P2 just made a move,
                                    # Player1 now starts the game.
        self.board = [int(0)] * 9
        self.winning_combinations = [(0, 1, 2),
                                     (3, 4, 5),
                                     (6, 7, 8),

                                     (0, 3, 6),
                                     (1, 4, 7),
                                     (2, 5, 8),

                                     (0, 4, 8),
                                     (2, 4, 6)]

    def Clone(self):
        """
        Clone state of the game.
        This include two pieces of information:
        self.playerJustMoved - which player has just made its move
        self.board - current state of the board
        :return:
        """
        cloned_state = OXOState()
        cloned_state.playerJustMoved = self.playerJustMoved
        cloned_state.board = self.board

    def DoMove(self, move):
        """
        Do a legal move.
        """

        assert 0 <= move < 9
        assert move == int(move)
        assert self.board[move] == 0

        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[move] = self.playerJustMoved

    def GetMoves(self):
        """
        Obtain a list of all legal moves
        """
        return [i for i in range(9) if self.board[i] == 0]

    def GetResult(self, player):
        """
        Get result of the game from point of view of the specified player
        """
        for (x, y, z) in self.winning_combinations:
            if self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == player:
                    return 1.0
                else:
                    return 0.0
        if self.GetMoves() == []:
            return 0.5

            # assert function returns something

    def __repr__(self):
        s = ""
        for i in range(9):
            s += ".XO"[self.board[i]]
            if i % 3 == 2: s += "\n"

        return s





