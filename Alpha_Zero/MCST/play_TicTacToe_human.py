from TicTacToe import OXOState

def play_tic_tac_toe():

    game = OXOState()

    while True:
        print(game)
        legal_moves = game.GetMoves()
        if len(legal_moves) == 0:
            if game.GetResult(game.playerJustMoved) == 1:
                print("Player {} wins!".format(game.playerJustMoved))
            elif game.GetResult(game.playerJustMoved) == 0:
                print("Player {} wins!".format(3 - game.playerJustMoved))
            else:
                print("Draw!")
            break
        print('Now plays Player: {}'.format(3 - game.playerJustMoved))
        move = int(input('Choose a move:'))
        assert move in legal_moves

        game.DoMove(move)

if __name__ == "__main__":
    play_tic_tac_toe()