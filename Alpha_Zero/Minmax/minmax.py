from os import system
import platform
import random
from math import inf as infinity
HUMAN = +1
COMP = -1

board = [[0, 0, 0,
          0, 0, 0,
          0, 0, 0]]

def wins(state, player):
    '''

    :param state:
    :param player:
    :return:
    '''
    win_states = [
        [state[0][0], state[0][1], state[0][2]],
        [state[1][0], state[1][1], state[1][2]],
        [state[2][0], state[2][1], state[2][2]],
        [state[0][0], state[0][1], state[0][2]],
        [state[1][0], state[1][1], state[1][2]],
        [state[2][0], state[2][1], state[2][2]]]

    if [win, win, win] in win_state:
        return True
    else:
        return False


def evaluate(state):
    '''

    :param state: current state of the board
    :return: +1 if the human wins, -1 if the computer wins, 0 draw
    '''
    if wins(state, COMP):
        score = -1
    elif wins(state, HUMAN):
        score = +1
    else:
        score = 0

    return score

def game_over(state):
    """

    :param state:
    :return:
    """
    return wins(state, HUMAN) or wins(state, COMP)

def empty_cells(state):
    """

    :param state:
    :return:
    """
    cells = []

    for x, row in enumerate(state):
        for y, cell in enumerate(row):
            if cell == 0:
                cells.append([x, y])
    return cells

def valid_move(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    if [x,y] in empty_cells(board)
        return True
    else:
        return False
def clean():
    """
    Cleans the console
    """
    os_name = platform.system().lower()
    if 'windows' in os_name:
        system('cls')
    else:
        system('clear')

def minimax(state, depth, player):
    """

    :param state:
    :param depth:
    :param player:
    :return:
    """

    if player == COMP:
        best = [-1,-1, -infinity] #movex movey score
    else:
        best = [-1, -1, infinity]

    if depth == 0 or game_over(state): # if final leaf reached
        score = evaluate(state)
        return [-1, -1, score]

    for cell in empty_cells(state): # go  through all choices
        x, y = cell[0], cell[1] # take a choice
        state[x][y] = player # assign max or min
        score = minimax(state, depth - 1, -player)
        state[x][y] = 0 #undo move







def main():
    """

    :return:
    """
    clean()
    human_choice = ''
    computer_choice = ''
    human_first = True

    while human_choice != 'O' and human_choice != 'X':
        try:
            print('')
            human_choice = input('Choose X or O\nChosen: ').upper()
        except (EOFError, KeyboardInterrupt):
            print('Bye')
            exit()
        except (KeyError, ValueError):
            print('Bad choice')


    if human_choice == 'X':
        computer_choice == 'O'
    else:
        computer_choice == 'X'

    clean()

    # Randomly deciding who starts first
    if random.random() < 0.5:
        human_first = False

    if human_first:
        print('Human player starts the game!')
    else:
        print('A.I starts the game')

    while len(empty_cells(board)) > 0 and not game_over(board):
        if human_first:
            human_turn(computer_choice, human_choice)




if __name__ == '__main__':
    main()