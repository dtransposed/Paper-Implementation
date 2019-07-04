from TicTacToe import TicTacToeGame
import numpy as np
import random


class Node:

    def __init__(self, move=None, parent=None, state=None):
        self.move = move
        self.parent_node = parent
        self.child_nodes = []
        self.Q = 0
        self.N = 0
        self.untried_moves = state.get_moves()
        self.player_just_moved = state.player_just_moved

    def select_child(self):
        s = sorted(self.child_nodes, key=lambda c: c.Q / c.N + np.sqrt(2 * np.log(self.N) / c.N))[-1]
        return s

    def add_child(self, move, state):
        new_child_node = Node(move =move, parent=self, state=state)
        self.untried_moves.remove(move)
        self.child_nodes.append(new_child_node)
        return new_child_node

    def update(self, result):
        self.Q = self.Q + result
        self.N = self.N + 1


def uct(root_state, max_iterations):

    root_node = Node(state=root_state)

    for i in range(max_iterations):
        node = root_node
        state = root_state.clone()

        ## SELECTION ##
        while node.untried_moves == [] and node.child_nodes != []:
            node = node.select_child()
            state.do_move(node.move)

        ## EXPANSION ##
        if node.untried_moves != []:
            move = random.choice(node.untried_moves)
            state.do_move(move)
            node = node.add_child(move, state)

        ## ROLLOUT ##
        while state.get_moves() != []:
            state.do_move(random.choice(state.get_moves()))

        ## UPDATE ##
        while node != None:
            node.update(state.get_result(node.player_just_moved))
            node = node.parent_node

    return sorted(root_node.child_nodes, key=lambda c: c.N)[-1].move


def play_UCT:
    state = TicTacToeGame()
    print("Welcome to the game. You will face UCT-trained AI opponent.")
    human_turn = True
    if random.random() < 0.5:
        print('Human starts the game!\n')
    else:
        print('AI starts the game\n')
        human_turn = False

    human_started = human_turn

    while state.get_moves() != []:
        if human_turn:
            print(str(state))
            move = int(input('Please choose the field:\n'))
            state.do_move(move)
            human_turn = False

        else:
            move = uct(root_state=state, max_iterations = 5000)
            state.do_move(move)
            human_turn = True

    if (state.get_result(state.player_just_moved) == 1.0 and human_started) or \
            (state.get_result(state.player_just_moved) == 0.0 and not human_started):
        print("AI wins!".format(state.player_just_moved))
    elif (state.get_result(state.player_just_moved) == 0.0 and human_started) or \
            (state.get_result(state.player_just_moved) == 1.0 and not human_started):
        print("Human wins!".format(3- state.player_just_moved))
    else:
        print("Draw!")










