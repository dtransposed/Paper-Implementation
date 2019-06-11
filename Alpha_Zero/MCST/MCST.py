from math import *
from TicTacToe import OXOState
import random

class Node:
    def __init__(self, move = None, parent = None, state= None):
        self.move = move    # Move that got us to that node
        self.parentNode = parent    # "None" if root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves()    # Future child nodes to be explored
        self.playerJustMoved = state.playerJustMoved

    def UCTSelectChild(self):
        s = sorted(self.childNodes, key=lambda c: c.wins / c.visits + sqrt(2 * log(self.visits) / c.visits))[-1]
        return s

    def AddChild(self, move, state):
        n = Node(move=move, parent=self, state=state)
        self.untriedMoves.remove(move)
        self.childNodes.append(n)
        return n

    def Update(self, result):
        self.visits = self.visits + 1
        self.wins = self.wins + result  # Result from POV of PlayerJustMoved


def UCT(rootstate, itermax, verbose = False):

    rootnode = Node(state=rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # SELECTION
        # Starting at root node, recursively select optimal child nodes until
        # leaf node L is reached

        while node.untriedMoves == [] and node.childNodes != []:
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # EXPANSION
        # If L is not a terminal node, create one or more child nodes and
        # select one C

        if node.untriedMoves != []:
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
            node = node.AddChild(m, state)

        # ROLLOUT
        # Run a simulated playout from C until a result is achieved

        while state.GetMoves() !=[]:
            state.DoMove(random.choice(state.GetMoves()))

        # BACKPROPAGATION
        # Update the current move sequence with the simulation
        # results

        while node != None:
            node.Update(state.GetResult(node.playerJustMoved))
            node = node.parentNode

    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move





def UCTPlayGame():

    state = OXOState()

    while (state.GetMoves() != []):
        print(str(state))
        if state.playerJustMoved == 1:
            m = UCT(rootstate = state, itermax = 1000, verbose = False) # play with values for itermax and verbose = True
        else:
            m = UCT(rootstate = state, itermax = 100, verbose = False)
        print("Best Move: " + str(m) + "\n")
        state.DoMove(m)
    if state.GetResult(state.playerJustMoved) == 1.0:
        print("Player " + str(state.playerJustMoved) + " wins!")
    elif state.GetResult(state.playerJustMoved) == 0.0:
        print("Player " + str(3 - state.playerJustMoved) + " wins!")
    else: print("Nobody wins!")


if __name__ == "__main__":
    UCTPlayGame()

