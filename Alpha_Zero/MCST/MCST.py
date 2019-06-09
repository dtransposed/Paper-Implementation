from math import *
from TicTacToe import OXOState
import random

class Node:
    def __init__(self, move = None, parent = None, state= None):
        self.move = move # move that got us to that node
        self.parentNode = parent # "None" if root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves() # future child nodes

    def UCTSelectChild(self):

        s = sorted(self.childNodes, key = lambda c:
                   c.wins/c.visits + sqrt(2*log(self.visits/c.visits)))[-1]
        return s

    def AddChild(self, move, state):
        n = Node(move = move, parent = self, state = state)
        self.untriedMoves.remove(move)
        self.childNodes.append(n)

    def Update(self, result):
        self.visits = self.visits + 1
        self.wins = self.wins + result # result from POV of PlayerJustMoved

def UCT(rootstate, itermax, verbose = False):

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # SELECT
        while node.untriedMoves == [] and node.childNodes != []: #until node is fully expanded and not terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # EXPAND
        if node.untriedMoves != []: # if node non-terminal, otherwise continue
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
            node = node.AddChild(m, state)

        #ROLLOUT
        while state.GetMoves() !=[]: #when state is non-terminal
            state.DoMove(random.choice(state.GetMoves()))
        #BACKPROPAGATE
        while node != None:
            node.Update(state.GetResult(node.playerJustMoved))
            node = node.parentNode
    if verbose:
        print(rootnode.TreeToString(0))
    else:
        print(rootnode.ChildrenToString())

    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move





def UCTPlayGame():

    state = OXOState()
    while state.GetMoves != []: # until board is full
        print(str(state))
        if state.playerJustMoved == 1:
            m = UCT(rootstate= state, itermax = 1000, verbose= False)
        else:
            m = UCT(rootstate=state, itermax = 100, verbose = False)
        print('Best Move: {}'.format(m))
        state.DoMove(m)
    if state.GetResult(state.playerJustMoved) == 1:
        print("Player {} wins!".format(state.playJustMoved))
    elif state.GetResult(state.playerJustMoved) == 0:
        print("Player {} wins!".format(3 - state.playerJustMoved))
    else:
        print("Draw!")







if __name__ == "__main__":
    UCTPlayGame()

