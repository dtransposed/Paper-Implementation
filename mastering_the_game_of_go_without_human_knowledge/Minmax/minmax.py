from sys import maxsize

class Node:
    def __init__(self, i_depth, i_player_num, i_sticks_remaining, i_value = 0):
        self.i_value = i_value
        self.i_sticks_remaining = i_sticks_remaining
        self.i_player_num = i_player_num
        self.i_depth = i_depth
        self.children = []
        self.create_children()

    def create_children(self):
        if self.i_depth >= 0:
            for i in range(1,3):
                v = self.i_sticks_remaining - i
                self.children.append(Node(self.i_depth - 1,
                                          -self.i_player_num,
                                          v,
                                          self.real_val(v)))
    def real_value(self, value):
        if value == 0:
            return maxsize * self.i_player_num
        elif (value < 0):
            return maxsize * -self.i_player_num
        else:
            return 0

def MinMax(node, i_depth, i_player_num):
    if i_depth == 0 or abs(node.i_value) == maxsize:
        return node.i_value
    i_best_value = maxsize * -i_player_num

    for child in node.children:
        i_val = MinMax(child, i_depth - 1, - i_player_num)
        if abs(maxsize * i_player_num - i_val) < abs(maxsize*i_player_num - i_best_value):
            i_best_value = i_val

    return i_best_value

def check_if_won(i_sticks, i_player_num):
    human_victory_message = "The human player has won!"
    human_defeat_message = "The human player has lost..."
    if i_sticks <= 0:
        if i_player_num > 0:
            if i_sticks == 0:
                print( human_victory_message )
            else:
                print(human_defeat_message)

        else:
            if i_sticks == 0:
                print(human_defeat_message)
            else:
                print( human_victory_message)
        return False
    return True

def play_game():
    i_stick_total = 11
    i_depth = 4
    i_current_player = 1
    while i_stick_total > 0:
        print('{} sticks remain. How many sticks would you like to pick up?')
        i_choice = int(input('1 or 2:'))
        i_stick_total -= i_choice
        if check_if_won(i_stick_total, i_current_player):
            i_current_player *= -1
            node = Node(i_depth, i_current_player, i_stick_total)
            best_choice = -100
            i_best_value = -i_current_player * maxsize





