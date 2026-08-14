import sys
import time


# Define the 4 possible actions in 8-puzzle
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

# Define an EightPuzzle class
class EightPuzzle:

    def __init__(self, squares):
        self.squares = tuple(squares)

        idx = -1
        for i in range(len(self.squares)):
            if self.squares[i] == '_':
                idx = i
        self.idx = idx

    def __eq__(self, obj):
        # required to test if this object is in a collection
        if obj is None:
            return False
        return self.squares == obj.squares

    def __hash__(self):
        # required to allow this object to be placed inside a hashtable (i.e. set/dict)
        return hash(self.squares)

    def move_left(self):
        new_squares = list(self.squares)
        new_squares[self.idx] = self.squares[self.idx-1]
        new_squares[self.idx-1] = self.squares[self.idx]
        return EightPuzzle(new_squares)

    def move_right(self):
        new_squares = list(self.squares)
        new_squares[self.idx] = self.squares[self.idx+1]
        new_squares[self.idx+1] = self.squares[self.idx]
        return EightPuzzle(new_squares)

    def move_up(self):
        new_squares = list(self.squares)
        new_squares[self.idx] = self.squares[self.idx-3]
        new_squares[self.idx-3] = self.squares[self.idx]
        return EightPuzzle(new_squares)

    def move_down(self):
        new_squares = list(self.squares)
        new_squares[self.idx] = self.squares[self.idx+3]
        new_squares[self.idx+3] = self.squares[self.idx]
        return EightPuzzle(new_squares)

    def get_successors(self):
        successors = []

        if self.idx % 3 > 0:
            successors.append(self.move_left())
        else:
            successors.append(None)

        if self.idx % 3 < 2:
            successors.append(self.move_right())
        else:
            successors.append(None)

        if self.idx // 3 > 0:
            successors.append(self.move_up())
        else:
            successors.append(None)

        if self.idx // 3 < 2:
            successors.append(self.move_down())
        else:
            successors.append(None)

        return successors

    def num_inversions(self):
        """ Write code here to calculate the number of inversions of self.squares"""
        n_inversions = 0
        return n_inversions

    def get_parity(self):
        """ Write code here to determine the parity of self.squares using your num_inversions method"""
        return 0

    def __str__(self):
        s = ""
        for c in self.squares:
            s += c
        return s


# Node representation
# This is the object that represents a path/node on the frontier container
# It usually holds the state, parent, action, path_cost (optional)
class StateNode:

    """ Write code here to implement a node representation for entries in the frontier container
        You should have a constructor that sets the state (puzzle), parent, action (and optionally path_cost / num_steps)
    """
    def __init__(self, puzzle):
        #TO DO: add required arguments for the node
        self.puzzle = puzzle

    # We add get_successors to our Node to abstract away the dependence on the environment class (EightPuzzle)
    # and enable getting of a node's successors when running search
    # Here, we should only return states that are valid
    def get_successors(self):
        #TO DO: implement the get_successors function
        s = []
        suc = self.puzzle.get_successors()

        return s

    def __eq__(self, obj):
        return self.puzzle == obj.puzzle


def bfs(initial, goal):
    """ Implement Breadth-First-Search Here"""

    return None


def dfs(initial, goal):
    """ Implement Depth-First-Search Here"""

    return None


def main(arglist):
    p1 = EightPuzzle("1348627_5")
    # p1 = EightPuzzle("281_43765")
    # p1 = EightPuzzle("281463_75")

    p2 = EightPuzzle("1238_4765")

    if p1.get_parity() != p2.get_parity():
        print('No Solution')
        return

    t0 = time.time()
    for _ in range(50):
        actions_bfs = bfs(p1, p2)
    t_bfs = (time.time() - t0) / 50
    num_actions_bfs = len(actions_bfs)

    t0 = time.time()
    for _ in range(1):
        actions_dfs = dfs(p1, p2)
    t_dfs = (time.time() - t0) / 1
    num_actions_dfs = len(actions_dfs)

    print(f'BFS: time = {t_bfs} seconds, #actions = {num_actions_bfs}, actions = {actions_bfs}\n'
          f'DFS: time = {t_dfs} seconds, #actions = {num_actions_dfs}')


if __name__ == '__main__':
    main(sys.argv[1:])

