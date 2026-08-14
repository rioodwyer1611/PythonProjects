import sys
import copy
import time
from visualiser import *
from typing import List, Optional

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

class EightPuzzle:
    def __init__(self, squares: str) -> None:
        self.squares = tuple(squares)

        idx = -1
        for i in range(len(self.squares)):
            if self.squares[i] == '_':
                idx = i
        self.idx = idx

    def __eq__(self, obj: object) -> bool:
        if obj is None:
            return False
        return self.squares == obj.squares

    def __hash__(self) -> int:
        return hash(self.squares)

    ### Actions
    # These are all our transitions!! T: (s,a) -> S'
    def move_left(self) -> 'EightPuzzle':
        new_squares = list(self.squares)
        new_squares[self.idx] = self.squares[self.idx - 1]
        new_squares[self.idx - 1] = self.squares[self.idx]
        return EightPuzzle(new_squares)

    def move_right(self) -> 'EightPuzzle':
        new_squares = list(self.squares)
        new_squares[self.idx] = self.squares[self.idx + 1]
        new_squares[self.idx + 1] = self.squares[self.idx]
        return EightPuzzle(new_squares)

    def move_up(self) -> 'EightPuzzle':
        new_squares = list(self.squares)
        new_squares[self.idx] = self.squares[self.idx - 3]
        new_squares[self.idx - 3] = self.squares[self.idx]
        return EightPuzzle(new_squares)

    def move_down(self) -> 'EightPuzzle':
        new_squares = list(self.squares)
        new_squares[self.idx] = self.squares[self.idx + 3]
        new_squares[self.idx + 3] = self.squares[self.idx]
        return EightPuzzle(new_squares)

    ## Of course, to get the successors, we use the transition state, over all possible actions!
    # So naturally, we can see that we've used our T here
    def get_successors(self) -> List[Optional['EightPuzzle']]:
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

    def num_inversions(self) -> int:
        total = 0
        for i in range(len(self.squares)):
            if self.squares[i] == '_':
                continue
            si = int(self.squares[i])
            for j in range(i, len(self.squares)):
                if self.squares[j] == '_':
                    continue
                sj = int(self.squares[j])
                if si > sj:
                    total += 1
        return total

    def get_parity(self) -> int:
        return self.num_inversions() % 2

    def __str__(self) -> str:
        s = ""
        for c in self.squares:
            s += c
        return s

# Node representation
# This is the object that represents a path/node on the frontier container
# It usually holds the state, parent, action, path_cost (optional)
class StateNode:
    """
    This class represents a node in the frontier of our search algorithm.

    For each entry in the frontier it stores
        - The current state of the puzzle
        - The parent node for the current state
        - The action from the parent that derived this node

    The class also contains a method get_successors which returns all children for this node.
    """

    def __init__(self, puzzle: EightPuzzle, parent: Optional['StateNode'], action_from_parent: Optional[int]) -> None:
        self.puzzle: EightPuzzle = puzzle
        self.parent: Optional['StateNode'] = parent
        self.action_from_parent: Optional[int] = action_from_parent
        # Here you can also add the path_cost which is the step_cost for BFS & DFS
        # because we don't consider edge weights or action costs in BFS/DFS

    # We add get_successors to our Node to abstract away the dependence on the environment class (EightPuzzle)
    # and enable getting of a node's successors when running search
    # Here, we only return states that are valid
    def get_successors(self) -> List['StateNode']:
        s = []
        suc = self.puzzle.get_successors()

        if suc[0] is not None:
            s.append(StateNode(suc[0], self, LEFT))
        if suc[1] is not None:
            s.append(StateNode(suc[1], self, RIGHT))
        if suc[2] is not None:
            s.append(StateNode(suc[2], self, UP))
        if suc[3] is not None:
            s.append(StateNode(suc[3], self, DOWN))

        return s

    def __eq__(self, obj: object) -> bool:
        return self.puzzle == obj.puzzle

# Breadth-First-Search
def bfs(initial: EightPuzzle, goal: EightPuzzle) -> Optional[List[int]]:
    frontier = [StateNode(initial, None, None)]
    visited = set([])

    i = 0
    while len(frontier) > 0:
        # expand node
        node = frontier.pop(0)
        if node.puzzle == goal:
            # Get list of actions taken, dw about it it
            actions = []
            while node.action_from_parent is not None:
                actions.append(node.action_from_parent)
                node = node.parent
            return list(reversed(actions))

        # add successors
        suc = node.get_successors()
        for s in suc:
            if s.puzzle not in visited:
                frontier.append(s)
                visited.add(s.puzzle)
        i += 1

    return None

# Depth-First-Search
def dfs(initial: EightPuzzle, goal: EightPuzzle) -> Optional[List[int]]:
    frontier = [StateNode(initial, None, None)]
    visited = set([])

    i = 0
    while len(frontier) > 0:
        # expand node
        node = frontier.pop(-1)
        if node.puzzle == goal:
            actions = []
            while node.action_from_parent is not None:
                actions.append(node.action_from_parent)
                node = node.parent
            return list(reversed(actions))

        # add successors
        suc = node.get_successors()
        for s in suc:
            if s.puzzle not in visited:
                frontier.append(s)
                visited.add(s.puzzle)
        i += 1
        # if i % 10000 == 0:
        #     print(f'frontier size: {len(frontier)}')
        #     print(f'visited size: {len(visited)}')

    return None


def main(arglist: List[str]) -> None:
    VERBOSE = True  # Make True to run visualiser
    #p1 = EightPuzzle("1348627_5")
    p1 = EightPuzzle("281_43765")
    # p1 = EightPuzzle("281463_75")

    p2 = EightPuzzle("1238_4765")

    print(p1)
    print(p2)
    if p1.get_parity() != p2.get_parity():
        print('No Solution')
        return

    t0 = time.time()
    for _ in range(50):
        actions_bfs = bfs(p1, p2)
    t_bfs = (time.time() - t0) / 50
    num_actions_bfs = len(actions_bfs) if actions_bfs else 0

    t0 = time.time()
    for _ in range(1):
        actions_dfs = dfs(p1, p2)
    t_dfs = (time.time() - t0) / 1
    num_actions_dfs = len(actions_dfs) if actions_dfs else 0

    print(f'BFS: time = {t_bfs} seconds, #actions = {num_actions_bfs}, actions = {actions_bfs}\n'
          f'DFS: time = {t_dfs} seconds, #actions = {num_actions_dfs}')

    if VERBOSE:
        solution_steps = [p1]
        current_puzzle = p1
        for action in actions_bfs:  # Change to actions_dfs for dfs
            if action == LEFT:
                current_puzzle = current_puzzle.move_left()
            elif action == RIGHT:
                current_puzzle = current_puzzle.move_right()
            elif action == UP:
                current_puzzle = current_puzzle.move_up()
            elif action == DOWN:
                current_puzzle = current_puzzle.move_down()
            solution_steps.append(current_puzzle)

        animate_solution(p1, solution_steps)


if __name__ == '__main__':
    main(sys.argv[1:])
