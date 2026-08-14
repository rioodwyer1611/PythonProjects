import tkinter as tk
from tkinter import ttk
import time


class PuzzleVisualiser:
    def __init__(self, root, puzzle, solution_steps):
        self.root = root
        self.puzzle = puzzle
        self.solution_steps = solution_steps
        self.current_step = 0
        self.animation_running = False

        self.create_widgets()
        self.update_puzzle(self.puzzle)

    def create_widgets(self):
        self.frame = tk.Frame(self.root)
        self.frame.grid(row=0, column=0, padx=10, pady=10)

        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                self.buttons[i][j] = tk.Label(self.frame, text="", width=6, height=3, relief="ridge", anchor="center",
                                              font=("Arial", 24))
                self.buttons[i][j].grid(row=i, column=j)

        self.play_button = tk.Button(self.root, text="Play/Rerun", command=self.start_animation)
        self.play_button.grid(row=1, column=0, pady=10)

    def update_puzzle(self, puzzle):
        squares = list(puzzle.squares)
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                text = squares[idx]
                self.buttons[i][j].config(text=text if text != '_' else '')

    def start_animation(self):
        if not self.animation_running:
            self.current_step = 0
            self.animation_running = True
            self.animate()

    def animate(self):
        if self.current_step < len(self.solution_steps):
            puzzle_state = self.solution_steps[self.current_step]
            self.update_puzzle(puzzle_state)
            self.current_step += 1
            self.root.after(500, self.animate)
        else:
            self.animation_running = False


def animate_solution(puzzle, solution_steps):
    root = tk.Tk()
    root.title("8-Puzzle Visualiser")
    visualiser = PuzzleVisualiser(root, puzzle, solution_steps)
    root.mainloop()

