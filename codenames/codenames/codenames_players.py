import numpy as np

# TODO: Add classes for spymaster and guessers


class RandomCodenamesPlayer:

    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.get_action_size())
        valids = self.game.get_valid_moves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.get_action_size())
        return a


class HumanCodenamesPlayer:

    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.get_valid_moves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i / self.game.n), int(i % self.game.n))
        while True:
            a = input()

            x, y = [int(x) for x in a.split(' ')]
            a = self.game.n * x + y if x != -1 else self.game.n**2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a


class GreedyCodenamesPlayer:

    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.get_valid_moves(board, 1)
        candidates = []
        for a in range(self.game.get_action_size()):
            if valids[a] == 0:
                continue
            next_board, _ = self.game.get_next_state(board, 1, a)
            score = self.game.get_score(next_board, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]
