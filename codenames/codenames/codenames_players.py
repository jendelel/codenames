import numpy as np
from codenames.codenames.codenames_game import CodenamesGame

# TODO: Add classes for spymaster and guessers


class Clue:

    def __init__(self, word, number, words_meant=set()):
        self.word = word
        self.number = number
        self.words_meant = words_meant

    def __str__(self):
        return '{} ({}, {})'.format(self.word, self.number, self.words_meant)

    def __repr__(self):
        return self.__str__()


class RandomCodenamesTeam:

    # Assume team 1 by default

    def __init__(self, game):

        assert isinstance(game, CodenamesGame)
        self.game = game

    def play(self, board):

        valid_moves = self.game.get_valid_moves(board=board, team=1)
        clue = self.generate_clue(moves=valid_moves)

        # Can change it to generate a single guess
        actions = self.generate_guesses(board=board, clue=clue)

        return actions

    def generate_clue(self, moves):

        a = np.random.randint(self.game.get_action_size())

        while moves[a] != 1:
            a = np.random.randint(self.game.get_action_size())
            loc = a % len(self.game.clue_vocab_size)
            clue_size = len(self.game.index_to_powerset[loc])
            words_meant = set(list(self.game.index_to_powerset[loc]))

        clue = Clue(word=a, number=clue_size, words_meant=words_meant)

        return clue

    @staticmethod
    def generate_guesses(board, clue):

        guesses = set()
        pieces = list(board.pieces)

        while len(guesses) <= clue.number + 1:
            guesses.update(np.random.choice(pieces))

        return guesses


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
