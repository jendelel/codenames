
from codenames.codenames.codenames_logic import CodenamesBoard
from codenames.codenames.codenames_game import CodenamesGame
import unittest
import numpy as np


class Codenames(unittest.TestCase):
    
    def setUp(self):
        
        num_blue = 2
        num_red = 2

        self.clues_vocab = {'ant', 'deer', 'elephant', 'dog', 'cat'}
        blue_words = {'puppy', 'kitten'}
        red_words = {'giraffe', 'bear'}

        self.vocab = self.clues_vocab.union(blue_words, red_words)

        self.board = CodenamesBoard(num_blue=num_blue, num_red=num_red, vocab=self.vocab)
        self.board.reset(blue=blue_words, red=red_words, assassin=set(), neutral=set())

        self.game = CodenamesGame(num_blue=num_blue, num_red=num_red)
        self.game.clue_vocab_size = len(self.clues_vocab)
        self.game.powerset_indices = self.board.powerset_indices

    def test_action_size(self):

        legal_blue_powersets = 3
        legal_red_powersets = 3

        possible_actions = len(self.clues_vocab) * (legal_blue_powersets + legal_red_powersets)
        computed_actions = self.game.get_action_size()

        np.testing.assert_equal(possible_actions, computed_actions)
    
    def test_allowed_clues_match(self):
        
        assert self.clues_vocab == self.board.allowed_clue_words

    def test_valid_moves(self):

        expected = [1, 1, 1, 0, 0, 0] * len(self.clues_vocab)
        computed = self.game.get_valid_moves(board=self.board, team=1)

        np.testing.assert_equal(computed, expected)

    def test_guess(self):

        next_team = 1
        guesses = ['puppy', 'giraffe']

        # First guess
        guess = guesses[0]
        pieces, next_team = self.game.get_next_state(board=self.board, team=next_team, action=guess,
                                                     is_last_guess=False)

        assert set(list(pieces)) == {'kitten', 'bear', 'giraffe'}
        assert next_team == 1

        # Next guess
        guess = guesses[1]
        pieces, next_team = self.game.get_next_state(board=self.board, team=next_team, action=guess,
                                                     is_last_guess=True)

        assert set(list(pieces)) == {'kitten', 'bear'}
        assert next_team == -1

    def test_game_ended(self):

        game_ended = self.game.get_game_ended(board=self.board, team=1)

        assert game_ended == 0


if __name__ == '__main__':
    unittest.main()
