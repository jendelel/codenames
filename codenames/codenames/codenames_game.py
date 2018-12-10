from codenames.codenames.codenames_logic import CodenamesBoard
from codenames.game import Game
import numpy as np


class CodenamesGame(Game):

    def __init__(self, num_blue, num_red, num_assassin=0, num_neutral=0):

        super(CodenamesGame, self).__init__()
        self.num_blue = num_blue
        self.num_red = num_red
        self.num_assassin = num_assassin
        self.num_neutral = num_neutral
        self.clue_vocab_size = 0

    def get_init_board(self):

        """ Gets the initial game word ie words belonging to different categories"""

        board = CodenamesBoard(num_blue=self.num_blue, num_red=self.num_red,
                               num_assassin=self.num_assassin, num_neutral=self.num_neutral)

        board.reset()
        self.clue_vocab_size = len(board.allowed_clue_words)
        self._powerset(board=board)

        # TODO: board.pieces returns the words. Add an embeddings return?

        return board.pieces

    def _powerset(self, board):

        self.powerset_to_index = board.powerset_indices
        self.index_to_powerset = {i: subset for subset, i in self.powerset_to_index.items()}

    def get_board_size(self):

        """ Returns the number of words """

        return self.num_blue + self.num_red + self.num_assassin + self.num_neutral

    def get_action_size(self):

        """ Returns the possible number of guesses that can be made"""

        # For each clue word, you can associate it to 1 word, 2 words and so on
        # associate each clue word with a possible power set

        blue_actions = self.clue_vocab_size * (2 ** self.num_blue - 1)
        red_actions = self.clue_vocab_size * (2 ** self.num_red - 1)

        # TODO: Do you add for assassins and neutral, since you're not looking to guess them?

        return blue_actions + red_actions

    def get_next_state(self, board, team, action, is_last_guess=False):

        """ Given a board state, team and action, get the next board state and team"""

        # Action is a single guess

        assert(isinstance(board, CodenamesBoard))
        turn_complete = board.guess(team=team, guessed_word=action, is_last_guess=is_last_guess)

        # TODO: pieces gives out words, need to change them to embeddings?

        if turn_complete:
            return board.pieces, -team

        else:
            return board.pieces, team

    def get_valid_moves(self, board, team):

        """ Get valid moves for given board state """

        valids = np.array([0] * self.get_action_size())

        assert isinstance(board, CodenamesBoard)

        # Indices associated with the powerset
        indices = board.get_powerset_indices(team=team)

        for i in range(self.clue_vocab_size):

            # Powerset repeated for each clue word, need to update indices accordingly
            indices_to_add = indices + i * (len(self.powerset_to_indices))
            indices_to_add = np.array(indices_to_add)
            valids[indices_to_add] = 1

        return valids

    def get_game_ended(self, board, team):

        """ Returns 0 if game not ended, 1 if team wins and -1 if -team wins"""

        assert isinstance(board, CodenamesBoard)

        assassin_word = board.assassin

        # Game over if last guess was assassin
        if board.last_guess[team] == assassin_word:
            return -1

        elif board.last_guess[-team] == assassin_word:
            return 1

        # Game over if one team guesses all before others
        elif not board.has_words(team) and board.has_words(-team) > 0:
            return 1

        elif not board.has_words(-team) and board.has_words(team) > 0:
            return -1

        # Guesses left
        else:
            return 0

    @staticmethod
    def get_score(board, team):

        """ Returns the number of words guessed for given team"""

        if team == 1:
            return len(board.blue_guessed)

        elif team == -1:
            return len(board.red_guessed)

    def string_representation(self, board):
        pass

    def get_symmetries(self, board, pi):
        pass

    def get_canonical_form(self, board, team):
        pass

def display(board):

    pass