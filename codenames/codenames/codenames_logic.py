'''
Author: Eric P. Nichols
Date: Feb 8, 2008.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''

from itertools import chain, combinations
import numpy as np


class CodenamesBoard:

    def __init__(self, num_blue, num_red, num_assassin=0, num_neutral=0):

        self.num_blue = num_blue
        self.num_red = num_red
        self.num_assassin = num_assassin
        self.num_neutral = num_neutral

        self._generate_dicts()

    def _generate_dicts(self):

        """ Generate the vocab to index and index to vocab dict"""

        self.vocab = set()

        pass

    @staticmethod
    def powerset(words):
        s = list(words)
        return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

    def _prepare_wordsets(self):

        """ Samples words from the vocabulary"""

        # TODO: Add sampling from vocabulary file

        self.blue = set()
        self.red = set()
        self.assassin = set()
        self.neutral = set()

        # Get the initial powerset indices
        powset_blue = self.powerset(self.blue)
        powset_blue.pop(0)  # Remove the empty set
        powset_red = self.powerset(self.red)
        powset_red.pop(0)   # Remove the empty set

        assert len(powset_blue) == 2 ** self.num_blue - 1
        assert len(powset_red) == 2 ** self.num_red - 1

        combined_powset = powset_blue + powset_red
        self.powerset_indices = {i: subset for i, subset in enumerate(combined_powset)}

        self.allowed_clue_words = self.vocab - (self.blue.union(self.red, self.assassin, self.neutral))

    def _init_guessed_words(self):

        """ Initializes guessed_XXX (XXX = {blue, red, assassin, neutral}) to empty"""

        self.blue_guessed = set()
        self.red_guessed = set()
        self.neutral_guessed = set()

    def _init_last_guess(self):

        """ Initialize last_guess """

        # Last_guess keeps track of most recent guesses by both teams
        # Used for checking game ended condition in case an assassin was guessed

        self.last_guess = {1: '', -1: ''}

    def reset(self):

        """ Generates the starting board """

        self._prepare_wordsets()
        self._init_guessed_words()
        self._init_last_guess()

    @property
    def word_sets(self):
        yield self.blue
        yield self.red
        yield self.assassin
        yield self.neutral

    # TODO: Property pieces: returns the words; return embeddings?

    @property
    def pieces(self):
        for word_set in self.word_sets:
            for word in word_set:
                yield word

    @pieces.setter
    def pieces(self, board):

        self.blue = board.blue
        self.red = board.red
        self.assassin = board.assassin
        self.neutral = board.neutral

    def get_powerset_indices(self, team):

        """ Returns the indices of the possible subsets of remaining words"""

        def get_indices(x):
            return self.powerset_indices[x]

        if team == 1:
            powset = self.powerset(self.blue)
        else:
            powset = self.powerset(self.red)

        powset.pop(0)
        return np.array(list(map(get_indices, powset)))

    def guess(self, team, guessed_word, is_last_guess=False):

        """ Changes the board state after a word was guessed"""

        self.last_guess[team] = guessed_word
        turn_complete = False

        # Word guessed correctly, update the board state
        # check if this is the last guess
        if team == 1 and guessed_word in self.blue:

            self.blue.remove(guessed_word)
            self.blue_guessed.update(guessed_word)

            if is_last_guess:
                turn_complete = True

            return turn_complete

        if team == -1 and guessed_word in self.red:

            self.red.remove(guessed_word)
            self.red_guessed.update(guessed_word)

            if is_last_guess:
                turn_complete = True

            return turn_complete

        # Word guessed incorrectly, update state and change turn
        if team == 1 and guessed_word not in self.blue:
            turn_complete = True

            if guessed_word in self.red:
                self.red.remove(guessed_word)
                self.red_guessed.update(guessed_word)

            elif guessed_word in self.neutral:
                self.neutral.remove(guessed_word)
                self.neutral_guessed.update(guessed_word)

            # TODO: Add an extra for assassin

            return turn_complete

        if team == -1 and guessed_word not in self.red:
            turn_complete = True

            if guessed_word in self.red:
                self.blue.remove(guessed_word)
                self.blue_guessed.update(guessed_word)

            elif guessed_word in self.neutral:
                self.neutral.remove(guessed_word)
                self.neutral_guessed.update(guessed_word)

            return turn_complete

    def has_words(self, team):

        """ Keeps count of how many words are remaining for each team"""

        if team == 1:
            return len(self.blue)

        elif team == -1:
            return len(self.red)


class Board:

    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]

    def __init__(self, n):
        "Set up initial board configuration."

        self.n = n
        # Create the empty board array.
        self.pieces = [None] * self.n
        for i in range(self.n):
            self.pieces[i] = [0] * self.n

        # Set up the initial 4 pieces.
        self.pieces[int(self.n / 2) - 1][int(self.n / 2)] = 1
        self.pieces[int(self.n / 2)][int(self.n / 2) - 1] = 1
        self.pieces[int(self.n / 2) - 1][int(self.n / 2) - 1] = -1
        self.pieces[int(self.n / 2)][int(self.n / 2)] = -1

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]

    def count_diff(self, color):
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == color:
                    count += 1
                if self[x][y] == -color:
                    count -= 1
        return count

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == color:
                    new_moves = self.get_moves_for_square((x, y))
                    moves.update(new_moves)
        return list(moves)

    def has_legal_moves(self, color):
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == color:
                    new_moves = self.get_moves_for_square((x, y))
                    if len(new_moves) > 0:
                        return True
        return False

    def get_moves_for_square(self, square):
        """Returns all the legal moves that use the given square as a base.
        That is, if the given square is (3,4) and it contains a black piece,
        and (3,5) and (3,6) contain white pieces, and (3,7) is empty, one
        of the returned moves is (3,7) because everything from there to (3,4)
        is flipped.
        """
        (x, y) = square

        # determine the color of the piece.
        color = self[x][y]

        # skip empty source squares.
        if color == 0:
            return None

        # search all possible directions.
        moves = []
        for direction in self.__directions:
            move = self._discover_move(square, direction)
            if move:
                # print(square,move,direction)
                moves.append(move)

        # return the generated move list
        return moves

    def execute_move(self, move, color):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """

        # Much like move generation, start at the new piece's square and
        # follow it on all 8 directions to look for a piece allowing flipping.

        # Add the piece to the empty square.
        # print(move)
        flips = [flip for direction in self.__directions for flip in self._get_flips(move, direction, color)]
        assert len(list(flips)) > 0
        for x, y in flips:
            # print(self[x][y],color)
            self[x][y] = color

    def _discover_move(self, origin, direction):
        """ Returns the endpoint for a legal move, starting at the given origin,
        moving by the given increment."""
        x, y = origin
        color = self[x][y]
        flips = []

        for x, y in Board._increment_move(origin, direction, self.n):
            if self[x][y] == 0:
                if flips:
                    # print("Found", x,y)
                    return x, y
                else:
                    return None
            elif self[x][y] == color:
                return None
            elif self[x][y] == -color:
                # print("Flip",x,y)
                flips.append((x, y))

    def _get_flips(self, origin, direction, color):
        """ Gets the list of flips for a vertex and direction to use with the
        execute_move function """
        # initialize variables
        flips = [origin]

        for x, y in Board._increment_move(origin, direction, self.n):
            # print(x,y)
            if self[x][y] == 0:
                return []
            if self[x][y] == -color:
                flips.append((x, y))
            elif self[x][y] == color and len(flips) > 0:
                # print(flips)
                return flips

        return []

    @staticmethod
    def _increment_move(move, direction, n):
        # print(move)
        """ Generator expression for incrementing moves """
        move = list(map(sum, zip(move, direction)))
        # move = (move[0]+direction[0], move[1]+direction[1])
        while all(map(lambda x: 0 <= x < n, move)):
            # while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
            yield move
            move = list(map(sum, zip(move, direction)))
            # move = (move[0]+direction[0],move[1]+direction[1])
