import numpy as np
from enum import Enum
from itertools import chain, combinations


class Team(Enum):
    BLUE = 1
    RED = 2

    
class State:
    def __init__(self, blue, red, assasin=set(), neutral=set()):
        self.blue = blue
        self.red = red
        self.assasin = assasin
        self.neutral = neutral
        
    @property
    def word_sets(self):
        yield self.blue
        yield self.red
        yield self.assasin
        yield self.neutral
        
    @property
    def words(self):
        for word_set in self.word_sets:
            for word in word_set:
                yield word
                
                
    @property
    def hidden_str(self):
        return str(list(self.words))
    
    @property
    def truth_str(self):
        return """
        Blue:\t\t{},
        Red:\t\t{},
        Assasin:\t{},
        Neutral:\t{}
        """.format(self.blue, self.red, self.assasin, self.neutral)

class Clue:
    def __init__(self, word, number, words_meant=set()):
        self.word = word
        self.number = number
        self.words_meant = words_meant
        
    def __str__(self):
        return '{} ({}, {})'.format(self.word, self.number, self.words_meant)
    
    def __repr__(self):
        return self.__str__()
    
class DistanceGuesser:
    def __init__(self, word_vectors, vocab, card_words):
        self.word_vectors = word_vectors
        self.vocab = vocab
        self.card_words = card_words
    
    def _distance(self, a, b):
        return self.word_vectors.distance(a, b)
    
    def guess(self, state, clue, iteration, team=Team.BLUE):
        positive_words = state.blue if team == Team.BLUE else state.red
        min_word = None
        min_loss = np.inf
        for guess_word in state.words:
            loss = self._distance(clue.word, guess_word)
            if loss < min_loss:
                print("  Guess attempt:", guess_word, loss)
                min_loss = loss
                min_word = guess_word
        return min_word
    
class DistanceMaster:
    def __init__(self, word_vectors, vocab, card_words):
        self.word_vectors = word_vectors
        self.vocab = vocab
        self.card_words = card_words
        self.mem = {}
    
    @staticmethod
    def _powerset(iterable):
        """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
    def give_clue(self, state, team=Team.BLUE):
        positive_words = state.blue if team == Team.BLUE else state.red
        results = []
        for subset in DistanceMaster._powerset(positive_words):
            if len(subset) < 1:
                continue
            #words_meant = list(positive_words)
            words_meant = list(subset)
            #print("  Words meant:", words_meant)
            neg_words = set(state.words) - set(words_meant)

            clue = Clue(word=self._find_clue_word(state, words_meant, team=team), number=len(words_meant), words_meant=words_meant)
            
            #loss = self._loss(state, clue.word, team=team) # expected loss
            loss = self._loss2(words_meant, neg_words, clue.word)
            results.append((loss, clue))
            #print("  Clue chosen:", clue, loss)
            
        results = sorted(results, key=lambda x: x[0])
        result_clue = results[0][1]
        return result_clue
    
    def _distance(self, a, b):
        if b < a:
            a, b = b, a
        
        if a not in self.mem:
            self.mem[a] = {b: self.word_vectors.distance(a, b)}
        elif b not in self.mem[a]:
            self.mem[a][b] = self.word_vectors.distance(a, b)
        return self.mem[a][b]
    
    def _loss(self, state, clue_word, team=Team.BLUE, c=2):
        # 1. loss is minimized
        # 2. if blue, subtract distance to blue words with weight c, add distance to red words with weight c, add distance to negative words with weight 1, add distance to assasin with weight 10
        # 3. if red, invert loss terms for teams
        blue_loss = 0
        red_loss = 0
        neutral_loss = 0
        assasin_loss = 0
        for word in state.blue:
            blue_loss += self._distance(clue_word, word)
        for word in state.red:
            red_loss += self._distance(clue_word, word)
        for word in state.neutral:
            neutral_loss += self._distance(clue_word, word)
        for word in state.assasin:
            assasin_loss += self._distance(clue_word, word)
        
        blue_loss /= len(state.blue)
        red_loss /= len(state.red)
        neutral_loss /= len(state.neutral)
        assasin_loss /= len(state.assasin)
        
        if team == team.BLUE:
            loss = + blue_loss - red_loss
        elif team == team.RED:
            loss = - blue_loss + red_loss
        
        return loss * c - neutral_loss - assasin_loss * 10
    
    def _loss2(self, positive_words, negative_words, clue_word):
        pos_loss = 0
        neg_loss = 0
        for word in positive_words:
            pos_loss += self._distance(clue_word, word)
        for word in negative_words:
            neg_loss += self._distance(clue_word, word)
        
        pos_loss /= len(positive_words)
        neg_loss /= len(negative_words)
        return (+ pos_loss - neg_loss) * len(positive_words);
        
    def _find_clue_word(self, state, words_meant, team=Team.BLUE):
        def filter_word_by_rules(clue_word):
            for word_set in state.word_sets:
                for word in word_set:
                    if clue_word in word or word in clue_word: # TODO replace this check by a better one
                        return True
            return False
        
        min_loss = np.inf
        min_word = None
        assert team == Team.BLUE
        pos_words = set(words_meant)
        neg_words = set(state.words) - pos_words
        for clue_word in self.vocab:
            if filter_word_by_rules(clue_word):
                continue
            #loss = self._loss(state, clue_word, team=team) # TODO: This should reflect the word subset choice.
            loss = self._loss2(pos_words, neg_words, clue_word)
            if loss < min_loss:
                # print("    Clue attempt:", clue_word, loss)
                min_loss = loss
                min_word = clue_word
        return min_word

    
def reward_function(state, clue, guess, iteration, team=Team.BLUE):    
    if guess in state.assasin: # lose game
        return -100
    elif guess in state.neutral: # lose a turn
        return -1
    elif (guess in state.blue and team == Team.BLUE) or (guess in state.red and team == Team.RED): # get a turn + correct guess
        return + iteration + 1
    else: # incorrect guess and other team gets a point
        return - iteration - 1
    
class Configuration:
    blue = (5, 5)
    red = (5, 5)
    assasin = (1, 1)
    neutral = (5, 5)
    
    @staticmethod
    def _rand(conf_tuple):
        return np.random.randint(conf_tuple[0], conf_tuple[1] + 1)
    
    def instantiate(self):
        c = Configuration()
        c.blue = Configuration._rand(self.blue)
        c.red = Configuration._rand(self.red)
        c.assasin = Configuration._rand(self.assasin)
        c.neutral = Configuration._rand(self.neutral)
        return c
    
class StateGenerator:
    def __init__(self, word_vectors, vocab, card_words):
        self.configuration = Configuration()
        self.word_vectors = word_vectors
        self.vocab = vocab
        self.card_words = list(card_words)
    
    def generate_state(self):
        c = self.configuration.instantiate()
        total = c.blue + c.red + c.assasin + c.neutral
        chosen_idx = np.random.choice(len(self.card_words), size=total, replace=False)
        chosen = [self.card_words[idx] for idx in chosen_idx]
        ts = c.blue + c.red
        ass = ts + c.assasin
        return State(blue=set(chosen[:c.blue]), red=set(chosen[c.blue:c.blue+c.red]), assasin=set(chosen[ts:ass]), neutral=set(chosen[ass:]))
    