"""
Use this script to play any two agents against each other, or play manually with
any agent.
"""
from .arena import Arena
from .mcts import MCTS
from .utils import dotdict

from .codenames.codenames_game import CodenamesGame, display
from .codenames.codenames_players import GreedyCodenamesPlayer
from .codenames.codenames_players import HumanCodenamesPlayer
from .codenames.codenames_players import RandomCodenamesPlayer
from .codenames.nnet import NNetWrapper as NNet

import numpy as np

g = CodenamesGame(6)

# all players
rp = RandomCodenamesPlayer(g).play
gp = GreedyCodenamesPlayer(g).play
hp = HumanCodenamesPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./pretrained_models/codenames/', '6x100x25_best.pth.tar')
args1 = dotdict({'num_mcts': 50, 'cpuct': 1.0})
mcts1 = MCTS(g, n1, args1)


def n1p(x):
    return np.argmax(mcts1.get_action_prob(x, temp=0))

# TODO: Add complete function calls
arena = Arena(n1p, hp, g, display=display)
print(arena.play_games(2, verbose=True))
