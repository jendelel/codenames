import Arena
from MCTS import MCTS
from codenames.CodenamesGame import CodenamesGame, display
from codenames.CodenamesPlayers import *
from codenames.NNet import NNetWrapper as NNet

import numpy as np
from utils import *
"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = CodenamesGame(6)

# all players
rp = RandomCodenamesPlayer(g).play
gp = GreedyCodenamesPlayer(g).play
hp = HumanCodenamesPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./pretrained_models/codenames/', '6x100x25_best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

arena = Arena.Arena(n1p, hp, g, display=display)
print(arena.playGames(2, verbose=True))
