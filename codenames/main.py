from .coach import Coach
from .codenames.codenames_game import CodenamesGame
from .codenames.nnet import NNetWrapper as nn
from .utils import dotdict

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,
    'checkpoint': './data_out/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

if __name__ == "__main__":
    g = CodenamesGame(25)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
