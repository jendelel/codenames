from .coach import Coach
from .codenames.codenames_game import CodenamesGame
from .codenames.nnet import NNetWrapper as nn
from .utils import dotdict

args = dotdict({
    'num_iters': 1000,
    'num_episodes': 100,
    'temp_threshold': 15,
    'update_threshold': 0.6,
    'queue_max_len': 200000,
    'num_mcts': 25,  # Number of MCTS simulations
    'arena_compare': 40,
    'cpuct': 1,  # ?
    'checkpoint': './data_out/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'num_iters_for_train_examples_history': 20,
})

if __name__ == "__main__":
    g = CodenamesGame(25)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.load_train_examples()
    c.learn()
