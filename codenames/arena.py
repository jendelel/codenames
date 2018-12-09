from resources.pytorch_classification.utils import Bar, AverageMeter
import time


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in codenames/CodenamesGame). Is necessary for verbose
                     mode.

        see codenames/CodenamesPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def play_game(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        current_player = 1
        board = self.game.get_init_board()
        it = 0
        while self.game.get_game_ended(board, current_player) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(current_player))
                self.display(board)
            action = players[current_player + 1](self.game.get_canonical_form(board, current_player))

            valids = self.game.get_valid_moves(self.game.get_canonical_form(board, current_player), 1)

            if valids[action] == 0:
                print(action)
                assert valids[action] > 0
            board, current_player = self.game.get_next_state(board, current_player, action)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.get_game_ended(board, 1)))
            self.display(board)
        return self.game.get_game_ended(board, 1)

    def play_games(self, num, verbose=False):

        # TODO: play_game says executes one episode of game; Change name to play_one_episode and then this function
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.play_games', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num / 2)
        one_won = 0
        two_won = 0
        draws = 0
        for _ in range(num):
            game_result = self.play_game(verbose=verbose)
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                eps=eps + 1, maxeps=maxeps, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        self.player1, self.player2 = self.player2, self.player1

        for _ in range(num):
            game_result = self.play_game(verbose=verbose)
            if game_result == -1:
                one_won += 1
            elif game_result == 1:
                two_won += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                eps=eps + 1, maxeps=num, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        bar.finish()

        return one_won, two_won, draws
