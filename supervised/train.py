from __future__ import unicode_literals, print_function, division
import random

import torch
import torch.nn as nn
from torch import optim

import argparse
import time
import network
import data
import util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlayerTrainer:

    def __init__(self, hidden_size, board_vocab_size, clues_vocab_size):
        self.decoder = network.AttnDecoderRNN(hidden_size * 2, board_vocab_size).to(device)
        self.clue_scaler = nn.Linear(clues_vocab_size, hidden_size).to(device)

    def train_step(self, encoder_hidden, encoder_outputs, clue_logits, intended_output, criterion, team_word_mask):
        intended_length = intended_output.size(0)
        loss = 0

        clue_scaled = self.clue_scaler(clue_logits)
        decoder_hidden = torch.cat([encoder_hidden, clue_scaled], 1)
        decoder_input = torch.tensor([[data.SOS_TOKEN]], device=device)

        for di in range(intended_length):
            decoder_output, decoder_hidden, decoder_attention_unused = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, intended_output[di])
            if decoder_input.item() == data.EOS_TOKEN or not team_word_mask[decoder_input.item()]:
                break

        return loss

    def parameters(self):
        return self.decoder.parameters() + self.clue_scaler.parameters()


def train(board_input,
          board_vocab,
          encoder,
          clue_predictor,
          clue_scaler,
          decoder,
          player_trainer,
          encoder_opt,
          decoder_opt,
          player_opt,
          criterion,
          team_word_mask,
          att_max_len=network.ATTN_MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()
    player_opt.zero_grad()

    input_length = board_input.size(0)
    encoder_outputs = torch.zeros(att_max_len, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(board_input[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    clue_logits = clue_predictor(encoder_hidden)

    decoder_input = torch.tensor([[data.SOS_TOKEN]], device=device)

    clue_scaled = clue_scaler(clue_logits)
    decoder_hidden = torch.cat([encoder_hidden, clue_scaled], 1)

    intended_output = []
    for di in range(9):  # At most 9 cards of the same color.
        decoder_output, decoder_hidden, decoder_attention_unused = decoder(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        intended_output.append(decoder_input.item())

        if decoder_input.item() == data.EOS_TOKEN:
            break

    intended_output_idx = torch.tensor([board_vocab.word_to_idx[word] for word in intended_output], dtype=torch.long, device=device).view(
        len(intended_output), 1)
    loss += player_trainer.train_step(encoder_hidden, encoder_outputs, clue_logits, intended_output_idx, criterion, team_word_mask)

    loss.backward()

    encoder_opt.step()
    decoder_opt.step()
    player_opt.step()

    return loss.item() / len(intended_output)


def train_iters(encoder,
                clue_predictor,
                clue_scaler,
                decoder,
                player_trainer,
                board_vocab,
                n_iters,
                print_every=1000,
                plot_every=100,
                learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.adam(decoder.parameters() + clue_predictor.parameters() + clue_scaler.parameters(), lr=learning_rate)
    player_optimizer = optim.adam(player_trainer.parameters(), lr=learning_rate)

    data.generate_board(board_vocab)
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        board = data.generate_board(board_vocab)
        training_input = data.tensors_from_board(board_vocab, board)

        # Choose team
        team = data.TEAM_IDX['BLUE'] if random.random() < 0.5 else data.TEAM_IDX['RED']
        team_word_mask = [team == t for (_, t) in board]

        loss = train(training_input, board_vocab, encoder, clue_predictor, clue_scaler, decoder, player_trainer, encoder_optimizer,
                     decoder_optimizer, player_optimizer, criterion, team_word_mask)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (util.time_since(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    util.show_plot(plot_losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board_words", help="Path to file with board words. Each word on a separate line.")
    parser.add_argument("--clue_words", help="Path to file with clue words. Each word on a separate line.")
    args = parser.parse_args()
    assert args.board_words and args.clue_words

    hidden_size = 256
    board_vocab = data.Vocab.load_from_file(args.board_words)
    clue_vocab = data.Vocab.load_from_file(args.clue_words)

    encoder = network.EncoderRNN(len(board_vocab), hidden_size).to(device)
    clue_scaler = nn.Linear(len(clue_vocab), hidden_size).to(device)
    clue_predictor = network.ClueClassificator(hidden_size, len(clue_vocab)).to(device)
    attn_decoder = network.AttnDecoderRNN(hidden_size * 2, len(board_vocab), dropout_p=0.1).to(device)
    player_trainer = PlayerTrainer(hidden_size, len(board_vocab), len(clue_vocab)).to(device)

    train_iters(encoder, clue_predictor, clue_scaler, attn_decoder, player_trainer, board_vocab, 75000, print_every=5000)
