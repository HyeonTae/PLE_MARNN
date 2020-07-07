import os
import sys
import math
import numpy as np

import torch
import torch.nn as nn

from models.baseRNN import BaseRNN

# GPU check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncoderRNN(BaseRNN):

    def __init__(self, vocab_size, max_len, hidden_size,
                 embedding_size, input_dropout_p, dropout_p, position_embedding,
                 pos_embedding, n_layers, bidirectional, rnn_cell, variable_lengths,
                 embedding, update_embedding, get_context_vector, pos_add, use_memory, memory_dim):
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        self.get_context_vector = get_context_vector
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.s_rnn = rnn_cell
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.pos_add = pos_add
        if pos_add == 'cat':
            rnn_input_size = embedding_size*2
        else:
            rnn_input_size = embedding_size
        self.rnn = self.rnn_cell(rnn_input_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        self.position_embedding = position_embedding
        self.pos_embedding = pos_embedding

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.use_memory = use_memory
        if use_memory is not None:
            self.init_memory_augmented(max_len, memory_dim)

    def init_memory_augmented(self, max_len, memory_dim):
        self.memory_size = max_len
        self.memory_dim = memory_dim

        self.W_n = nn.Linear(self.hidden_size, self.memory_dim)
        self.W_a = nn.Linear(self.hidden_size, 3)
        self.W_sh = nn.Linear (self.memory_dim, self.hidden_size)

        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

    def init_memory(self, batch_size):
        return torch.zeros (batch_size, self.memory_size, self.memory_dim).to(device)

    def init_lstm_hidden(self, batch_size):
        return (torch.zeros (self.n_layers, batch_size, self.hidden_size).to(device),
                torch.zeros (self.n_layers, batch_size, self.hidden_size).to(device))

    def init_gru_hidden(self, batch_size):
        return (torch.zeros (self.n_layers, batch_size, self.hidden_size).to(device))

    def sin_encoding(self, batch_size, max_len, input_lengths, d_model):
        pe = np.zeros((batch_size, max_len, d_model))
        for batch in range(batch_size):
            for pos in range(max_len):
                if input_lengths[batch] - pos >0:
                    for i in range(0, d_model, 2):
                        pe[batch, pos, i] = math.sin(
                                (input_lengths[batch]-pos)/(10000**(i/d_model)))
                        if i+1 == d_model:
                            break
                        pe[batch, pos, i+1] = math.cos(
                                (input_lengths[batch]-pos)/(10000**(i/d_model)))
                else:
                    for i in range(0, d_model, 2):
                        pe[batch, pos, i] = 0.0
                        if i+1 == d_model:
                            break
                        pe[batch, pos, i+1] = 0.0
        pos = torch.from_numpy(pe)
        if torch.cuda.is_available():
            pos = pos.type(torch.cuda.FloatTensor)
        return pos

    def length_encoding(self, batch_size, max_len, input_lengths):
        pe = []
        for batch in range(batch_size):
            p = []
            for pos in range(max_len):
                if input_lengths[batch] - pos >0:
                    p.append(input_lengths[batch]-pos)
                else:
                    p.append(0)
            pe.append(p)
        pos = torch.tensor(pe)
        if torch.cuda.is_available():
            pos = pos.cuda()
        posemb = self.pos_embedding(pos)
        return posemb

    def forward(self, input_var, input_lengths=None):
        context = None
        batch_size = input_var.size(0)
        seq_len = input_var.size(1)
        memory = None
        encoder_action = None

        if self.position_embedding == "sin":
            posemb = self.sin_encoding(
                batch_size, seq_len, input_lengths, self.embedding_size)
        if self.position_embedding == "length":
            posemb = self.length_encoding(batch_size, seq_len, input_lengths)

        if self.use_memory is None:
            embedded = self.embedding(input_var)
            if self.position_embedding is not None:
                if self.pos_add == 'cat':
                    embedded = torch.cat((embedded, posemb), dim=2)
                elif self.pos_add == 'add':
                    embedded += posemb
            if self.variable_lengths:
                embedded = nn.utils.rnn.pack_padded_sequence(
                        embedded, input_lengths, batch_first=True, enforce_sorted=False)
            output, hidden = self.rnn(embedded)
            if self.variable_lengths:
                output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            memory = self.init_memory(batch_size)
            inpemb = self.embedding(input_var)
            encoder_action = torch.tensor(()).to(device)
            for j in range(seq_len):
                embedded = inpemb[:, j, :].clone().unsqueeze(1)
                if self.position_embedding is not None:
                    if self.pos_add == 'cat':
                        embedded = torch.cat((embedded, posemb[:, j, :].clone().unsqueeze(1)), dim=2)
                    elif self.pos_add == 'add':
                        embedded += posemb[:, j, :].clone().unsqueeze(1)

                if j == 0:
                    if self.s_rnn == "gru":
                        h0 = self.init_gru_hidden(batch_size)
                        hidden0_bar = self.W_sh(memory[:, 0].clone()).view(1, batch_size, -1) + h0
                        ht, hidden = self.rnn(embedded, hidden0_bar)
                    elif self.s_rnn == "lstm":
                        hidden0 = self.init_lstm_hidden(batch_size)
                        h0, c0 = hidden0
                        hidden0_bar = self.W_sh(memory[:, 0].clone()).view(1, batch_size, -1) + h0
                        ht, hidden = self.rnn(embedded, (hidden0_bar, c0))
                    output = ht
                else:
                    if self.s_rnn == "gru":
                        hidden_bar = self.W_sh(memory[:, 0].clone()).view(1, batch_size, -1) + hidden
                        ht, hidden = self.rnn(embedded, hidden_bar)
                    elif self.s_rnn == "lstm":
                        h, c = hidden
                        hidden_bar = self.W_sh(memory[:, 0].clone()).view(1, batch_size, -1) + h
                        ht, hidden = self.rnn(embedded, (hidden_bar, c))
                    output = torch.cat((output, ht), dim=1)

                self.action_weights = self.softmax(self.W_a(ht)).view(batch_size, -1)
                encoder_action = torch.cat((encoder_action, self.action_weights.unsqueeze(1)), dim=1)
                self.new_elt = self.sigmoid(self.W_n(ht))
                if self.use_memory == "stack":
                    push_side = torch.cat((self.new_elt, memory[:, :-1].clone()), dim=1)
                elif self.use_memory == "queue":
                    for i in range(self.memory_size):
                        if memory[:, i].eq(0).all() == 1:
                            push_side = torch.cat((memory[:, :i].clone(),
                                self.new_elt, memory[:, i+1:].clone()), dim=1)
                            break

                pop_side = torch.cat((memory[:, 1:].clone(), torch.zeros(
                        batch_size, 1, self.memory_dim).to(device)), dim=1)

                memory = (self.action_weights[:, 0].clone().unsqueeze(1).unsqueeze(1) * push_side 
                        + self.action_weights[:, 1].clone().unsqueeze(1).unsqueeze(1) * pop_side
                        + self.action_weights[:, 2].clone().unsqueeze(1).unsqueeze(1) * memory)

        return output, hidden, context, encoder_action, memory
