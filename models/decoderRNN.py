import os
import sys
import random
import math
import copy

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.attention import Attention
from models.hard_attention import HardAttention
from models.baseRNN import BaseRNN

# GPU check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DecoderRNN(BaseRNN):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'
    KEY_ENCODER_OUTPUTS = 'encoder_outputs'
    KEY_ENCODER_CONTEXT = 'encoder_context'
    KEY_ENCODER_ACTION = 'encoder_action'
    KEY_DECODER_ACTION = 'decoder_action'

    def __init__(self, vocab_size, max_len, hidden_size, embedding_size,
            sos_id, eos_id, input_dropout_p, dropout_p, position_embedding,
            pos_embedding, n_layers, bidirectional, rnn_cell, use_attention,
            attn_layers, hard_attn, pos_add, use_memory, memory_dim):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p,
                n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.output_size = vocab_size
        self.attn_layers = attn_layers
        self.max_length = max_len
        self.use_attention = use_attention
        self.hard_attn = hard_attn
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.s_rnn = rnn_cell
        self.init_input = None
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.output_size, embedding_size)
        self.pos_embedding = pos_embedding
        self.position_embedding = position_embedding
        self.pos_add = pos_add
        if pos_add == 'cat':
            rnn_input_size = embedding_size*2
        else:
            rnn_input_size = embedding_size
        self.rnn = self.rnn_cell(rnn_input_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        if use_attention:
            if hard_attn:
                self.attention = Attention(self.hidden_size)
                self.hard_attention = HardAttention(self.hidden_size)
                self.out = nn.Linear(self.hidden_size*2, self.output_size)
            else:
                self.attention1 = Attention(int(self.hidden_size/attn_layers))
                self.out = nn.Linear(self.hidden_size, self.output_size)
        else:
            self.out = nn.Linear(self.hidden_size, self.output_size)
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

    def forward_step(self, input_var, posemb, hidden, encoder_outputs, di, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        memory = None
        decoder_action = None

        if self.use_memory is None:
            embedded = self.embedding(input_var)
            if self.position_embedding is not None:
                if self.pos_add == 'cat':
                    embedded = torch.cat((embedded, posemb), dim=2)
                else:
                    embedded += posemb
            output, hidden = self.rnn(embedded, hidden)
        else:
            memory = self.init_memory(batch_size)
            inpemb = self.embedding(input_var)
            decoder_action = torch.tensor(()).to(device)
            for j in range(output_size):
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
                decoder_action = torch.cat((decoder_action, self.action_weights.unsqueeze(1)), dim=1)
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

        attn = None
        if self.use_attention:
            if self.hard_attn:
                s_output, s_attn = self.attention(output, encoder_outputs)
                h_output, h_attn = self.hard_attention(output, encoder_outputs, di)
                output = torch.cat((s_output, h_output), dim=2)
                attn = torch.cat((s_attn, h_attn), dim=1)
                hidden_sizes = self.hidden_size*2
            else:
                hidden_sizes = self.hidden_size
                if self.attn_layers is 1:
                    output, attn = self.attention1(output, encoder_outputs)
                if self.attn_layers is 2:
                    o1, o2 = torch.chunk(output, 2, dim=2)
                    e1, e2 = torch.chunk(encoder_outputs, 2, dim=2)
                    output, attn = self.attention1(o1, e1)
                    output2, attn2 = self.attention1(o2, e2)
                    output = torch.cat((output1, output2), dim=2)
                    attn = torch.cat((attn1, attn2), dim=1)
                elif self.attn_layers is 4:
                    o1, o2, o3, o4 = torch.chunk(output, 4, dim=2)
                    e1, e2, e3, e4 = torch.chunk(encoder_outputs, 4, dim=2)
                    output1, attn1 = self.attention1(o1, e1)
                    output2, attn2 = self.attention1(o2, e2)
                    output3, attn3 = self.attention1(o3, e3)
                    output4, attn4 = self.attention1(o4, e4)
                    output = torch.cat((output1, output2, output3, output4), dim=2)
                    attn = torch.cat((attn1, attn2, attn3, attn4), dim=1)
        else:
            hidden_sizes = self.hidden_size

        predicted_softmax = function(
                self.out(output.contiguous().view(-1, hidden_sizes)), dim=1).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn, decoder_action, memory

    def sin_encoding(self, tgt_vocab, inputs,
            batch_size, max_len, inputs_lengths, d_model):
        zero_tok = tgt_vocab.stoi['0']
        pe = np.zeros((batch_size, max_len, d_model))
        for batch in range(batch_size):
            for m in range(max_len):
                if inputs_lengths[batch] <= 0:
                    for i in range(0, d_model, 2):
                        pe[batch, m, i] = 0.0
                        if i+1 == d_model:
                            break
                        pe[batch, m, i+1] = 0.0
                else:
                    if inputs[batch][m] == zero_tok:
                        inputs_lengths[batch] -= 1
                    for i in range(0, d_model, 2):
                        pe[batch, m, i] = math.sin((
                            inputs_lengths[batch])/(10000**(i/d_model)))
                        if i+1 == d_model:
                            break
                        pe[batch, m, i+1] = math.cos((
                            inputs_lengths[batch])/(10000**(i/d_model)))
        pos = torch.from_numpy(pe)
        if torch.cuda.is_available():
            pos = pos.type(torch.cuda.FloatTensor)
        return pos, inputs_lengths

    def length_encoding(self, tgt_vocab, inputs,
            batch_size, max_len, inputs_lengths):

        zero_tok = tgt_vocab.stoi['0']
        pe = []
        for batch in range(batch_size):
            p = []
            for i in range(max_len):
                if inputs_lengths[batch] <= 0:
                    p.append(0)
                else:
                    if inputs[batch][i] == zero_tok:
                        inputs_lengths[batch] -= 1
                    p.append(inputs_lengths[batch])
            pe.append(p)
        pos = torch.tensor(pe)
        if torch.cuda.is_available():
            pos = pos.cuda()
        posemb = self.pos_embedding(pos)
        return posemb, inputs_lengths

    def forward(self, tgt_vocab, inputs=None, input_lengths=None, encoder_hidden=None,
            encoder_outputs=None, encoder_context=None, encoder_action=None,
            function=F.log_softmax, teacher_forcing_ratio=0):
        ret_dict = dict()
        ret_dict[DecoderRNN.KEY_ENCODER_OUTPUTS] = encoder_outputs.squeeze(0)
        if encoder_context is not None:
            ret_dict[DecoderRNN.KEY_ENCODER_CONTEXT] = encoder_context.squeeze(0)
        else:
            ret_dict[DecoderRNN.KEY_ENCODER_CONTEXT] = None

        if encoder_action is not None:
            ret_dict[DecoderRNN.KEY_ENCODER_ACTION] = encoder_action
        else:
            ret_dict[DecoderRNN.KEY_ENCODER_ACTION] = None

        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        # input.shape = batch_size x sequence_length
        # encoder_outputs.shape = batch_size x sequence_length (50) x hidden_size (50 x 2)
        # encoder_hidden = tuple of the last hidden state and the last cell state.
        # Last cell state = number of layers * batch_size * hidden_size
        # Last hidden state = the same as above
        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden,
                                                             encoder_outputs, function, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols
        if use_teacher_forcing:
            pos = None
            if self.position_embedding == "sin":
                pos, _ = self.sin_encoding(tgt_vocab, inputs.cpu().tolist(),
                    batch_size, max_length, input_lengths, self.embedding_size)
            elif self.position_embedding == "length":
                pos, _ = self.length_encoding(tgt_vocab, inputs.cpu().tolist(),
                    batch_size, max_length, input_lengths)
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn, decoder_action, stack = self.forward_step(
                    decoder_input, pos, decoder_hidden, encoder_outputs, di=0, function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            decoder_action = torch.tensor(()).to(device)
            input_len = copy.deepcopy(input_lengths)
            for di in range(max_length):
                decoder_pos = None
                if self.position_embedding == "sin":
                    pos, input_len = self.sin_encoding(tgt_vocab, decoder_input.cpu().tolist(),
                        batch_size, 1, input_len, self.embedding_size)
                    decoder_pos = pos[:, 0].unsqueeze(1)
                elif self.position_embedding == "length":
                    pos, input_len = self.length_encoding(tgt_vocab, decoder_input.cpu().tolist(),
                        batch_size, 1, input_len)
                    decoder_pos = pos[:, 0].unsqueeze(1)

                decoder_output, decoder_hidden, step_attn, action, stack = self.forward_step(
                        decoder_input, decoder_pos, decoder_hidden, encoder_outputs, di, function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                decoder_input = symbols
                if action is not None:
                    decoder_action = torch.cat((decoder_action, action), dim=1)

        if decoder_action is not None:
            ret_dict[DecoderRNN.KEY_DECODER_ACTION] = decoder_action
        else:
            ret_dict[DecoderRNN.KEY_DECODER_ACTION] = None

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length
