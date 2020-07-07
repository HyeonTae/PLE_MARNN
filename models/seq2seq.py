import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from models.encoderRNN import EncoderRNN
from models.decoderRNN import DecoderRNN

class Seq2seq(nn.Module):

    def __init__(self, config, src_vocab_size, tgt_vocab_size, sos_id, eos_id,
                 pretrained_pos_weight=None):
        super(Seq2seq, self).__init__()

        if config["position_embedding"] == "length":
            if pretrained_pos_weight is None:
                self.pos_embedding = nn.Embedding(config["max_len"], config["embedding_size"])
                self.pos_embedding.weight.requires_grad = config["update_embedding"]
            else:
                self.pos_embedding = nn.Embedding.from_pretrained(
                        torch.from_numpy(pretrained_pos_weight))
                self.pos_embedding.weight.requires_grad = False
        else:
            self.pos_embedding = None

        self.encoder = EncoderRNN(vocab_size=src_vocab_size,
                                  max_len=config["max_len"],
                                  hidden_size=config["hidden_size"],
                                  embedding_size=config["embedding_size"],
                                  input_dropout_p=config["input_dropout_p"],
                                  dropout_p=config["dropout_p"],
                                  position_embedding=config["position_embedding"],
                                  pos_embedding=self.pos_embedding,
                                  n_layers=config["n_layers"],
                                  bidirectional=config["bidirectional"],
                                  rnn_cell=config["rnn_cell"],
                                  variable_lengths=config["variable_lengths"],
                                  embedding=config["embedding"],
                                  update_embedding=config["update_embedding"],
                                  get_context_vector=config["get_context_vector"],
                                  pos_add=config["pos_add"],
                                  #use_memory=config["use_memory"],
                                  use_memory=None,
                                  memory_dim=config["memory_dim"])
        self.decoder = DecoderRNN(vocab_size=tgt_vocab_size,
                                  max_len=config["max_len"],
                                  hidden_size=config["hidden_size"]*2 if config["bidirectional"] else config["hidden_size"],
                                  embedding_size=config["embedding_size"],
                                  sos_id=sos_id,
                                  eos_id=eos_id,
                                  input_dropout_p=config["input_dropout_p"],
                                  dropout_p=config["dropout_p"],
                                  position_embedding=config["position_embedding"],
                                  pos_embedding=self.pos_embedding,
                                  n_layers=config["n_layers"],
                                  bidirectional=config["bidirectional"],
                                  rnn_cell=config["rnn_cell"],
                                  use_attention=config["use_attention"],
                                  attn_layers=config["attn_layers"],
                                  hard_attn=config["hard_attn"],
                                  pos_add=config["pos_add"],
                                  use_memory=config["use_memory"],
                                  memory_dim=config["memory_dim"])
        self.decode_function = F.log_softmax

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, tgt_vocab, input_lengths=None,
            target_variable=None, teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden, encoder_context, encoder_action, encoder_memory = self.encoder(
                                                                    input_variable, input_lengths)
        result = self.decoder(tgt_vocab=tgt_vocab,
                              inputs=target_variable,
                              input_lengths=input_lengths,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              encoder_context=encoder_context,
                              encoder_action=encoder_action,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result
