from __future__ import print_function, division

import os
import torch
import torchtext
import itertools

from loss.loss import NLLLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Evaluator(object):
    def __init__(self, loss, batch_size):
        self.loss = loss
        self.EOS_SLICE = torch.ones((batch_size, 1)).long().to(device) * 1
        self.PAD_SLICE = torch.ones((batch_size, 1)).long().to(device) * 0
        self.batch_size = batch_size

    def evaluate(self, model, data):
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        match_sentence = 0
        total_lengths = 0

        condition_positive = 0
        prediction_positive = 0
        true_positive = 0

        check_sentence = True

        num_train, num_validation, num_test = data.data_size


        #device = None if torch.cuda.is_available() else -1
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #batch_iterator = torchtext.data.BucketIterator(
        #    dataset=data, batch_size=self.batch_size,
        #    sort=True, sort_key=lambda x: len(x.src),
        #    device=device, train=False)

        tgt_vocab = data.get_tl_dictionary()
        pad = 0
        eos = 1
        zero = 0
        unk = 0
        #print(tgt_vocab.stoi)

        with torch.no_grad():
            for i in range(int(num_validation / self.batch_size)):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size

                x, x_len, y, y_len = data.get_batch(start, end, which='valid')
                input_variables = torch.tensor(x.T.tolist())
                target_variables = torch.tensor(y.T.tolist())
                input_lengths = x_len.tolist()
                target_lengths = y_len + 1

                if torch.cuda.is_available():
                    input_variables = input_variables.cuda()
                    target_variables = target_variables.cuda()

                target_variables = torch.cat((self.EOS_SLICE, target_variables), dim=1)


                #input_variables, input_lengths = getattr(batch, 'src')
                #target_variables = getattr(batch, 'tgt')

                # Drop last
                #if input_variables.size(0) != self.batch_size:
                #    continue

                decoder_outputs, decoder_hidden, other = model(
                        input_variables, input_lengths, target_variables, target_lengths)
                correct_list = []
                # Evaluation
                seqlist = other['sequence']
                for step, step_output in enumerate(decoder_outputs):
                    target = target_variables[:, step + 1]
                    loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                    predict = seqlist[step].view(-1)
                    non_padding = target.ne(pad)
                    correct = predict.eq(target).masked_select(non_padding).sum().item()
                    correct_list.append(predict.eq(target).masked_select(non_padding).tolist())

                    CP = target.ne(zero).eq(target.ne(eos)).masked_select(non_padding)
                    PP = predict.ne(zero).eq(predict.ne(eos)).masked_select(non_padding)
                    c_mask = target.ne(pad).eq(target.ne(eos)).eq(target.ne(unk)).eq(target.ne(zero))
                    TP = target.masked_select(c_mask).eq(predict.masked_select(c_mask))

                    match += correct
                    total += non_padding.sum().item()

                    condition_positive += CP.sum().item()
                    prediction_positive += PP.sum().item()
                    true_positive += TP.sum().item()
                q = list(itertools.zip_longest(*correct_list))

                for i in q:
                    check_sentence = False
                    for j in i:
                        if(j == 0):
                            check_sentence = True
                    if(check_sentence == False):
                        match_sentence += 1
                total_lengths += len(input_lengths)

        if total == 0:
            character_accuracy = 0
            sentence_accuracy = 0
        else:
            character_accuracy = match / total
            sentence_accuracy = match_sentence / total_lengths

        if condition_positive == 0:
            recall = 0
        else:
            recall = true_positive / condition_positive

        if prediction_positive == 0:
            precision = 0
        else:
            precision = true_positive / prediction_positive

        if precision == 0 and recall == 0:
            f1_score = 0
        else:
            f1_score = 2.0 * ((precision * recall) / (precision + recall))

        return loss.get_loss(), character_accuracy, sentence_accuracy, f1_score
