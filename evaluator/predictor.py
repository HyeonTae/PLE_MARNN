import torch
from torch.autograd import Variable

class Predictor(object):

    def __init__(self, model, src_vocab, tgt_vocab):
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def get_decoder_features(self, src_seq):
        src_id_seq = torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()

        with torch.no_grad():
            _, _, other = self.model(src_id_seq, self.tgt_vocab, [len(src_seq)])

        #print(self.model.decoder.action_weights.cpu().numpy())
        return other

    def predict(self, src_seq):
        other = self.get_decoder_features(src_seq)

        length = other['length'][0]

        tgt_att_list = []
        encoder_outputs = []
        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        if 'attention_score' in list(other.keys()):
            for i in range(len(other['attention_score'][0][0])):
                tgt_att_list.append([other['attention_score'][di][0].data[i].cpu().numpy() for di in range(length)])
            encoder_outputs = other['encoder_outputs'].cpu().numpy()

        if other['encoder_action'] is not None:
            action = torch.cat((other['encoder_action'],
                other['decoder_action'][:, :length-1, :]), dim=0).cpu().numpy()
        else:
            action = None

        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        return tgt_seq, tgt_att_list, encoder_outputs, action

    def predict_n(self, src_seq, n=1):
        other = self.get_decoder_features(src_seq)

        result = []
        for x in range(0, int(n)):
            length = other['topk_length'][0][x]
            tgt_id_seq = [other['topk_sequence'][di][0, x, 0].data[0] for di in range(length)]
            tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
            result.append(tgt_seq)

        return result
