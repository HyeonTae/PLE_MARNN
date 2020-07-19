import torch
import torch.nn as nn
import torch.nn.functional as F

class HardAttention(nn.Module):
    def __init__(self, dim, insert_tok):
        super(HardAttention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.insert_tok = insert_tok
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def init_inserted(self, batch_size):
        self.inserted = [0 for i in range(batch_size)]

    def forward(self, input_var, output, context, di):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        attn[:, :, :] = 0.0
        if output.size(1) != 1:
            for batch in range(batch_size):
                for i in range(attn.size(1)):
                    if i == 0:
                        attn[batch, i, i] = 1.0
                    else:
                        if attn.size(2) >= i-self.inserted[batch]:
                            if input_var[batch][i] in self.insert_tok:
                                self.inserted[batch] += 1
                            attn[batch, i, i-self.inserted[batch]-1] = 1.0
        else:
            if self.mask is not None:
                attn.data.masked_fill_(self.mask, -float('inf'))
            for batch in range(batch_size):
                if di == 0:
                    attn[batch, :, di] = 1.0
                else:
                    if attn.size(2) >= di-self.inserted[batch]:
                        if input_var[batch][0] in self.insert_tok:
                            self.inserted[batch] += 1
                        attn[batch, :, di-self.inserted[batch]-1] = 1.0
        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn
