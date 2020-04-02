import torch
import torch.nn as nn
import torch.nn.functional as F

class HardAttention(nn.Module):
    def __init__(self, dim):
        super(HardAttention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, output, context, di):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        attn[:, :, :] = 0.0
        if output.size(1) is not 1:
            for i in range(attn.size(1)):
                if attn.size(2) is i:
                    attn[:, i, i - 1] = 1.0
                else:
                    attn[:, i, i] = 1.0
        else:
            if self.mask is not None:
                attn.data.masked_fill_(self.mask, -float('inf'))
            if attn.size(2) is di:
                attn[:, :, di - 1] = 1.0
            elif attn.size(2) > di:
                attn[:, :, di] = 1.0
        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn
