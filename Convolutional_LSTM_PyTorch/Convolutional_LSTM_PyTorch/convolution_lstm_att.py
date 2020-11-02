"""
# usecase
model = ConvLSTM(input_size=(height, width),
                 input_dim=channels,
                 hidden_dim=[64, 64, 128],
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True
                 bias=True,
                 return_all_layers=False)
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AttentiveConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, att_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(AttentiveConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.att_dim = att_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.W_i = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=kernel_size, padding=1, bias=True)
        self.W_f = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=kernel_size, padding=1, bias=True)
        self.W_c = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=kernel_size, padding=1, bias=True)
        self.W_o = nn.Conv2d(self.input_dim, self.hidden_dim, kernel_size=kernel_size, padding=1, bias=True)

        self.U_i = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, padding=1, bias=True)
        self.U_f = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, padding=1, bias=True)
        self.U_c = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, padding=1, bias=True)
        self.U_o = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, padding=1, bias=True)

        ### attention module
        self.att_w = nn.Conv2d(self.att_dim, self.att_dim, kernel_size=kernel_size, padding=1, bias=True)
        self.att_u = nn.Conv2d(self.input_dim, self.att_dim, kernel_size=kernel_size, padding=1, bias=True)
        self.att_v = nn.Conv2d(self.att_dim, 1, kernel_size=kernel_size, padding=1, bias=False)
        #self.att_v = nn.Conv2d(self.att_dim, 1, kernel_size=kernel_size, padding=0, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        init.normal_(self.W_i.weight)
        init.normal_(self.W_f.weight)
        init.normal_(self.W_c.weight)
        init.normal_(self.W_o.weight)

        init.orthogonal_(self.U_i.weight)
        init.orthogonal_(self.U_f.weight)
        init.orthogonal_(self.U_c.weight)
        init.orthogonal_(self.U_o.weight)

        init.normal_(self.att_w.weight)
        init.normal_(self.att_u.weight)
        init.zeros_(self.att_v.weight)

        init.zeros_(self.W_i.bias)
        init.zeros_(self.W_f.bias)
        init.zeros_(self.W_c.bias)
        init.zeros_(self.W_o.bias)

        init.zeros_(self.U_i.bias)
        init.zeros_(self.U_f.bias)
        init.zeros_(self.U_c.bias)
        init.zeros_(self.U_o.bias)

        init.zeros_(self.att_w.bias)
        init.zeros_(self.att_u.bias)


    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        ### Attentive Module
        x_shape = [int(s) for s in input_tensor.size()]

        e = self.att_v(torch.tanh(self.att_w(h_cur) + self.att_u(input_tensor))) # 10 512 30 40

        a = torch.softmax(e.view(x_shape[0], x_shape[2]*x_shape[3]), dim=1)
        #import pdb;pdb.set_trace()
        #a = F.sigmoid(e.view(x_shape[0], x_shape[2]*x_shape[3])) ########################変更箇所
        #import pdb;pdb.set_trace()
        a = a.view(x_shape[0], 1, x_shape[2], x_shape[3])
        att_pred = a.view(x_shape[0], x_shape[2], x_shape[3])

        x_tilde = input_tensor * a.repeat(1, x_shape[1], 1, 1)

        ### Attentive Module End

        # ここから
        x_i = self.W_i(x_tilde)
        x_f = self.W_f(x_tilde)
        x_c = self.W_c(x_tilde)
        x_o = self.W_o(x_tilde)

        i = torch.sigmoid(x_i + self.U_i(h_cur))
        f = torch.sigmoid(x_f + self.U_f(h_cur))
        c = f * c_cur + i * torch.tanh(x_c + self.U_c(h_cur))
        o = torch.sigmoid(x_o + self.U_o(h_cur))

        c_next = c
        h_next = o * torch.tanh(c)

        return h_next, [h_next, c_next], a

class AttentiveConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, att_dim, kernel_size, bias=True):
        super(AttentiveConvLSTM, self).__init__()

        self.kernel_size = (3, 3)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.height = 30
        self.width = 40

        self.cell = AttentiveConvLSTMCell(input_size,
                                          input_dim,
                                          hidden_dim,
                                          att_dim,
                                          self.kernel_size,
                                          bias)

    def _init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device))

    def forward(self, input_tensor, hidden_state=None):
        cur_states = self._init_hidden(input_tensor.size(0))

        for t in range(4):
            last_output, cur_states, attention_map = self.cell(input_tensor[:,t,:,:,:], cur_states)

        return last_output, attention_map
  
