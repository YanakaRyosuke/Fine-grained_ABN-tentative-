import torch.nn as nn
import torch.nn.functional as F
from Convolutional_LSTM_PyTorch.convolution_lstm_att import *

def _init_hidden(batch_size, channel_size, height, width):
        return (torch.zeros(batch_size, channel_size, height, width).to(device),
                torch.zeros(batch_size, channel_size, height, width).to(device))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3) 
        #self.pool1 = nn.MaxPool2d(2, 2) 
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3) 
        #self.pool2 = nn.MaxPool2d(2, 2) 
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3) 
        self.pool3 = nn.MaxPool2d(2, 2) 
        self.bn3 = nn.BatchNorm2d(64)

        self.att_convlstm_cell = AttentiveConvLSTMCell(
                                          input_size=(13, 13),
                                          input_dim=64,
                                          hidden_dim=64,
                                          att_dim=64,
                                          kernel_size=(3, 3),
                                          bias=True
                                        )

        self.fc0 = nn.Linear(64*13*13, 10)##直し忘れに気をつけて
        self.fc1 = nn.Linear(64*13*13, 10)##
        self.fc2 = nn.Linear(64*13*13, 10)##
        self.fc3 = nn.Linear(64*13*13, 10)##
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_0, x_1, x_2, x_3):
        
        x_0 = F.relu(self.bn1(self.conv1(x_0)))
        #x_0 = F.relu(self.bn2(self.pool2(self.conv2(x_0))))
        x_0 = F.relu(self.bn2(self.conv2(x_0)))
        x_0 = F.relu(self.bn3(self.pool3(self.conv3(x_0))))

        x_1 = F.relu(self.bn1(self.conv1(x_1)))
        #x_1 = F.relu(self.bn2(self.pool2(self.conv2(x_1))))
        x_1 = F.relu(self.bn2(self.conv2(x_1)))
        x_1 = F.relu(self.bn3(self.pool3(self.conv3(x_1))))

        x_2 = F.relu(self.bn1(self.conv1(x_2)))
        #x_2 = F.relu(self.bn2(self.pool2(self.conv2(x_2))))
        x_2 = F.relu(self.bn2(self.conv2(x_2)))
        x_2 = F.relu(self.bn3(self.pool3(self.conv3(x_2))))

        x_3 = F.relu(self.bn1(self.conv1(x_3)))
        #x_3 = F.relu(self.bn2(self.pool2(self.conv2(x_3))))
        x_3 = F.relu(self.bn2(self.conv2(x_3)))
        x_3 = F.relu(self.bn3(self.pool3(self.conv3(x_3)))) #weight share

        x_init = x_0

        cur_states_init = _init_hidden(x_init.shape[0], x_init.shape[1], x_init.shape[2], x_init.shape[3])  # (batch_size, decoder_dim)
        
        h0, cur_states0, attention_map0 = self.att_convlstm_cell(x_init, cur_states_init)
        h1, cur_states1, attention_map1 = self.att_convlstm_cell(x_1, cur_states0)
        h2, cur_states2, attention_map2 = self.att_convlstm_cell(x_2, cur_states1)
        h3, cur_states3, attention_map3 = self.att_convlstm_cell(x_3, cur_states2)

        attention_maps = [attention_map0, attention_map1, attention_map2, attention_map3]
        rx_0 = h0.view(h0.size(0), -1)
        rx_0 = self.fc0(rx_0)

        rx_1 = h1.view(h1.size(0), -1)
        rx_1 = self.fc1(rx_1)

        rx_2 = h2.view(h2.size(0), -1)
        rx_2 = self.fc2(rx_2)

        rx_3 = h3.view(h3.size(0), -1)
        rx_3 = self.fc3(rx_3)

        soft = rx_0 + rx_1 + rx_2 + rx_3
        soft = self.softmax(soft)
        #import pdb;pdb.set_trace()

        return soft, attention_maps

