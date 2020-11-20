#res_copy.py
#run day 2020/11/14
#https://axa.biopapyrus.jp/deep-learning/pytorch/pytorch-vgg16.html 参考サイト
#VGG7ぐらいにしたけどエラー何だっけ
#RuntimeError: Given groups=1, weight of size [256, 256, 3, 3], expected input[13, 512, 1, 1] to have 256 channels, but got 512 channels instead
#Total params: 30.75M
#cifar_copyの44行目らへんのbatchsizeとかlearning-rateとか変えているので注意
#160行目付近のmodel.load_state_dict(torch.load(PATH), strict=False)
#あと画像の入力サイズもネットワークによって合わせるように

import torch.nn as nn
import torch.nn.functional as F
from Convolutional_LSTM_PyTorch.convolution_lstm_att import *
import pdb

def _init_hidden(batch_size, channel_size, height, width):
        return (torch.zeros(batch_size, channel_size, height, width).to(device),
                torch.zeros(batch_size, channel_size, height, width).to(device))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv01 = nn.Conv2d(3, 64, 3)
        self.conv02 = nn.Conv2d(64, 64, 3)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv03 = nn.Conv2d(64, 128, 3)
        self.conv04 = nn.Conv2d(128, 128, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv05 = nn.Conv2d(128, 256, 3)
        self.conv06 = nn.Conv2d(256, 256, 3)
        self.conv07 = nn.Conv2d(256, 256, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        """
        self.conv08 = nn.Conv2d(256, 512, 3)
        self.conv09 = nn.Conv2d(512, 512, 3)
        self.conv10 = nn.Conv2d(512, 512, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        """
        """
        self.conv11 = nn.Conv2d(512, 512, 3)
        self.conv12 = nn.Conv2d(512, 512, 3)
        self.conv13 = nn.Conv2d(512, 512, 3)
        self.pool5 = nn.MaxPool2d(2, 2)
        """


######################################################
        self.att_convlstm_cell = AttentiveConvLSTMCell(
                                          input_size=(23, 23),
                                          input_dim=256,
                                          hidden_dim=256,
                                          att_dim=256,
                                          kernel_size=(3, 3),
                                          bias=True
                                        )

        self.fc0 = nn.Linear(256*23*23, 10)##直し忘れに気をつけて
        self.fc1 = nn.Linear(256*23*23, 10)##ここはFinal PerceptionのFC層なので
        self.fc2 = nn.Linear(256*23*23, 10)##最後の，データセットのクラス数
        self.fc3 = nn.Linear(256*23*23, 10)##
        self.softmax = nn.Softmax(dim=1)

######################################################


######################################################
    def forward(self, x_0, x_1, x_2, x_3):
        x_0 = F.relu(self.conv01(x_0))
        #import pdb;pdb.set_trace()
        x_0 = F.relu(self.conv02(x_0))
        #import pdb;pdb.set_trace()
        x_0 = self.pool1(x_0)
        #import pdb;pdb.set_trace()

        x_0 = F.relu(self.conv03(x_0))
        #import pdb;pdb.set_trace()
        x_0 = F.relu(self.conv04(x_0))
        #import pdb;pdb.set_trace()
        x_0 = self.pool2(x_0)
        #import pdb;pdb.set_trace()

        x_0 = F.relu(self.conv05(x_0))
        #import pdb;pdb.set_trace()
        x_0 = F.relu(self.conv06(x_0))
        #import pdb;pdb.set_trace()
        x_0 = F.relu(self.conv07(x_0))
        #import pdb;pdb.set_trace()
        x_0 = self.pool3(x_0)
        #import pdb;pdb.set_trace()
        """
        x_0 = F.relu(self.conv08(x_0))
        #import pdb;pdb.set_trace()
        x_0 = F.relu(self.conv09(x_0))
        #import pdb;pdb.set_trace()
        x_0 = F.relu(self.conv10(x_0))
        #import pdb;pdb.set_trace()
        x_0 = self.pool4(x_0)
        #import pdb;pdb.set_tra  

        """
        #x_0 = F.relu(self.conv11(x_0))
        #import pdb;pdb.set_trace()
        #x_0 = F.relu(self.conv12(x_0))
        #import pdb;pdb.set_trace()
        #x_0 = F.relu(self.conv13(x_0))
        #x_0 = self.pool5(x_0)
        #import pdb;pdb.set_trace()
        """
        x_0 = self.avepool1(x_0)
        #import pdb;pdb.set_trace()

        # 行列をベクトルに変換
        x_0 = x_0.view(-1, 512 * 7 * 7)
        #import pdb;pdb.set_trace()
        
        x_0 = F.relu(self.fc1(x_0))
        #import pdb;pdb.set_trace()
        x_0 = self.dropout1(x_0)
        #import pdb;pdb.set_trace()
        x_0 = F.relu(self.fc2(x_0))
        #import pdb;pdb.set_trace()
        x_0 = self.dropout2(x_0)
        #import pdb;pdb.set_trace()
        x_0 = self.fc3(x_0)
        #import pdb;pdb.set_trace()
        """
#########################################
        x_1 = F.relu(self.conv01(x_1))
        x_1 = F.relu(self.conv02(x_1))
        x_1 = self.pool1(x_1)

        x_1 = F.relu(self.conv03(x_1))
        x_1 = F.relu(self.conv04(x_1))
        x_1 = self.pool2(x_1)

        x_1 = F.relu(self.conv05(x_1))
        x_1 = F.relu(self.conv06(x_1))
        x_1 = F.relu(self.conv07(x_1))
        x_1 = self.pool3(x_1)
        """
        x_1 = F.relu(self.conv08(x_1))
        x_1 = F.relu(self.conv09(x_1))
        x_1 = F.relu(self.conv10(x_1))
        x_1 = self.pool4(x_1)
        """
#########################################
        x_2 = F.relu(self.conv01(x_2))
        x_2 = F.relu(self.conv02(x_2))
        x_2 = self.pool1(x_2)

        x_2 = F.relu(self.conv03(x_2))
        x_2 = F.relu(self.conv04(x_2))
        x_2 = self.pool2(x_2)

        x_2 = F.relu(self.conv05(x_2))
        x_2 = F.relu(self.conv06(x_2))
        x_2 = F.relu(self.conv07(x_2))
        x_2 = self.pool3(x_2)
        """
        x_2 = F.relu(self.conv08(x_2))
        x_2 = F.relu(self.conv09(x_2))
        x_2 = F.relu(self.conv10(x_2))
        x_2 = self.pool4(x_2)
        """

#########################################
        x_3 = F.relu(self.conv01(x_3))
        x_3 = F.relu(self.conv02(x_3))
        x_3 = self.pool1(x_3)

        x_3 = F.relu(self.conv03(x_3))
        x_3 = F.relu(self.conv04(x_3))
        x_3 = self.pool2(x_3)

        x_3 = F.relu(self.conv05(x_3))
        x_3 = F.relu(self.conv06(x_3))
        x_3 = F.relu(self.conv07(x_3))
        x_3 = self.pool3(x_3)
        """
        x_3 = F.relu(self.conv08(x_3))
        x_3 = F.relu(self.conv09(x_3))
        x_3 = F.relu(self.conv10(x_3))
        x_3 = self.pool4(x_3)
        """

        #################################
        x_init = x_0

        cur_states_init = _init_hidden(x_init.shape[0], x_init.shape[1], x_init.shape[2], x_init.shape[3])  # (batch_size, decoder_dim)
        
        h0, cur_states0, attention_map0 = self.att_convlstm_cell(x_init, cur_states_init)
        h1, cur_states1, attention_map1 = self.att_convlstm_cell(x_1, cur_states0)
        h2, cur_states2, attention_map2 = self.att_convlstm_cell(x_2, cur_states1)
        h3, cur_states3, attention_map3 = self.att_convlstm_cell(x_3, cur_states2)
        #import pdb;pdb.set_trace()

        attention_maps = [attention_map0, attention_map1, attention_map2, attention_map3]
        rx_0 = h0.view(h0.size(0), -1)
        rx_0 = self.fc0(rx_0)
        #import pdb;pdb.set_trace()

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
