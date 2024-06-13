import torch
# torch.use_deterministic_algorithms(True, warn_only=True)
import torch.nn as nn

import config.config_train as config

# from torchsummary import summary



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

activations = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "selu": nn.SELU(),
    "elu": nn.ELU(),
    "LeakyReLU": nn.LeakyReLU(0.1),
    "rrelu": nn.RReLU(0.1, 0.3),
}


sequence_length = config.params["genmap_colNum"]
input_size = config.params["genmap_rowNum"]


dropout = nn.Dropout(config.params["dropout"])
activation_function = activations[config.params["acf_indx"]]
bottleneck = config.params["bottleneck"]
last_layer = config.params["last_layer"]
num_class = config.params["num_class"]


class Discriminator_MLP(nn.Module):
    def __init__(self):
        super(Discriminator_MLP, self).__init__()
        dp = 0.3  # 0.3
        lkr = 0.1  # 0.2
        first_layer = 1024
        second_layer = int(first_layer / 2)  # 512
        third_layer = int(second_layer / 2)  # 256
        self.model = nn.Sequential(
            nn.Linear(input_size * sequence_length, first_layer),
            nn.LeakyReLU(lkr),
            nn.Dropout(dp),
            nn.Linear(first_layer, second_layer),
            nn.LeakyReLU(lkr),
            nn.Dropout(dp),
            nn.Linear(second_layer, third_layer),
            nn.LeakyReLU(lkr),
            nn.Dropout(dp),
            nn.Linear(third_layer, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("LayerNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator_Conv(nn.Module):
    def __init__(self):
        super(Discriminator_Conv, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.sequence_length = sequence_length
        self.input_size = input_size

        Filter_num_1 = 8
        kernel_size_1 = 7
        Filter_num_2 = 32
        kernel_size_2 = 3
        Filter_num_3 = 1
        kernel_size_3 = 3
        lkr = 0.2
        dp = 0.1

        self.norm1 = nn.LayerNorm(self.input_size, eps=1e-6)
        self.block_conv1 = nn.Sequential(
            nn.Conv2d(
                1, Filter_num_1, kernel_size_1, stride=1, padding=3
            ),  # (input_dim + 2*padding_side - filter) // stride + 1
            nn.LeakyReLU(lkr, inplace=True),
            nn.Dropout(dp),
            nn.AvgPool2d(2, stride=2),
        )
        self.norm2 = nn.LayerNorm(int(self.input_size / 2), eps=1e-6)
        self.block_conv2 = nn.Sequential(
            nn.Conv2d(Filter_num_1, Filter_num_2, kernel_size_2, stride=1, padding=1),
            nn.LeakyReLU(lkr, inplace=True),
            nn.Dropout(dp),
            # int(self.sequence_length/2)xint(self.input_size/2)
            nn.AvgPool2d(2, stride=2),
            # int(self.sequence_length/4)xint(self.input_size/4)
        )
        self.norm3 = nn.LayerNorm(int(self.input_size / 4), eps=1e-6)
        self.block_conv3 = nn.Sequential(
            # int(self.sequence_length/4)xint(self.input_size/4)
            nn.Conv2d(Filter_num_2, Filter_num_3, kernel_size_3, stride=1, padding=1),
            nn.LeakyReLU(lkr, inplace=True),
            nn.Dropout(dp),
            # int(self.sequence_length/4)xint(self.input_size/4)
            nn.AvgPool2d(
                (int(self.sequence_length / 4), int(self.input_size / 4)), stride=1
            ),
        )

        self.apply(weights_init)
        self.avgpool = nn.AvgPool2d(2, stride=2)

    def forward(self, input):
        input = input.reshape(len(input), 1, self.sequence_length, self.input_size)
        # summary(self.model,input)
        out1 = self.block_conv1(self.norm1(input))
        input_2 = self.avgpool(input) + out1
        out2 = self.block_conv2(self.norm2(input_2))
        out = self.block_conv3(self.norm3(out2))
        out = out.reshape(out.shape[0], 1)
        out = self.sigmoid(out)
        return out
