# notes on the SmokeNet archetecure
'''

'''

SmokeNet = [64 ]

class SmokeNet(nn.module):
    def __init__(self, in_channels=3, num_classes=6):
        super(SmokeNet, self).__init__()

        self.in_channels = in_channels
        self.conv_layers = create_conv_layers(SmokeNet_convs)

        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(7,7), stride=(1,1))
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64, kernel_size=(1,1), stride=(1,1),padding(0,0))
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=(3,3), stride=(1,1),padding(0,0))

        def forward(self, x):
            pass

        def create_conv_layers(self, architecture):
            pass

