'''
Python program source code
for research article "VGGreNet: A Light-Weighted VGGNet with Reused Convolutional Set"

Version 1.0
(c) Copyright 2020 Ka-Hou Chan <chankahou (at) ipm.edu.mo>

The python program source code is free software: you can redistribute
it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

The python program source code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License
along with the Kon package.  If not, see <http://www.gnu.org/licenses/>.
'''

import torch

#####################################################################################

class VGGre11(torch.nn.Module):
    def __init__(self, in_features=3, num_classes=1000):
        super(VGGre11, self).__init__()

        self.conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 64, kernel_size=3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True),
            )

        self.reConv2d = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512, 4096), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, 4096), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes),
            )

    def forward(self, input):
        feature = self.conv2d(input)
        while (feature.size(2)>1 or feature.size(3)>1):
            feature = self.reConv2d(feature)
        return self.classifier(feature)

#####################################################################################

class VGGre13(torch.nn.Module):
    def __init__(self, in_features=3, num_classes=1000):
        super(VGGre13, self).__init__()

        self.conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 64, kernel_size=3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True),
            )

        self.reConv2d = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512, 4096), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, 4096), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes),
            )

    def forward(self, input):
        feature = self.conv2d(input)
        while (feature.size(2)>1 or feature.size(3)>1):
            feature = self.reConv2d(feature)
        return self.classifier(feature)

#####################################################################################

class VGGre16(torch.nn.Module):
    def __init__(self, in_features=3, num_classes=1000):
        super(VGGre16, self).__init__()

        self.conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 64, kernel_size=3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True),
            )

        self.reConv2d = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512, 4096), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, 4096), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes),
            )

    def forward(self, input):
        feature = self.conv2d(input)
        while (feature.size(2)>1 or feature.size(3)>1):
            feature = self.reConv2d(feature)
        return self.classifier(feature)

#####################################################################################
    
class VGGre19(torch.nn.Module):
    def __init__(self, in_features=3, num_classes=1000):
        super(VGGre19, self).__init__()

        self.conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 64, kernel_size=3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True),
            )

        self.reConv2d = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512, 4096), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, 4096), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes),
            )

    def forward(self, input):
        feature = self.conv2d(input)
        while (feature.size(2)>1 or feature.size(3)>1):
            feature = self.reConv2d(feature)
        return self.classifier(feature)
