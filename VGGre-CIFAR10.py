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

import os
import torch
import torchvision
from VGGre import VGGre16

trainSet = torchvision.datasets.CIFAR10(root="../data", train=True, download=True)
#print(trainSet.data.dtype, trainSet.data.shape) #uint8 (50000, 32, 32, 3)
#print(len(trainSet)) #50000
trainSet.transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((trainSet.data.reshape(-1, 3).mean(0)/255.0), (trainSet.data.reshape(-1, 3).std(0)/255.0)),
])

testSet = torchvision.datasets.CIFAR10(root="../data", train=False, download=True)
#print(testSet.data.dtype, testSet.data.shape) #uint8 (10000, 32, 32, 3)
#print(len(testSet)) #10000
testSet.transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((trainSet.data.reshape(-1, 3).mean(0)/255.0), (trainSet.data.reshape(-1, 3).std(0)/255.0)),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#####################################################################################

model = VGGre16(num_classes=len(trainSet.classes)).to(device)
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
if os.path.exists('VGGre16-CIFAR10.pkl'):
    print("Loaded VGGre16-CIFAR10.pkl")
    pkl = torch.load('VGGre16-CIFAR10.pkl')
    model.load_state_dict(pkl['model.state_dict'])
    #optimizer.load_state_dict(pkl['optimizer.state_dict'])
    #scheduler.load_state_dict(pkl['scheduler.state_dict'])

print(model)
print(optimizer)
for parameter in model.parameters():
    print(parameter.shape)
#print(sum(p.numel() for p in model.parameters() if p.requires_grad))

trainItr = torch.utils.data.DataLoader(trainSet, batch_size=100, shuffle=True)
testItr = torch.utils.data.DataLoader(testSet, batch_size=100, shuffle=False)
loss_func = torch.nn.CrossEntropyLoss()

while (scheduler.last_epoch < 1000):
    torch.cuda.empty_cache()

    model.train()
    with torch.enable_grad():
        trainLoss = 0.0
        trainAccuracy = 0.0
        for step, (train_data, train_label) in enumerate(trainItr):
            optimizer.zero_grad()
            train_data = train_data.to(device)
            train_label = train_label.to(device)

            label = model(train_data)
            loss = loss_func(label, train_label)
            #print('Step: %03d' %step, 'Train Loss: %.4f' %loss.data)
            trainLoss += loss.data
            trainAccuracy += (label.argmax(-1) == train_label).sum()

            loss.backward()
            optimizer.step()
        trainLoss /= len(trainItr)
        trainAccuracy /= len(trainSet)

    #continue
    model.eval()
    with torch.no_grad():
        testLoss = 0.0
        testAccuracy = 0.0
        for step, (test_data, test_label) in enumerate(testItr):
            test_data = test_data.to(device)
            test_label = test_label.to(device)

            label = model(test_data)
            loss = loss_func(label, test_label)
            testLoss += loss.data
            testAccuracy += (label.argmax(-1) == test_label).sum()
        testLoss /= len(testItr)
        testAccuracy /= len(testSet)

    scheduler.step(trainLoss)
    print(f'Epoch: {scheduler.last_epoch:02} ; Train|Test Loss: {trainLoss:.4f}|{testLoss:.4f} ; Train|Test Accuracy: {trainAccuracy:.4f}|{testAccuracy:.4f}')
    if (scheduler.last_epoch%100):
        continue

    torch.save(
        {
            'model.state_dict': model.state_dict(),
            'optimizer.state_dict': optimizer.state_dict(),
            'scheduler.state_dict': scheduler.state_dict()
            },
        'VGGre16-CIFAR10-(%03d,%.04f,%.04f).pkl' %(scheduler.last_epoch, trainAccuracy, testAccuracy)
        )
