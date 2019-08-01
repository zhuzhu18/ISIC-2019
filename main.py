import torch
from .resnet50 import resnet50
import torch.backends.cudnn as cudnn
import argparse
import os
from .utils import progress_bar
from .dataset import ISIC
from torchvision import transforms


device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch dog vs cat Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch-size', default=16, type=int, help='minibatch size')

args = parser.parse_args()

# data
print("==> preparing data..")
transform_train = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

img_dir = '/media/zhuzhu/ec114170-f406-444f-bee7-a3dc0a86cfa2/dataset/ISIC_2019_Training_Input'
truth_csv = '/media/zhuzhu/ec114170-f406-444f-bee7-a3dc0a86cfa2/dataset/ISIC_2019/ISIC_2019_Training_GroundTruth.csv'

train_set = ISIC(img_dir, truth_csv, transform=transform_train)
test_set = []             # 测试集的数据还未写

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = []          # 测试集的数据还未写

classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']

# model
print('==> building model..')
net = resnet50().to(device)
net.load_state_dict(torch.load('resnet50-19c8e357.pth'))      # 下载resnet50的预训练模型到当前项目文件夹内

channel_in = net.fc.in_features           # 最后的全连接层通道数
net.fc = torch.nn.Linear(channel_in, len(classes))          # 修改最后的全连接层输出的类别数,默认是imagenet数据集的1000类

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    correct = 0
    total = 0
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predict.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss : %.3f | Acc : %.3f%%(%d/%d)'%
                     (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        _, predict = torch.max(outputs, 1)
        correct += torch.eq(predict, targets).sum().item()
        test_loss += loss.item()
        total += targets.size(0)
        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # save checkpoints
    acc = 100.*correct / total
    if acc > best_acc:
        print('Saving checkpoint...')
        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')
        torch.save({
            'net':net.state_dict(),
            'epoch':epoch,
            'acc':acc
        }, 'checkpoint/ckpt.pth')
        best_acc = acc

start_epoch = 0
if args.resume:
    # load checkpoint
    print('==> resuming from checkpoint')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('checkpoint/ckpt.pth')
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['net'])

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    # test(epoch)