from torch import nn
import torch as t
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

train_dataset = datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)
#############################   model  #######################
class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1,64,3,1),  ##26*26,64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 64, 3, 1), ##24*24, 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),     ##12*12, 64

            nn.Conv2d(64, 128, 3, 1),  ##10*10, 128
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 128, 3, 1),  ##8*8, 128
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 512, 3, 1),  ##6*6, 512
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.MaxPool2d(2, 2),  ##3*3, 512
        )
        self.classify = nn.Sequential(
            nn.Linear(9*512,9*512),
            nn.Linear(9*512,512),
            nn.Linear(512,10)
        )

    def forward(self,x):
        x=x.reshape(-1,1,28,28)
        out1 = self.feature(x)
        out1 = out1.view(out1.size(0),-1)
        out = self.classify(out1)
        return out


#############################   parameter  #######################
epoch = 40
learning_rate = 1e-3
weight_decay=0
if t.cuda.is_available():
    device = t.device("cuda")
else :
    device = t.device("cpu")
model = VGG()
#model.load_state_dict(t.load("1.pth"))
model.to(device)

writer = SummaryWriter('./board')
#dataiter = iter(train_loader)
#images, labels = dataiter.next()
#img_grid = torchvision.utils.make_grid(images)
#writer.add_graph('imag_classify',img_grid)

#writer.close()
#############################   optim  #######################
adam = optim.Adam(model.parameters(),learning_rate,weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
##############################  main ########################

for e_poch in range(epoch):
    model.train()
    num=1
    eval_loss = 0
    print("############################ epoch: %i #############################"%e_poch)
    for img,label in train_loader:
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        loss=criterion(out,label)
        eval_loss += loss
        adam.zero_grad()
        loss.backward()
        adam.step()

        num+=1
    writer.add_scalar('train_loss', eval_loss .item()/len(train_loader), e_poch)
    print("loss: %f" % (eval_loss.item()))

t.save(model.state_dict(), "2.pth")

model.eval()
eval_acc = 0
num=1
print("############################ eval #############################")
for img,label in test_loader:
    img = img.to(device)
    label = label.to(device)
    out = model(img)
    loss = criterion(out, label)
    print("index: %i, loss: %f" % (num, loss.item()))
    _, pred = t.max(out,1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
    num+=1
print("accuracy: %f"%(eval_acc/len(test_dataset)))
