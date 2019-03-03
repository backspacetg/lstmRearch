import lstmTest
import torch as t
import torch.nn as nn
from torch import optim
from copyTask import Net
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

#超参数
EPOCH = 2
batch = 4
learnRate = 0.01
inputSize = 28
T = 28
hideSize = 64
outputSize = 10

#数据集导入
#定义导入时的变换与归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
#训练集的定义
trainset = tv.datasets.MNIST(
    root='data/',
    train=True,
    download=True,
    transform=transform)
#训练集对应的dataloader
trainloader = t.utils.data.DataLoader(
                    trainset,
                    batch_size=batch,
                    shuffle=True,
                    num_workers=2)
#测试集
testset = tv.datasets.MNIST(
    root='data/',
    train = False,
    download = True,
    transform = transform)
#测试集dataloader
testloader = t.utils.data.DataLoader(
                    testset,
                    batch_size=4,
                    shuffle=True,
                    num_workers=2)

#定义可训练变量
net = Net(inputSize,hideSize,outputSize)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=learnRate)
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

# 加载训练结果
#net.load_state_dict(t.load('save/netLSTMnist.pkl'))
#optimizer.load_state_dict(t.load('save/optimizerLSTMnist.pkl'))

#时间序列的依次输入操作
def timeSequenceInput(inputnet,inputimages):
    # 输入格式：(timestep,batchsize,inputsize)
    inputs = (inputimages.squeeze()).transpose(0,1)
    h = t.zeros((inputs[0].size())[0],hideSize)
    c = t.zeros((inputs[0].size())[0],hideSize)
    for timestep in inputs:
        #print(timestep.size(),h.size(),c.size())
        outputs,h,c = inputnet(timestep,h,c)
    return outputs

#训练
if __name__ == '__main__':
    t.set_num_threads(8)
    runningloss = 0.0
    print('start')
    for epoch in range(EPOCH):
        for i,data in enumerate(trainloader,1):
            #每个batch开始时将梯度清零
            optimizer.zero_grad()
            #初始化细胞状态和初始输入为零向量
            #h = t.zeros(batch,hideSize)
            #c = t.zeros(batch,hideSize)

            # 输入格式：(timestep,batchsize,inputsize)
            inputs,labels = data
            #inputs = (inputs.squeeze()).transpose(0,1)

            #对于时间序列中每一个时间点，传入输入数据，并只取最后的输出
            #for timestep in inputs:
                #outputs,h,c = net(timestep,h,c)

            outputs = timeSequenceInput(net,inputs)
            loss = criterion(outputs,labels)
            #print(loss.item())
            runningloss += loss.item()
            loss.backward()
            optimizer.step()

            #每2000个打印一次损失
            if i%1000 == 0:
                print('epoch: %d train: %d loss: %.3f'%(epoch+1,i+1,runningloss/2000))
                runningloss = 0
    t.save(net.state_dict(), 'save/netLSTMnist.pkl')
    t.save(optimizer.state_dict(), 'save/optimizerLSTMnist.pkl')
    print('finished')

    dataiter = iter(testloader)
    for i in range(2):
        images,labels = dataiter.next()
        print(labels)
        #plt.imshow(show(tv.utils.make_grid((images+1)/2)))
        #plt.show()
        outputs = timeSequenceInput(net,images)
        _,predicted = t.max(outputs,1)
        print(predicted)
    #在整个测试集上实验正确率
    correct = 0
    total = 0
    for i,data in enumerate(testloader,0):
        images,labels = data
        outputs = timeSequenceInput(net,images)
        _,predicted = t.max(outputs,1)
        total += batch
        correct += t.sum((predicted==labels))

    print('correctRate: ',100*correct.data.numpy()/total,'%')











