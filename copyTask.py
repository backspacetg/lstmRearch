import lstmTest
import numpy as np
import torch as t
from torch import optim
import torch.nn as nn
import math
import visdom


t.manual_seed(1000)
T = 20
trainSize = 100000
testSize = 100
inputSize = 1
outputSize = 9
hideSize = 32
batchSize = 100
def dataGenerator(T):
    item = [1,2,3,4,5,6,7,8,0,9]
    ranks = np.random.randint(8,size=10)
    inputs = []
    #必须是一维tensor的数组，否则没有办法使用torch.cat函数
    for r in ranks:
        inputs.append([item[r]])
    for i in range(T-1):
        inputs.append([item[8]])
    inputs.append([item[9]])
    for j in range(10):
        inputs.append([item[8]])
    outputs = []
    for ii in range(T+10):
        outputs.append([item[8]])
    for rr in ranks:
        outputs.append([item[rr]])

    return t.FloatTensor(inputs), t.LongTensor(outputs) #返回longTensor原因是交叉熵损失函数的标签项只能是long的

def datasetGenerator(T,size):
    data = []
    labels = []
    for i in range(size):
        x,y = dataGenerator(T)
        data.append(x)
        labels.append(y)
    data = t.stack(data)
    labels = t.stack(labels)
    return data,labels.long()


class Net(nn.Module):
    def __init__(self,inputSize,hideSize,outputSize):
        super(Net,self).__init__()
        self.lstm = lstmTest.LstmCell(inputSize,hideSize)
        self.fc = nn.Linear(hideSize,outputSize)
    def forward(self,x,h,c):
        hideState,cellState = self.lstm(x,h,c)
        output = self.fc(hideState)
        return output,hideState,cellState


if __name__ == '__main__':
    t.set_num_threads(8)
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    net = Net(inputSize,hideSize,outputSize).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.001)
    #加载训练结果
    #net.load_state_dict(t.load('save/net2.pkl'))
    #ptimizer.load_state_dict(t.load('save/optimizer2.pkl'))
    #生成训练集
    trainInput, supposedOutputs = datasetGenerator(T,trainSize)
    trainInput = trainInput.to(device)
    supposedOutputs = supposedOutputs.to(device)
    #生成测试集
    testInputs,testOutputs = datasetGenerator(T,testSize)
    testInputs = testInputs.to(device)
    testOutputs = testOutputs.to(device)
    #可视化
    #vis = visdom.Visdom(env=u'copyTask') #指定字符串为Unicode对象
    #vis.close()

    for epoch in range(90): #遍历数据集
        print('in epoch ',epoch)
        for i in range(trainSize//batchSize):
            loss = 0
            h = t.zeros(batchSize,hideSize)
            c = t.zeros(batchSize,hideSize)
            ind = np.random.choice(trainSize,batchSize)
            inputsquence = trainInput[ind]
            label = supposedOutputs[ind]
            inputsquence.transpose_(0,1)
            label.transpose_(0,1)
            for j in range(T+20):
                output,h,c = net(inputsquence[j],h,c)
                loss += criterion(output,label[j].squeeze(1))#由损失函数的输入格式决定

                if(math.isnan(loss)):
                    print(inputsquence.squeeze())
                    print(label.squeeze())
                    print(output,h,c)
                    print('lstm infomation: \n',net.lstm.combinedLinear.weight,'\n',net.lstm.combinedLinear.bias)
                    print('Linear infomation: \n',net.fc.weight,'\n',net.fc.bias)
                    quit()

            loss /= (T+20)
            net.zero_grad()
            loss.backward()
            optimizer.step()

            if(i%100==0):
                print('till train ',(trainSize//batchSize)*epoch+i,': loss=',loss)
                x= (trainSize//batchSize) * epoch + i
                #logx = math.log10(x+1)
                #vis.line(X=t.Tensor([logx]),Y=loss.unsqueeze(0),win='loss2',update='append')
                #vis.line(X=t.Tensor([x]), Y=loss.unsqueeze(0), win='loss', update='append')

    correctNumber = 0
    #正确率测试
    for jk in range(testSize):
        #print("result for test",jk)
        h = t.zeros(1,hideSize)
        c = t.zeros(1,hideSize)
        for testTime in range(T+20):
            output,h,c = net(testInputs[jk][testTime].unsqueeze(0),h,c)
            _,max = t.max(output,1)
            if t.equal(max.data,testOutputs[jk][testTime]):
                correctNumber+=1
    print('correctRate: ',correctNumber/(testSize*(T+20)))

    #保存模型
    t.save(net.state_dict(),'save/net0_95.pkl')
    t.save(optimizer.state_dict(),'save/optimizer0_95.pkl')




