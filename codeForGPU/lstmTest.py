import math
import torch as t
import torch.nn as nn
from torch.autograd import Function


scaleX = 0.7

class Sigmoid(Function):
    #缩放系数初步设置为 scaleX
    @staticmethod
    def forward(ctx,x):
        output = 1/ (1 + t.exp(-x))
        ctx.save_for_backward(output)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        gradx = scaleX * output * (1-output) * grad_output
        return gradx

class Tanh(Function):
    @staticmethod
    def forward(ctx, x):
        paraP = t.exp(x)
        paraN = t.exp(-x)
        output = (paraP-paraN)/(paraP+paraN)
        ctx.save_for_backward(output)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        gradx = scaleX * (1-output*output) * grad_output
        return gradx

class LstmCell(nn.Module):
    def __init__(self,inSize,hideSize):
        super(LstmCell,self).__init__()
        self.inputSize = inSize
        self.hiddenSize = hideSize

        self.combinedLinear = nn.Linear(inSize+hideSize,4*hideSize)
        self.combinedLinear = self.combinedLinear

        '''
        stdv = 1/math.sqrt(hideSize)
        for para in self.parameters():
            para.data.uniform_(-stdv,stdv)
        '''

    def forward(self,x,h,c):
        combinedInput = t.cat((x,h),1)
        combinedOutput = self.combinedLinear(combinedInput)

        forgetPart = combinedOutput[:,0:self.hiddenSize]
        inputPart = combinedOutput[:,self.hiddenSize:2*self.hiddenSize]
        cellPart = combinedOutput[:,2*self.hiddenSize:3*self.hiddenSize]
        outputPart = combinedOutput[:,3*self.hiddenSize:4*self.hiddenSize]

        newCellState = Sigmoid.apply(forgetPart) * c + Sigmoid.apply(inputPart) * Tanh.apply(cellPart)
        newHiddenState = Sigmoid.apply(outputPart) * Tanh.apply(newCellState)

        return newHiddenState,newCellState