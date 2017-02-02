#coding:utf-8
from select import kevent

import numpy as np
import matplotlib.pyplot as plt


def draw(trainingData, w, round, x):
    plt.figure('Round'+str(round))
    drawLine(w)
    drawTrainingData(trainingData)

    if x is not None:
        plt.scatter(x[1], x[2], s= 400, c = 'red', marker=r'$\bigodot$')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def drawLine( w):
    w = w.transpose().tolist()[0]
    x1 = [x1 for x1 in xrange(80)]
    x2 = [ (w[0] + w[1] * i)/(-w[2]) for i in x1]
    plt.plot(x1, x2)



def drawTrainingData(trainingData):
    pointSize = 100
    positive_color = 'red'
    positive_marker = 'o'
    negative_clor = 'blue'
    negative_marker = 'x'

    positive_x1 = []
    positive_x2 = []
    negative_x1 = []
    negative_x2 = []
    for x, y in trainingData:
        x = x.transpose().tolist()[0]

        if y == 1:
            positive_x1.append(x[1])
            positive_x2.append(x[2])
        elif y == -1:
            negative_x1.append(x[1])
            negative_x2.append(x[2])

    plt.scatter(positive_x1, positive_x2, s= pointSize, c = positive_color, marker = positive_marker)
    plt.scatter(negative_x1, negative_x2, s= pointSize, c = negative_clor, marker=negative_marker)



def vector(l):
    return np.mat(l).transpose()



def trainingDataPreprocess(trainingData):
    '''Add x0 dimension & transform to np.mat object'''
    processedTrainingData = [ (vector([1, sample[0], sample[1]]), sample[2])  for sample in trainingData]
    return processedTrainingData



def PLA(trainingData):
    w = np.mat([1,2127,205]).transpose() # Step 1: 向量w赋初值

    k = 0 # 第k轮计数
    while True:
        k += 1

        (status, x, y) = anyMistakePoint(trainingData, w)
        draw(trainingData, w, k, x) # 画图
        if status == 'YES': # Step 2: 切分正确，学习完成
            return w
        else:
            w = w + y*x # Step 3: 修正w



sign = lambda x:1 if x > 0 else -1 if x < 0 else -1
def mSign(m):
    '''判断某个矩阵的[0][0]元素正负.大于0返回1，否则返回-1'''
    x = m.tolist()[0][0]
    return 1 if x > 0 else -1 if x < 0 else -1



def anyMistakePoint(training_data, w):
    '''训练数据中是否有点被切分错误'''
    status = 'YES'
    for (x, y) in training_data:
        if mSign(w.transpose() * x) <> sign(y):
            status = 'NO'
            return (status, x, y)

    return status, None, None


if __name__=="__main__":

    trainingData = [
    [10, 300, -1],
    [15, 377, -1],
    [50, 137, 1],
    [65, 92 , 1],
    [45, 528, -1],
    [61, 542, 1],
    [26, 394, -1],
    [37, 703, -1],
    [39, 244, 1],
    [41, 398, 1],
    [53, 495, 1],
    [32, 119, 1],
    [24, 577, -1],
    [56, 412, 1]
    ]

    processedTrainingData = trainingDataPreprocess(trainingData)
    w = PLA(processedTrainingData)
    
