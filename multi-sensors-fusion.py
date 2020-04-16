'''FUNCTION DOC
2020-04-03 Aiken Hong 洪振鑫
多传感信息融合作业3： 
集中式，分布式估计融合系统
----------------------------------------------------
Define：
给定双初值:x(-1)=x(0)=0
产生随机运动序列 xn = 1.74x(n-1) - 0.81x(n-2) +v0(n)
Sensor1的观测方程：y(n) = x(n) + v1(n)
Sensor2的观测方程：y(n) = x(n) + v2(n)
0均值白高斯：v0：方差0.04 V1：方差4.5 V2：方差 0.5
----------------------------------------------------
TODO：Write a report  
实验内容，实验原理，实验场景，实验结果展示和分析
'''

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse 
from scipy.stats import norm
from tqdm import tqdm 
import time 

'''------------------Utility Function--------------------------'''

# 协方差椭圆，可能需要叠加协方差椭圆，看着办吧,换成return ellipse，然后集成print
def LossEllipse(Mean, Cov, color='yellow', Cof=None):
    '''误差椭圆绘制,一般根据sigma，2sigma，3sigma，和95%置信度，图片保存到当前文件夹
    
    Parameter:Mean:均值，Cov:协方差矩阵， Cof：置信度'''
    # 计算协方差阵的特征值和特征向量
    vals,vecs = np.linalg.eigh(Cov)
    print('vals:{},\n vecs:{}'.format(vals,vecs))
    # 计算最大特征值对应的特征向量来计算矩阵的椭圆的偏移角
    k = np.argmax(vals)
    vecs = np.array(vecs)
    x,y = vecs[k,:]
    theta = np.degrees(np.arctan2(y,x))
    # 求解半长轴半短轴，并绘图
    h, w = np.sqrt(5.991 * vals)
    ell = Ellipse(Mean, 2*w, 2*h, theta, facecolor= color) 
    ell.set_alpha(0.5)
    # ell.set_fill(False)
    ax = plt.subplot(111,aspect='equal')
    ax.add_patch(ell)
    ax.plot(0,0,'ro') 
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    # 将误差椭圆保存在当前文件夹中
    # plt.savefig('lossEllipse.jpg')
    plt.show()

def PlotTrack(Num,x_real,Pred,Loss, title=None):
    '''绘制估计的轨迹和根据状态演化方程产生的轨迹'''
    n = np.linspace(1,Num+1,Num+1)
    plt.plot(n,Pred,color='red',label='pred')
    plt.plot(n,x_real,color='blue',label='read')
    plt.title(title)
    plt.legend()
    plt.show()

    LossEllipse([0,0], Loss)

def NoiseGenertor(Gsigma,Num,title=None,Gmean=0,shownoise=False):
    '''高斯白噪声生成，可选图像显示，'''
    x = np.linspace(0,1,Num)
    GaussNoise = norm.pdf(x,Gmean,Gsigma)
    if shownoise==True:
        plt.plot(x,GaussNoise)
        plt.title(title)
        plt.show()
    return GaussNoise

'''-------------Distributed Fusion Function--------------------'''
'''
TODO：分布式融合： 
在每个sensor上使用（KF：PPT4-1：32），得到一个滤波估计，将滤波估计(Track跟踪轨迹)，
（和方差啥？估计误差）的传送到Center
Center做 Track 的融合: PPT4-2：107
'''
def DistributedFusion(Num=50,show=False):
    '''分布式融合入口函数'''
    x_real = movement(Num+1,0,0)
    Sensor1, S1Loss = KalmanFliter(Num,x_real,R1,P,showfig=True)
    Sensor2, S2Loss = KalmanFliter(Num,x_real,R2,P,showfig=True)
    Loss = []
    Pred = []
    for i in tqdm(range(Num)):
        # part1 :P
        TempValue = np.linalg.inv(S1Loss[i]) + np.linalg.inv(S2Loss[i])
        Loss.append(np.linalg.inv(TempValue))
        # part2: x
        tempx1 = np.array([[Sensor1[i]],[Sensor1[i+1]]])
        tempx2 = np.array([[Sensor2[i]],[Sensor2[i+1]]])
        pp1 = np.dot(Loss[i],np.linalg.inv(S1Loss[i]))
        pp2 = np.dot(Loss[i],np.linalg.inv(S2Loss[i]))
        Ans = np.dot(pp1,tempx1) + np.dot(pp2,tempx2)
        Pred.append(Ans[1])
    Pred = np.array(Pred)
    Pred = Pred.reshape(Num)
    if show:
        cov = Loss[i]
        PlotTrack(Num-1,x_real[0:Num],Pred,cov,'final')


def KalmanFliter(Num,x_real,R,P,showfig=False):
    '''各个传感器的卡尔曼滤波情况

    Vn:第n个传感器的白高斯方差，用以产生观测;
    Num:追踪长度'''
    Noise = NoiseGenertor(R,Num+1)
    SensorGet = x_real + Noise
    # 👆生成传感器读取数据
    Value = []
    Value.append(np.mat([[x_real[0],],[x_real[0],]]))
    Loss = []
    Loss.append(P)
    # 存储初值和后续滤波结果
    for i in tqdm(range(Num)):
        x_predict = F*Value[i] 
        P_predict = F*P*F.T + Q0
        kalman = P_predict* H.T/(H*P_predict*H.T+R)
        z = SensorGet[i+1]
        temp = x_predict + kalman*(z - H*x_predict)
        Value.append(temp)
        P = (np.eye(2)-kalman*H)*P_predict
        Loss.append(P)
    # 重构Value
    value = [Value[i][1].tolist() for i in range(len(Value))]
    value = np.array(value)
    value = value.reshape([Num+1])
    
    if showfig:
        cov=Loss[i]
        PlotTrack(Num,x_real,value,cov,'Kalman Flither {}'.format(R))

    return value,Loss

'''---------------Central Fusion Function----------------------'''
''' 
TODO：集中式融合：
将每个sensor的观测，观测矩阵，噪声，传递到Center，
然后使用基于信息滤波器的方法做最终的滤波(Track跟踪) Slide4-2：96
'''
def CentralFusion(Num):
    '''集中式融合入口函数'''
    pass


'''-------------------Intro-----------------------'''
def movement(Num,x00,x01):
    '''得到状态转移序列（实际值）'''
    list1 = []
    list1.append(x00)
    list1.append(x01)
    Noise = NoiseGenertor(0.04,Num)
    for i in range(Num-2):
        Value = 1.74*list1[i+1] - 0.81*list1[i]
        list1.append(Value+Noise[i])

    assert len(Noise)==len(list1),'Wrong len for noise or list'
    return list1

if __name__ == "__main__":
    # 基本运动模型参数
    F = np.mat([[0,1],[-0.81,1.74]])
    P = np.mat([[1,0],[0,1]])
    Q0 = np.mat([[0,0],[0,0.04]])
    H = np.mat([0,1])
    R1 = 4.5
    R2 = 0.5
    # 执行参数
    mode = 1  #控制集中式或者分布式
    num = 100  #控制跟踪多远
    # 入口函数
    if mode == 1 :
        DistributedFusion(num,True)
    else:
        CentralFusion(num)
    
    # print(movement(20,0,0))
    # NoiseGenertor(6,100,'NOise2',shownoise=True)
    # mean = [0,0]
    # # cov = [[1,0.6],[0.6,2]]
    
    # cov = np.mat([[0.62961845, 0.64065115],
    #     [0.64065115, 0.73873982]])
    # LossEllipse(mean,cov)