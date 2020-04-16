'''FUNCTION DOC
2020-04-03 Aiken Hong 洪振鑫
多传感信息融合作业3： 
集中式，分布式估计融合系统
TODO：Define：状态演化方程(线性) + num_sensor & 各自的测量方程和噪声
TODO：Write a report  
实验内容，实验原理，实验场景，实验结果展示和分析
'''

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse 
from scipy.stats import norm
import math 
from tqdm import tqdm 
import time 

'''------------------Utility Function--------------------------'''
'''
TODO：输出tracking的结果和origin track
协方差椭圆，可能需要叠加协方差椭圆，看着办吧
'''
def LossEllipse(Mean, Cov, Cof):
    '''误差椭圆绘制,一般根据sigma，2sigma，3sigma，和95%置信度，图片保存到当前文件夹
    
    Parameter:Mean:均值，Cov:协方差矩阵， Cof：置信度'''
    # 计算协方差阵的特征值和特征向量
    vals,vecs = np.linalg.eigh(Cov)
    print('vals:{},\n vecs:{}'.format(vals,vecs))
    # 计算最大特征值对应的特征向量来计算矩阵的椭圆的偏移角
    k = np.argmax(vals)
    x,y = vecs[k,:]
    theta = np.degrees(np.arctan2(y,x))
    # 求解半长轴半短轴，并绘图
    h, w = np.sqrt(5.991 * vals)
    ell = Ellipse(mean, 2*w, 2*h, theta, facecolor='yellow') 
    ell.set_alpha(0.5)
    # ell.set_fill(False)
    ax = plt.subplot(111,aspect='equal')
    ax.add_patch(ell)
    ax.plot(0,0,'ro') 
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    # 将误差椭圆保存在当前文件夹中
    plt.savefig('lossEllipse.jpg')

def PlotTrack():
    '''绘制估计的轨迹和根据状态演化方程产生的轨迹'''
    # 得到点集和预测值后直接plt。plot就行了。
    # 问题在于叠加两个轨迹 that‘s all

    pass

def NoiseGenertor():
    '''噪声生成，用于各个需要噪声的场景量身定制'''
    # 可能不需要单独的函数把
    pass

'''-------------Distributed Fusion Function--------------------'''
'''
TODO：分布式融合： 
在每个sensor上使用（KF：PPT4-1：32），得到一个滤波估计，将滤波估计(Track跟踪轨迹)，
（和方差啥？估计误差）的传送到Center
Center做 Track 的融合: PPT4-2：107
'''

def DistributedFusion():
    '''分布式融合入口函数'''
    pass


'''---------------Central Fusion Function----------------------'''
''' 
TODO：集中式融合：
将每个sensor的观测，观测矩阵，噪声，传递到Center，
然后使用基于信息滤波器的方法做最终的滤波(Track跟踪) Slide4-2：96
'''
def CentralFusion():
    '''集中式融合入口函数'''
    pass


'''-------------------Intro-----------------------'''
if __name__ == "__main__":
    mean = [0,0]
    cov = [[1,0.6],[0.6,2]]
    LossEllipse(mean,cov,2)
    print(__doc__)



    # fw = 10
    # x = np.linspace(-100,100,1000)
    # mean0 = 0.0
    # var0 = 20.0
    # plt.figure(figsize=(fw,5))
    # plt.plot(x,norm.pdf(x, mean0, var0), label='Normal Distribution')
    # plt.ylim(0, 0.1);
    # plt.legend(loc='best');
    # plt.xlabel('Position');
    # plt.show()