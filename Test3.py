'''          FUNCTION DOC
2020-04-03 Aiken Hong 洪振鑫
多传感信息融合作业3： 
集中式，分布式估计融合系统
TODO：Define：状态演化方程(线性) + num_sensor & 各自的测量方程和噪声
TODO：Write a report  
实验内容，实验原理，实验场景，实验结果展示和分析
'''

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm
import math 
from tqdm import tqdm 
import time 

'''------------------Utility Function--------------------------'''
'''
TODO：输出tracking的结果和origin track以及协方差椭圆来展示效果
'''
def LossEllipse():
    '''误差椭圆绘制'''
    pass

def PlotTrack():
    '''绘制估计的轨迹和根据状态演化方程产生的轨迹'''
    
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

    print(__doc__)

