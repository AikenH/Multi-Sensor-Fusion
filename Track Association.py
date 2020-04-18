'''
2020-04-16 Aiken Hong 洪振鑫
多传感信息融合作业4： 
量测航迹关联：NN PDA算法实现 ，改变杂波密度
----------------------------------------------------
具体场景：
2.设计传感器，获取（针对于目标的测量数据/杂波测量数据）
detail：固定运动区域，在区域内设计杂波密度（数量）
----------------------------------------------------
Space Limited：100 * 100,   杂波数目均值:  3 
Groundtruth：
二维平面运动 yk= y + noise (var=0.15)；x =x +noise(var = 0.23) 噪声之间相互独立
F = [[1,0],[0,1]]  f = [[1,0],[0,1]] Q=[[0.15,0],[0,0.23]]
初值：0,0 P =[[1,0],[0,1]]
真实量测:
H = [[1,0],[0,1]]  v =独立高斯噪声 0.3 0.6 R=[[0.3,0],[0,0.6]]
实际量测 = 真实量测 + (noise双坐标独立的高斯噪声)
TODO:基于卡尔曼滤波的预报来进行航迹关联
TODO:NN 用关联正确率来评价， 
TODO:PDA 用RMSE来评价
'''

import numpy as np 
import time
from tqdm import tqdm
import matplotlib.pyplot as plt 
from multi_sensors_fusion import PlotTrack, NoiseGenertor
'''------------------Utility Function--------------------------'''
def ClutterGenerate(Lam,Num,maxx,maxy):
    '''生成杂波干扰：Number(possion) & Localization(uniform)
    
    Para: Lam:平均出现杂波的次数，Num：需要生成多少个时刻，maxxy：杂波空间边界约束'''
    # 首先生成每个时刻的杂波数目：
    numSeq = np.random.poisson(Lam,Num)
    # 对每个时刻的杂波均匀localization
    Locx, Locy = [], []
    for i in tqdm(range(Num)):
        tempx, tempy = [], [] 
        for j in range(numSeq[i]):
            tempx.append(np.random.randint(0,maxx))
            tempy.append(np.random.randint(0,maxy))
        Locx.append(tempx)
        Locy.append(tempy)
    return Locx, Locy

def Movement(Num,x0,y0,shownoise=False):
    Loc = []
    Loc.append([x0,y0])
    Gaussx = NoiseGenertor(0.23,Num-1,title='x noise',shownoise= shownoise)
    Gaussy = NoiseGenertor(0.15,Num-1,title='y noise',shownoise= shownoise)
    print('---------Start to generate groundtruth----------------')
    for i in tqdm(range(Num-1)):
        x = Loc[i][0] + Gaussx[i]
        y = Loc[i][1] + Gaussy[i]
        Loc.append([x,y])
    print(Loc)
    return Loc
'''------------------NN Associate Function--------------------------'''
def NNAssociate():
    '''最近邻的关联方式'''
    pass
'''------------------PDA Associate Function--------------------------'''
def PDAssociate():
    pass

if __name__ == "__main__":
    # t_s = time.time()
    # x,y = ClutterGenerate(3,5,100,100)
    # print('x:{} \n y:{}'.format(x,y))
    Movement(3,0,0,True)
    # t_t = time.time() - t_s
    # print(__doc__)
    # print('全过程运行时间：{}'.format(t_t))
