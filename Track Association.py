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
二维平面运动 yk= y + 0.023t (var=0)；x =x +0.015t(var = 0) 无噪声
F = [[1,0],[0,1]]  f = [[1,0],[0,1]] Q=[[0,0],[0,0]]
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
from multi_sensors_fusion import NoiseGenertor
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
            tempx.append(np.random.uniform(-1,maxx))
            tempy.append(np.random.uniform(-1,maxy))
        Locx.append(tempx)
        Locy.append(tempy)
    # Loc = [[Locx[i],Locy[i]] for i in range(Num)]
    # Locx = np.mat(Locx)
    # Locy = np.mat(Locy)
    return Locx,Locy

def Movement(Num,x0,y0,shownoise=False,showmove=False):
    '''产生匀速运动轨迹'''
    Loc = []
    Loc.append([x0,y0])
    # Gaussx = NoiseGenertor(0.23,Num-1,title='x noise',shownoise= shownoise)
    # Gaussy = NoiseGenertor(0.15,Num-1,title='y noise',shownoise= shownoise)
    print('---------Start to generate groundtruth----------------')
    for i in tqdm(range(Num-1)):
        x = Loc[i][0] + 0.015
        y = Loc[i][1] + 0.023
        Loc.append([x,y])
    
    Loc = np.mat(Loc)
    Loc = Loc.reshape(Num,-1)
    if showmove:
        x = Loc[:,0]
        y = Loc[:,1]
        plt.plot(x,y,color='red')
        # for i in range(Num):
            # plt.scatter(Loc[i][0],Loc[i][1],c='b')
        plt.title('movement')
        plt.show()
    
    return Loc

def Plottrack(N,move,cx,cy,S,x,y):
    mx = [move[i,0] for i in range(N)]
    my = [move[i,1] for i in range(N)]
    plt.plot(mx,my,color='black',label='real')
    plt.plot(x,y,color='red',label='detect')
    Sx = [S[i,0] for i in range(N)]
    Sy = [S[i,1] for i in range(N)]
    area = np.pi * 2**2  # 点面积
    for i in range(N):
        plt.scatter(Sx[i],Sy[i],s=area,c='#DC143C')
    for i in range(len(cx)):
        plt.scatter(cx[i],cy[i],s=area,c='#00CED1')
    plt.legend()
    plt.show()
'''------------------NN Associate Function--------------------------'''
def NNAssociate(N,R,P,show=False):
    '''最近邻的关联方式，基于kalman的预报'''
    Move = Movement(N,0,0,False,False)
    RightNum = 0
    # '''成传感器真实量测👇'''
    print('Start to generate real detect data')
    tempx = [Move[i,0] for i in range(N)]
    Noise = NoiseGenertor(R[0,0],N,title='x_noise',shownoise=False)
    tempx += Noise    
    tempy = [Move[i,0] for i in range(N)]  
    Noise = NoiseGenertor(R[1,1],N,title='y_noise',shownoise=False)
    tempy += Noise
    Sensor = [[tempx[i],tempy[i]] for i in range(N)]
    Sensor = np.mat(Sensor)
    print(Sensor[1])
    # 生成基于泊松分布的杂波量测👇
    print('generate disturb wave')
    Clutterx,Cluttery = ClutterGenerate(Lam,N,maxx,maxy)
    # 开始进行基于卡尔曼滤波的NN关联👇
    print('Kalman & NN process')
    Value = []
    Value.append(np.mat(Move[0]).T) #初始值设置（带点误差）
    P_mat = []
    P_mat.append(P) #初始的P
    for i in tqdm(range(N-1)):
        D = []
        # H =单位阵可以省略
        # 预报过程：(这里有一个变形，预报过程那个不再是误差而实控制参数)
        x_predict = F*Value[i] + f*U
        P_predict = F*P*F.T + Q
        kalman = P_predict * (P_predict + R).I 
        # 在更新前需要确保通过NN确定量测:
        # 先测真实量测的马氏距离
        S = P_predict+R
        temp = Sensor[i+1]-x_predict.T 
        D.append(temp*S*temp.T)
        #再测杂波的马氏距离
        for j in range(len(Clutterx[i+1])):
            S = P_predict+ R_uniform
            temp = (np.mat([[Clutterx[i+1][j]],[Cluttery[i+1][j]]])-x_predict).T
            D.append(temp*S*temp.T)
        D = np.array(D)
        # 找到最大值的下下标，根据情况确定量测，并带入z进行下一波预测
        index_val = np.argmin(D)
        if index_val == 0 :
            RightNum += 1
            z = Sensor[i+1].T
        else :
            z = np.mat([[Clutterx[i+1][index_val-1]],[Cluttery[i+1][index_val-1]]])
        temp = x_predict + kalman*(z - H*x_predict)
        Value.append(temp)
        P = (np.eye(2)-kalman*H)*P_predict
        P_mat.append(P)
    x = [Value[i][0,0] for i in range(len(Value))]
    y = [Value[i][1,0] for i in range(len(Value))]
    # print(x)
    print("选中真实量测作为具体量测的有 {} 次   总有有几次选择 {}".format(RightNum,N-1))
    Plottrack(N,Move,Clutterx,Cluttery,Sensor,x,y)
    return Move,Clutterx,Cluttery,Sensor,x,y
'''------------------PDA Associate Function--------------------------'''
def PDAssociate():
    pass
if __name__ == "__main__":
    t_s = time.time()
    # ---------------------------------------------------------------
    ''' 二维匀速直线运动基本参数:'''
    F = np.mat([[1,0],[0,1]])
    f = np.mat([[1,0],[0,1]])
    U = np.mat([[0.015],[0.023]])
    P = np.mat([[1,0],[0,1]])
    Q = np.mat([[0,0],[0,0]])
    H = np.mat([[1,0],[0,1]])
    R = np.mat([[0.3,0],[0,0.6]])
    '''执行参数'''
    Num = 100
    Lam = 5
    maxx = 0.015 *Num
    maxy = 0.023 *Num
    # 均匀分布的协方差（两个坐标之间假设独立）
    R_uniformx = np.power(maxx,2)/12
    R_uniformy = np.power(maxy,2)/12
    R_uniform = np.mat([[R_uniformx,0],[0,R_uniformy]]) 
    '''执行主函数'''
    NNAssociate(Num,R,P,False)
    # --------------------------------------------------------------
    t_t = time.time() - t_s
    # print(__doc__)
    print('全过程运行时间：{}'.format(t_t))
    # x,y = ClutterGenerate(3,5,100,100)
    # print("x:{},\ny:{}".format(x,y))
    # Movement(100,0,0,False,True)
