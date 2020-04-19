'''
2020-04-16 Aiken Hong 洪振鑫
多传感信息融合作业4： 
量测航迹关联：NN PDA算法实现 ，改变杂波密度
----------------------------------------------------
具体场景：
Space Limited：100 * 100,   杂波数目均值:  3 
Groundtruth：
二维平面运动 yk= y + 0.023t (var=0)；x =x +0.015t(var = 0) 无噪声
F = [[1,0],[0,1]]  f = [[1,0],[0,1]] Q=[[0,0],[0,0]]
初值：0,0 P =[[1,0],[0,1]]
真实量测:
H = [[1,0],[0,1]]  v =独立高斯噪声 0.15 0.25 R=[[0.15,0],[0,0.25]]
实际量测 = 真实量测 + (noise双坐标独立的高斯噪声)
PDA：参数：Pg=Pd=1 通过这样的设定简化模型后进行运算
预报还是基于KF方法，但是在更新和航迹关联的时候采用概率和全概公式进行
--------------------------------------------------------------
TODO:用RMSE(均方根误差，误差和的平均)来评价PDA的误差，Write a report
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

def Plottrack(N,move,cx,cy,S,x,y,isPDA=False):
    ''' 可视化绘图，绘制真实运动，量测，杂波点，滤波'''
    mx = [move[i,0] for i in range(N)]
    my = [move[i,1] for i in range(N)]
    plt.plot(mx,my,color='black',label='real')
    Sx = [S[i,0] for i in range(N)]
    Sy = [S[i,1] for i in range(N)]
    area = np.pi * 2**2  # 点面积
    if isPDA:   
        for i in range(len(x)):
            plt.scatter(x[i],y[i],s=area,c='b')
    else:
        plt.plot(x,y,color='red',label='detect')

    for i in range(N):
        plt.scatter(Sx[i],Sy[i],s=area,c='#DC143C')
    for i in range(len(cx)):
        plt.scatter(cx[i],cy[i],s=area,c='#00CED1')
    plt.legend()
    plt.show()

def SensorGenerate(Move,N,R,shows=False):
    ''' 生成真实的传感器量测'''
    tempx = [Move[i,0] for i in range(N)]
    Noise = NoiseGenertor(R[0,0],N,title='x_noise',shownoise=shows)
    tempx += Noise    
    tempy = [Move[i,0] for i in range(N)]  
    Noise = NoiseGenertor(R[1,1],N,title='y_noise',shownoise=shows)
    tempy += Noise
    Sensor = [[tempx[i],tempy[i]] for i in range(N)]
    Sensor = np.mat(Sensor)
    return Sensor

'''------------------NN Associate Function--------------------------'''
def NNAssociate(N,R,P,show=False):
    '''最近邻的关联方式，基于kalman的预报'''
    Move = Movement(N,0,0,False,False)
    RightNum = 0
    # '''成传感器真实量测👇'''
    print('Start to generate real detect data')
    Sensor = SensorGenerate(Move,N,R,shows=False)
    # print(Sensor[1])
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
        D.append(temp*S.I*temp.T)
        #再测杂波的马氏距离
        for j in range(len(Clutterx[i+1])):
            S = P_predict+ R_uniform
            temp = (np.mat([[Clutterx[i+1][j]],[Cluttery[i+1][j]]])-x_predict).T
            D.append(temp*S.I*temp.T)
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
def PDAssociate(N,R,P,show=False):
    ''' 基于PDA的航迹量测关联方式'''
    move = Movement(N,0,0,False,False)
    # Loss = 0
    #  获取传感器真实量测
    print('Start to generate real detect data')
    Sensor = SensorGenerate(move,N,R,shows=False)
    # 生成基于泊松分布的杂波量测👇
    print('generate disturb wave')
    Clutterx,Cluttery = ClutterGenerate(Lam,N,maxx,maxy)
    # 开始进行基于卡尔曼滤波的PDA量测航迹关联👇
    print('Kalman & PDA process')
    Value, P_mat = [], []
    Value.append(np.mat(move[0]).T) #初始值设置（带点误差）
    P_mat.append(P) #初始的P
    for i in tqdm(range(N-1)):
        # 预报过程：与之前的一致
        x_predict = F*Value[i] + f*U
        P_predict = F*P*F.T + Q
        kalman = P_predict * (P_predict + R).I 
        # PDA Part to comfirm detect & update Predict
        # 先测真实量测的马氏距离-> e ,存储 z~
        E,Z= [], []
        S = P_predict+R
        temp = Sensor[i+1]-x_predict.T
        Z.append(temp.T)
        E.append(np.exp(-0.5*temp*S.I*temp.T))
        # 再测杂波的e👇
        for j in range(len(Clutterx[i+1])):
            S = P_predict+ R_uniform
            temp = (np.mat([[Clutterx[i+1][j]],[Cluttery[i+1][j]]])-x_predict).T
            Z.append(temp.T)
            E.append(np.exp(-0.5*temp*S.I*temp.T))
        assert len(E) == len(Clutterx[i+1])+1, 'the number of e is wrong'
        # 生成概率组：
        tempsum = 0
        for j in range(len(E)):
            tempsum += E[j]
        Pro = [E[i]/tempsum for i in range(len(E))]
        Prob = [Pro[i][0,0] for i in range(len(Pro))]
        # sum2 = sum(np.array(Prob)) #验证概率是否合为一
        # PDA更新x 
        tempk = 0
        for k in range(len(Prob)): 
            tempk += Prob[k]*(x_predict + kalman*Z[k])
        Value.append(tempk)
        # PDA 更新p
        Pkc = (np.eye(2)-kalman*H)*P_predict
        tempp,tempz = 0, 0
        for k in range(len(Prob)): 
            tempp += Prob[k]*Z[k]*Z[k].T
            tempz += Prob[k]*Z[k]
        Pk = kalman*(tempp-tempz*tempz.T)*kalman.T
        P = Pk + Pkc
        P_mat.append(P)
    x = [Value[i][0,0] for i in range(len(Value))]
    y = [Value[i][1,0] for i in range(len(Value))]
    Plottrack(N,move,Clutterx,Cluttery,Sensor,x,y,True)
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
    R = np.mat([[0.15,0],[0,0.25]])
    '''执行参数'''
    Num = 300
    Lam = 3
    maxx = 0.015 *Num
    maxy = 0.023 *Num
    IsNN = True
    # IsNN = False
    # 均匀分布的协方差（两个坐标之间假设独立）
    R_uniformx = np.power(maxx,2)/12
    R_uniformy = np.power(maxy,2)/12
    R_uniform = np.mat([[R_uniformx,0],[0,R_uniformy]]) 
    '''执行主函数'''
    if IsNN:
        NNAssociate(Num,R,P,False)
    else:
        PDAssociate(Num,R,P,False)
    # --------------------------------------------------------------
    t_t = time.time() - t_s
    print(__doc__)
    print('全过程运行时间：{}'.format(t_t))
    # x,y = ClutterGenerate(3,5,100,100)
    # print("x:{},\ny:{}".format(x,y))
    # Movement(100,0,0,False,True)
