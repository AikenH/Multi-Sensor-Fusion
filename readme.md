# Multi-Sensor Fusion
## Intro:

仅仅为多传感信息融合的课程作业<br>
Coursework of multi-sensor information fusion<br>
见仁见智，慎用代码 <br>
Most likely not the code you need<br>


## Contents：

### 1. Centralized & Distributed Fusion based on linear states transfer equation. <br>
    线性状态转移方程下的集中式和分布式多传感信息融合
    include:  
    - Kalman Fliter 卡尔曼滤波<br>
    - SCC-Simple Convex Combination 简单凸组合融合算法<br>
    - GaussNoise 高斯噪声<br>
    - Loss Ellipse 误差椭圆<br>
    - Sequential fusion 集中式序贯融合<br>

### 具体实现场景：
Define：
给定双初值:x(-1)=x(0)=0 <br>
产生随机运动序列 xn = 1.74x(n-1) - 0.81x(n-2) +v0(n) <br>
Sensor1的观测方程：y(n) = x(n) + v1(n) <br>
Sensor2的观测方程：y(n) = x(n) + v2(n) <br>
0均值白高斯：v0：方差0.04 V1：方差4.5 V2：方差 9 <br>

### 2. Track Association<br>
    航机关联，在不同杂波密度的情况下的误差分析和比较<br>
    include：
    - NN(Basic on kalman )(直接将选点连线)
    - Clutter (poisson)
    - PDA(用选点来可视化细节，比自动拟合的曲线要好)

### 具体实现场景：
Space Limited：100 * 100,   杂波数目均值:  3 <br>
Groundtruth：<br>
二维平面运动 yk= y + 0.023t (var=0)；x =x +0.015t(var = 0) 无噪声<br>
F = [[1,0],[0,1]]  f = [[1,0],[0,1]] Q=[[0,0],[0,0]]<br>
初值：0,0 P =[[1,0],[0,1]]<br>
真实量测:<br>
H = [[1,0],[0,1]]  v =独立高斯噪声 0.15 0.25 R=[[0.15,0],[0,0.25]]<br>
实际量测 = 真实量测 + (noise双坐标独立的高斯噪声)<br>
PDA：参数：Pg=Pd=1 通过这样的设定简化模型后进行运算<br>
预报还是基于KF方法，但是在更新和航迹关联的时候采用概率和全概公式进行<br>


## Something else:

### <center>水平有限,请多多指教</center>
<center>Ability is Limited；Appreciate Comments</center>

<p align='right'>AIKEN-H<br></p> 
<p align='right'>2020-04-14(origin)</p>