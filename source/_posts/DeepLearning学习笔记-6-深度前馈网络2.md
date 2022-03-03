---
title: DeepLearning学习笔记-6.深度前馈网络(二)
date: 2022-02-28 00:14:57
toc: true
mathjax: true
categories:
    - 深度学习

tags:
    - 深度学习笔记
    - 花书
---

&emsp;&emsp;**深度前馈网络（Deep Feedforward Network）**，也叫做**前馈神经网络（Feedforward Neural Network）**或者**多层感知机（Multilayer Perception，MLP）**，典型的深度学习模型。卷积神经网络是一种专门的前馈网络。包含网络层、隐藏层和输出层。
<img src="/images/feedforward.png" width="400px"></img>
<!--more-->

### 感知机

&emsp;&emsp;感知机（Perception）由Rosenblatt在1957年提出，是**神经网络和支持向量机**的基础。由n个输入数据，通过权重与各数据之前的计算和，比较激活函数结果，得出输出，可解决与、或、非问题，一般用于分类问题。有输入输出、权重和偏置、NetSum、激活函数四部分组成。

<img src="/images/fnn_perception.png" width="500px"></img>

&emsp;&emsp;工作步骤：
* 输入$x_{i}$并乘以对于的权重$w_{i}$，得到$k_{i}$。
* 将所有相乘的值$k_{i}$相加，得到加权和。
* 将加权和应用于正确的激活函数。
* 数据应用，如将数据分为两部分（线性二院分类器）

#### 手动推导

<img src="/images/fnn_perception1.jpg" width="600px"></img>
<img src="/images/fnn_perception2.jpg" width="600px"></img>


### XOR 

&emsp;&emsp;**异或问题，相同为0，不同为1。** 如上图所示，无法用一条直线将异或问题来分类。为解决该问题，可再添加一条直线，用两条直线分割，也就意味着需要再添加一个感知机。

&emsp;&emsp;一个好玩的神经网络演示[playground.tensorflow.org][2]，如下图的分类类似于异或问题，当用一层神经网络时，无法分类。当增加一层，用两层神经网络便可对所有的点进行分类。

<img src="/images/fnn_xor1.png" width="800px"></img>
<img src="/images/fnn_xor2.png" width="800px"></img>

&emsp;&emsp;解决网络如图：
<img src="/images/fnn_xor.png" width="300px"></img>
&emsp;&emsp;整个网络为：
<center>$f(x;W,c,w,b)=w^{\top }max \{ 0,W^{\top}x+c \}+b$</center>

&emsp;&emsp;激活函数使用ReLU：
$$ReLU(x)=\begin{cases}
 x & \\text{ if } x> 0 \\\\
 0 & \\text{ if } x\le 0
\end{cases}$$

<img src="/images/activation_leru.png" width="400px"></img>

#### 手动推导
<img src="/images/fnn_xor3.jpg" width="600px"></img>


### 基于梯度的学习

#### 代价函数

#### 输出单元

### 隐藏单元

#### 整流线性单元及其扩展

#### logistic sigmoid与双曲正切函数

#### 其他隐藏单元

### 架构设计

### 反向传播和其他微分算法

#### 计算图

#### 微积分中的链式法则

#### 递归地使用链式法则来实现反向传播

#### 全连接MLP中的反向传播计算

#### 符号到符号的导数

#### 一般化的反向传播

#### 用于MLP训练的反向传播

#### 复杂化




[1]:https://www.jiqizhixin.com/graph/technologies/f9849d6c-6262-4c1f-8f42-6d976be17161
[2]:https://playground.tensorflow.org
