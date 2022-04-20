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

&emsp;&emsp;**深度前馈网络（Deep Feedforward Network，DFN）**，也叫做**前馈神经网络（Feedforward Neural Network，FNN）**或者**多层感知机（Multilayer Perception，MLP）**，典型的深度学习模型。目标是拟合一个函数，如有一个分类器$y=f^{*}(x)$将输入$x$映射到输出类别$y$。深度前馈网将这个映射定义为$f(x,\theta)$，并学习这个参数$\theta$的值来得到最好的函数拟合。

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

### 深度前馈网络

&emsp;&emsp;深度前馈网络中信息从$x$流入，通过中间$f$的计算，最后到达输出$y$。深度前馈网络示意图如下：

<img src="/images/fnn_3.png" width="300px"></img>

&emsp;&emsp;函数$f^{(1)},f^{(2)},f^{(3)}$链式连接，可表示为$f(x)=f^{(3)}(f^{(2)}(f^{(1)}(x)))$，这种链式结构是神经网络最为常用结构。$f^{(1)},f^{(2)}$被称为神经网络的第一层，第二层，也为网络的隐藏层（Hidden Layer），最后一层$f^{(3)}$为输出层（Output Layer）。链的长度为神经网络的深度，输入向量的每个元素均视作一个神经元。


### 基于梯度的学习

&emsp;&emsp;在[[DeepLearning学习笔记-4-数值计算][3]] 一文中已经介绍了梯度下降优化方法，训练算法几乎总是基于使用梯度来使得代价函数下降的各种方法。（基于梯度下降思想的改进和提纯）

#### 代价函数

&emsp;&emsp;任何能够衡量模型预测值与真实值之间的差异的函数都可以叫做代价函数。当输出神经元的激活函数是线性时(如ReLU函数)，二次代价函数是一种合适的选择；当输出神经元的激活函数是S型函数时(如sigmoid、tanh函数)，选择交叉熵代价函数则比较合理。

#### 输出单元

&emsp;&emsp;常用的线性、sigmoid、softmax输出单元为最常见输出单元。

##### 高斯输出分布的线性单元

##### 伯努利输出分布的sigmoid单元

##### 范畴输出分布的softmax单元

### 隐藏单元

&emsp;&emsp;层与层之间是全连接的，第$i$层的任意一个神经元一定与第$i+1$层的任意一个神经元相连。如图所示，大多数隐藏单元都可以描述为接受输入向量$x$，计算仿射变换$z=W^{\top}x+b$，然后使用一个逐元素的非线性函数$g(z)$得到隐藏单元的输出$\alpha$。而大多数隐藏单元的区别仅仅在于激活函数$g(z)$的形式。

<img src="/images/fnn_4.png" width="300px"></img>

&emsp;&emsp;如图所示，假设激活函数$g(z)$为$\sigma$，于是$f^{(1)}$层的隐藏单元可描述为：

$$\\left\\{\begin{matrix}
a_{1}=\sigma (z_{1})=\sigma (w_{11}x_{1}+w_{12}x_{2}+w_{13}x_{3}+b_{1}) \\\\
a_{2}=\sigma (z_{2})=\sigma (w_{21}x_{1}+w_{22}x_{2}+w_{23}x_{3}+b_{2}) \\\\
a_{3}=\sigma (z_{3})=\sigma (w_{31}x_{1}+w_{32}x_{2}+w_{33}x_{3}+b_{3}) \\\\
\\end{matrix}\\right.$$

&emsp;&emsp;选择隐藏单元实际上就是要选择一个合适的激活函数。常见激活函数：

* 整流线性单元（ReLU）：$g(z)=max{0, z}$。优点是易于优化，二阶导数几乎处处为0，**处于激活状态时一阶导数处处为1**，相比于引入二阶效应的激活函数，梯度方向对学习更有用。如果使用ReLU，第一步做线性变换$W^{\top}x+b$时的$b$一般设置成小的正值。缺陷时不能通过基于梯度的方法学习那些单元激活为0的样本。ReLU函数的梯度为

$$g^{\prime}(z)=\begin{cases}
 1 & x> 0 \\\\
 0 & x\le 0
\end{cases}$$

* sigmoid函数或双曲正切函数$tanh$。两者之间有一个联系:$tanh(z)=2\sigma (2z)-1$。两者都比较容易饱和，仅当$z$接近0时才对输入强烈敏感，因此使得基于梯度的学习变得非常困难，不适合做前馈网络中的隐藏单元。**如果必须要使用两种中的一个，那么tanh通常表现更好**，因为在0附近其类似于单位函数。即，如果网络的激活能一直很小，训练$\hat{y}=w^{\top }tanh(U^{\top }tanh(V^{\top }x))$类似于训练一个线性模型$\hat{y}=w^{\top }U^{\top }V^{\top }x$。RNN和一些自编码器有一些额外的要求，因此不能使用分段激活函数，此时这种类sigmoid单元更合适。
sigmoid函数写作$g(z)=\sigma (z)=\frac{1}{1+e^{-z}}$，梯度为$g^{\prime}(z)=\sigma (z)(1-\sigma (z))$
双曲正切函数写作$g(z)=tanh(z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}} $，梯度为$g^{\prime }(z)=1-tanh^{2}(z)$

* softplus函数
* 径向基函数
* 硬双曲正切函数

### 架构设计

&emsp;&emsp;架构指网络的整体结构，具有多少单元，以及这些单元应该如何连接。在实践中，神经网络具有多样性，卷积神经网络、循环神经网络会在后边章节学习、分析、总结。

### 反向传播和其他微分算法

&emsp;&emsp;**前向传播（Forward Propagation）**：前馈神经网络接受输入$x$并产生输出$\hat{y}$时，信息通过网络向前流动，输入$x$提供初始信息，然后传播到每一层的隐藏单元，最终产生输出$\hat{y}$。**反向传播（Back Propagation）**：允许来自代价函数的信息通过网络向后流动，以便计算梯度，是指计算梯度的方法。

#### 计算图

&emsp;&emsp;主要用图语言来描述神经网络。

#### 微积分中的链式法则

&emsp;&emsp;微积分中的链式法则用于计算复合函数的导数。反向传播是一种计算链式法则的算法，使用高效的特定运算顺序。设$x$是实数，$f$和$g$是从实数映射到实数的函数。假设$y=g(x)$，并且$z=f(g(x))=f(y)$。链式法则为：

<center>$\frac{dz}{dx}=\frac{dz}{dy}\frac{dy}{dx}$</center>

&emsp;&emsp;将这种标量进行扩展，假设$x\in \mathbb{R}^{m},y\in \mathbb{R}^{n}$，$g$是从$\mathbb{R}^{m}$到$\mathbb{R}^{n}$的映射，$f$是从$\mathbb{R}^{n}$到$\mathbb{R}$的映射。如果$y=g(x)$，并且$z=f(y)$，那么：

<center>$\frac{\partial z}{\partial x_{i}} =\sum_{j}\frac{\partial z}{\partial y_{j}} \frac{\partial y_{j}}{\partial x_{i}}$</center>

&emsp;&emsp;使用向量法，可以等价的写成

<center>$\nabla_{x}z=\left ( \frac{\partial y}{\partial x}  \right ) ^{\top }\nabla_{y}z$</center>

&emsp;&emsp;$\frac{\partial y}{\partial x}$是$g$的$n\times m$的Jacobian矩阵。

#### 递归地使用链式法则来实现反向传播

&emsp;&emsp;许多子表达式可能在梯度的整个表达式中重复若干次。

#### 全连接MLP中的反向传播计算

#### 符号到符号的导数

#### 一般化的反向传播

#### 用于MLP训练的反向传播

#### 复杂化




[1]:https://www.jiqizhixin.com/graph/technologies/f9849d6c-6262-4c1f-8f42-6d976be17161
[2]:https://playground.tensorflow.org
[3]:https://deeplearner.top/2022/01/04/DeepLearning%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0-4-%E6%95%B0%E5%80%BC%E8%AE%A1%E7%AE%97/
