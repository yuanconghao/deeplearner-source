---
title: DeepLearning学习笔记-5.机器学习基础
date: 2022-01-10 00:27:23
toc: true
mathjax: true
categories:
    - 深度学习

tags:
    - 深度学习笔记
    - 花书
---

深度学习是机器学习的一个特定分支。

<!--more-->

### 学习算法

&emsp;&emsp;机器学习算法描述一种能够**从数据中学习的算法**。**学习**指对于某类**任务T**和**性能度量P**，一个计算机程序被认为可以从**经验E**中学习，通过经验E改进后，它在任务T上由性能度量P衡量的性能有所提升。

#### 任务T

&emsp;&emsp;机器学习任务定义为机器学习系统应该如何处理样本（Example）。样本是指从某些机器学习系统处理的对象或事件中收集到的已经量化的特征（Feature）的集合，用向量$x \in R^{n}$表示，其中向量的每个元素$x_{i}$是一个特征。常见的机器学习任务如分类、输入缺失分类、回归、转录、机器翻译、结构化输出、异常检测、合成和采样、缺失值填补、去噪、密度估计或概率质量函数估计。

#### 性能度量P

&emsp;&emsp;为了评估机器学习算法的优劣，需要对算法的输出结果进行定量的衡量分析，需要合适的性能度量指标：

&emsp;&emsp;针对分类任务：

* 准确率：
$Accuracy=\frac{TP+TN}{TP+TN+FP+FN}$
* 错误率：
$Errorrate=1-Accuracy$
* 精确率：
$Precision=\frac{TP}{TP+FP}$
* 召回率：
$Recall=\frac{TP}{TP+FN}$
* F1值：
$F1=\frac{2\cdot Precision\cdot Recall}{Precision+Recall}$

| 指标 | 含义 |
| --- | --- |
| TP | True Positive，正样本预测为正例数目 |
| TN | True Negative，正样本预测为负例数目 |
| FP | False Positive，负样本预测为正例数目 |
| FN | False Negative，负样本预测为负例数目 |

&emsp;&emsp;针对回归任务：距离误差

#### 经验E

&emsp;&emsp;根据经验E的不同，机器学习算法可以分为：无监督学习算法（Unsupervised Learning）和监督学习算法（Supervised Learning）。
* 无监督学习：训练含有很多**样本特征**的数据集，算法需要从中学习出特征中隐藏的结构性质。例如：密度估计、聚类。
* 监督学习：训练含有很多特征（样本特征和标签值）的数据集。例如：分类、回归。

#### 示例：线性回归

&emsp;&emsp;**线性回归（Linear Regression）**的目标：建立一个系统，将向量$x \in R^{n}$作为输入，预测标量$y \in R$作为输出。线性回归的输出是其输入的线性函数。令$\hat{y}$表示模型预测$y$应该取得值：
<center>$\hat{y}=w^{\top}x$</center>
&emsp;&emsp;其中$w \in R^{n}$是参数向量。

&emsp;&emsp;性能度量P的定义：假设测试集的特征和标签分别用$X^{(test)}$和$y^{(test)}$表示。
&emsp;&emsp;性能度量的方式：均方误差（Mean Squared Error），如果$\hat{y}^{(test)}$表示模型在测试集上的预测值，那么均方误差公式：

<center>$MSE_{test}=\frac{1}{m}\sum_{i}(\hat{y}^{(test)}-y^{(test)})^{2}_{i}=\frac{1}{m}\left\|\hat{y}^{(test)}-y^{(test)} \right\|^{2}_{2}$</center>

&emsp;&emsp;当预测值和目标值之间的欧氏距离增加时，误差也会增加。为了构建一个机器学习算法，设计的算法通过观察训练集获取经验，减少$MSE_{test}$来改进权重$w$。一种直观方式是最小化训练集上的均方误差，即$MSE_{train}$。最小化$MSE_{train}$可以对导数为0进行求解：

<center>$\triangledown _{w}MSE_{train}=0$</center>
<center>$\Rightarrow \triangledown _{w}\frac{1}{m}\left\|\hat{y}^{(train)}-y^{(train)}\right\|^{2}_{2}=0 $</center>
<center>$\Rightarrow \frac{1}{m} \triangledown _{w}\left\|X^{(train)}w-y^{(train)}\right\|^{2}_{2}=0$</center>
<center>$\Rightarrow \triangledown _{w}(X^{(train)}w-y^{(train)})^{\top}(X^{(train)}w-y^{(train)})=0$</center>
<center>$\Rightarrow \triangledown _{w}(w^{\top}X^{(train)\top}X^{(train)}w-2w^{\top}X^{(train)\top}y^{(train)}+y^{(train)\top}y^{(train)})=0$</center>
<center>$\Rightarrow 2X^{(train)\top}X^{(train)}w-2X^{(train)\top}y^{(train)}=0$</center>
<center>$\Rightarrow w=(X^{(train)\top}X^{(train)})^{-1}X^{(train)\top}y^{(train)}$</center>


```python
import numpy as np
import math
import matplotlib.pyplot as plt

# y = wx + b
X = np.hstack((np.array([[-0.5,-0.45,-0.35,-0.35,-0.1,0,0.2,0.25,0.3,0.5]]).reshape(-1, 1), np.ones((10,1))*1))
y = np.array([-0.2,0.1,-1.25,-1.2,0,0.5,-0.1,0.2,0.5,1.2]).reshape(-1,1)

# 最小化MSE,w详见公式
w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print("Weight:", list(w))

hat_y = X.dot(w)
x = np.linspace(-1, 1, 50)
hat_y = x * w[0] + w[1]
plt.figure(figsize=(4,4))
plt.xlim(-1.0, 1.0)
plt.xticks(np.linspace(-1.0, 1.0, 5))
plt.ylim(-3, 3)
plt.plot(x, hat_y, color='red')
plt.scatter(X[:,0], y[:,0], color='black')
plt.xlabel('$x_1$')
plt.ylabel('$y$')
plt.title('$Linear Regression$')
plt.show()

# output
Weight: [array([1.49333333]), array([0.04966667])]
```
<img src="/images/linearregression.png"></img>


### 容量、过拟合和欠拟合

&emsp;&emsp;**泛化（Generalization）** ：不只在训练集上表现良好，能够在先前未观测到的新输入上表现良好的能力。
&emsp;&emsp;**训练误差（Training Error）** ：量化模型在训练集上的表现。
&emsp;&emsp;**测试误差（Test Error）** ：量化模型在测试集上的表现，或称为**泛化误差（Generalization Error）** 。理想的模型是在最小训练误差的同时，最小化泛化误差。

&emsp;&emsp;在实际应用过程中，会采样两个数据集，减小训练误差得到参数后，再在测试集中验证。这个过程中，会发生测试误差的期望大于训练误差的期望的情况。决定机器学习算法是否好的因素：

* 降低训练误差。
* 缩小训练误差与测试误差之间的差距。

&emsp;&emsp;两个因素分别对应机器学习的两大挑战：欠拟合（Underfitting）和过拟合（Overfitting）。
&emsp;&emsp;**欠拟合**是指模型在训练集上的误差较大，通常由于**训练不充分**或**模型不合适**导致。
&emsp;&emsp;**过拟合**是指模型在训练集和测试集上的误差差距过大，通常由于**模型过分拟合了训练集中的随机噪音，导致泛化能力较差**。可采用**正则化**，降低泛化误差。

&emsp;&emsp;**容量（Capacity）**：是描述了整个模型拟合各种函数的能力，通过调节机器学习模型的容量，可以控制模型是否偏于过拟合还是欠拟合。如果容量不足，模型将不能够很好地表示数据，表现为欠拟合；如果容量过大，模型就很容易过分拟合数据，因为其记住了不适合测试集的训练集特性，表现为过拟合。容量的控制方法有：

* 选择控制模型的假设空间（Hypothesis Space），即学习算法可以选择为解决方案的函数集。
* 添加正则项对模型进行偏好排除。

&emsp;&emsp;当机器学习算法的容量适合于所执行任务的复杂度和所提供训练数据的数量时，算法效果通常最佳。

&emsp;&emsp;通常，当模型容量上升时，训练误差会下降，直到其渐近最小可能误差（假设误差度量有最小值）。通常，泛化误差是一个关于模型容量的U形曲线函数。如图所示，容量和误差之间的关系，左侧训练误差和泛化误差都非常高，为欠拟合机制。当容量增加时，训练误差减小，但训练误差和泛化误差之间的间距扩大，当间距的大小超过了训练误差的下降，则进入到了过拟合机制，其中容量过大，超过了最佳容量。

<img src="/images/capacity.png" width="600px"></img>

#### 没有免费午餐定理

&emsp;&emsp;**没有免费午餐定理（No Free Lunch Theorem）** ，通俗来讲，“没有最优的学习算法”。在所有的数据生成分布上平均后，最先进的算法和简单的算法在性能上相差无异。机器学习没有通用的学习算法来处理所有的概率分布问题，而是需要根据什么样的分布，采用什么样的机器学习算法在该数据分布上效果最好。

#### 正则化

&emsp;&emsp;正则化（Regularization）是指修改学习算法，使其降低泛化误差而非训练误差。

### 超参数和验证集

&emsp;&emsp;**超参数**:用来控制学习算法的参数而非学习算法本身学出来的参数。例如，进行曲线的回归拟合时，曲线的次数就是一个超参数；在构建模型对一些参数的分布假设也是超参数。

&emsp;&emsp;**验证集**：通常在需要选取超参数时，将训练集再划分为训练和验证集两部分，使用新的训练集训练模型，验证集用来进行测试和调整超参。通常，80%的训练数据用于训练学习参数，20%用于验证。

#### 交叉验证

&emsp;&emsp;**k折交叉验证**：将数据集均分为不相交的k份，每次选取其中的一份作为测试集，其他的为训练集，训练误差为k次的平均误差。

----------
**k-折交叉验证算法**
**Define** KFlodXV($\mathbb{D},A,L,k$):
**Require:** $\mathbb{D}$为给定的数据集，其中元素为$z^{(i)}$
**Require:** $A$为学习算法，可视为一个函数（使用数据集作为输入，输出一个学好的函数）
**Require:** $L$为损失函数，可视为来自学好的函数$f$，将样本$z^{(i)}\in \mathbb{D}$映射到$\mathbb{R}$中标量的函数
**Require:** $k$为折数
&emsp;&emsp;将$\mathbb{D}$分为$k$个互斥子集$\mathbb{D}\_{i}$，它们的并集为$\mathbb{D}$
&emsp;&emsp;for $i$ from 1 to $k$ do
&emsp;&emsp;&emsp;&emsp;$f\_{i}=A(\mathbb{D}\setminus \mathbb{D}\_{i})$
&emsp;&emsp;&emsp;&emsp;for $z^{(j)}$ in $\mathbb{D}\_{i}$ do
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$e\_{j}=L(f\_{i},z^{(j)})$
&emsp;&emsp;&emsp;&emsp;end for
&emsp;&emsp;end for
&emsp;&emsp;Return $e$
----------

##### 代码实现

```python

def KFoldCV(D, A, k):
    """
    k-fold 交叉验证
    参数说明：
    D：给定数据集
    A：学习函数
    k：折数
    """
    np.random.shuffle(D)
    dataset = np.split(D, k)
    acc_rate = 0
    for i in range(k):
        train_set = dataset.copy()
        test_set = train_set.pop(i)
        train_set = np.vstack(train_set)
        A.train(train_set[:,:-1], train_set[:,-1]) # 每次的训练集
        labels = A.fit(test_set[:,:-1]) # 每次的测试集
        acc_rate += np.mean(labels==test_set[:,-1]) # 计算平均误差
    return acc_rate/k

```

### 估计、偏差和方差

#### 点估计

&emsp;&emsp;点估计试图为一些感兴趣的量提供单个“最优”预测。

#### 偏差

&emsp;&emsp;估计的偏差被定义为
<center>$bias(\hat{\theta}_{m})=\mathbb{E}(\hat{\theta}_{m})-\theta$</center>
&emsp;&emsp;期望作用在所有数据上，$\theta$用于定于数据生成分布的$\theta$的真实值。

#### 方差和标准差

&emsp;&emsp;估计的方差被定义为
<center>$Var(\hat{\theta})$</center>
&emsp;&emsp;方差反映的是模型每一次输出结果与模型输出期望之间的误差，即模型的稳定性。

&emsp;&emsp;标准差被记为
<center>$SE(\hat{\mu _{m}})=\sqrt{Var\left [ \frac{1}{m}\sum_{i=1}^{m}x^{(i)} \right ]}=\frac{\sigma }{\sqrt{m}}$</center>
&emsp;&emsp;其中，$\sigma^{2}$是样本$x^{(i)}$的真实方差，标准差通常被记为$\sigma$。

#### 误差与偏差和方差的关系

&emsp;&emsp;泛化误差可分解为偏差、方差和噪音之和。需要在模型复杂度之间权衡，使偏差和方差得以均衡，这样模型的整体误差才会最小。
<img src="/images/capacity3.png" width="600px"></img>
&emsp;&emsp;当容量增大时，偏差随之减小；而方差随之增大，泛化误差为U型。

### 最大似然估计

&emsp;&emsp;**最大似然估计（Maximum Likelihood Estimation，MLE）**是一种最为常见的估计准则，其思想是在已知分布产生的一些样本而未知分布具体参数的情况下，根据样本值推断最有可能产生样本的参数值。将数据的真实分布记为$P_{data(x)}$，为了使⽤$MLE$，需要先假设样本服从某⼀簇有参数确定的分布$P_{model(x;\theta)}$，现在的⽬标就是使⽤估计的$P_{model}$来拟合真实的$P_{data}$(条件一:"模型已定，参数未知")。
&emsp;&emsp;对于⼀组由$m$个样本组成的数据集$X={x^{(1)},...,x^{(m)}}$，假设数据独⽴且由未知的真实数据分布$P_{data(x)}$⽣成 (条件二：独立同分布采样的数据)，可以通过最⼤似然估计，获取真实分布的参数。
<center>$\theta _{ML}=\underset{\theta}{arg\ max}P_{model}(X;\theta)=\underset{\theta}{arg\ max}\coprod_{i=1}^{m}P_{model}(x^{(i)};\theta)$</center>
&emsp;&emsp;通常为了计算⽅便，会对$MLE$加上$log$，将乘积转化为求和然后将求和变为期望：$\theta _{ML}=\underset{\theta}{arg\ max}\sum_{i=1}^{m}logP_{model}(x^{(i)};\theta)$ 。

&emsp;&emsp;使⽤训练数据经验分布$\hat{P}\_{data}$相关的期望进⾏计算：$\theta \_{ML}=\underset{\theta}{arg\ max}\mathbb{E}\_{x\sim \hat{P}\_{data}}logP\_{model}(x;\theta)$。该式是许多监督学习算法的基础假设。

&emsp;&emsp;最⼤似然估计的⼀种解释是使$P_{model}$与$P_{data}$之间的差异性尽可能的⼩，形式化的描述为最⼩化两者的$KL$散度。

&emsp;&emsp;<font color="#ff0000">定义看了半天，看了个寂寞，直接举例推导：</font>
<img src="/images/mle_example.png" width="500px"></img>
&emsp;&emsp;一枚硬币抛10次，得到$X$数据为{反，正，正，正，正，反，正，正，正，反}。得到似然函数$f(x_{0};\theta)=(1-\theta)\times\theta\times\theta\times\theta\times\theta\times(1-\theta)\times\theta\times\theta\times\theta\times(1-\theta)=(1-\theta)^{3}\times \theta ^{7}$
&emsp;&emsp;博客<font color="#ff0000"><sup>[[2][2]][[3][3]]</sup></font>中已经推导解释的非常好，负责将代码实现。

#### 手动推算

<img src="/images/mle_2.png" width="500px"></img>

#### 代码实现
```python
import numpy as np
import matplotlib.pyplot as plt

def f(theta): # f(theta)
    return (1-theta)**3 * theta**7

X = np.arange(0, 1, 0.001)
Y = f(X)
Y = np.array(Y)
plt.plot(X,Y)
plt.title("$f(theta) = (1-theta)^3theta^7$")
plt.show()

# output
```
<img src="/images/mle1.png" width="500px"></img>

&emsp;&emsp;可以看出，$\theta=0.7$时，似然函数取得最大值。

&emsp;&emsp;**通俗来讲，是利用已知的样本结果信息，反推最大概率导致这些样本结果出现的模型参考值。**极大似然估计提供了一种给定观察数据来评估模型参数的方法，即：“模型已定，参数未知”。通过若干次试验，观察其结果，利用试验结果得到某个参数值能够使样本出现的概率为最大，则称为极大似然估计。

#### 最大似然的性质

* 真实分布$p_{data}$必须在模型族$p_{model}(.;\theta)$中。否则，没有估计可以还原$p_{data}$。
* 真实分布$p_{data}$必须刚好对应一个$\theta$值。否则，最大似然估计恢复出真实分布$p_{data}$后，也不能决定数据生成过程使用哪个$\theta$。

### 贝叶斯统计

&emsp;&emsp;通过贝叶斯准则来估计参数的后验分布情况，贝叶斯统计（Bayesian Statistics）认为训练数据是确定的，而参数是随机且不唯一的，每个参数都有相应的概率。在观察数据之前，将$\theta$的已知知识表示成先验概率分布$p(\theta)$。如有一组数据样本$\\{x^{(1)},...,x^{(m)}\\}$，通过贝叶斯规则结合数据似然$p(x^{(1)},...,x^{(m)}\mid \theta)$和先验，得到：

<center>$p(\theta \mid x^{(1)},...,x^{(m)})=\frac{p(x^{(1)},...,x^{(m)}\mid \theta )p(\theta)}{p(x^{(1)},...,x^{(m)})}$</center>

&emsp;&emsp;相对于最大似然估计，贝叶斯估计有两个重要区别：第一，不像最大似然方法预测时使用$\theta$的点估计，贝叶斯方法使用$\theta$的全分布。第二，贝叶斯为先验分布，先验通常表现为偏好更简单或更光滑的模型，当训练数据有限时，贝叶斯方法通常泛化得更好，当训练样本数目很大时，通常计算代价很大。

#### 手动推算

#### 代码实现

#### 最大后验估计

&emsp;&emsp;完整的贝叶斯估计需要使用参数的完整分布进行预测，但计算繁重。最大后验估计（Maximum A Posterior，MAP）来选取一个计算可行的单点估计参数作为贝叶斯估计的近似解，公式：

<center>$\theta _{MAP}=\underset{\theta}{arg \ max}\ p(\theta \mid x)=\underset{\theta}{arg \ max}\ log \ p(x\mid \theta)+log\ p(\theta)$</center>

&emsp;&emsp;MAP的估计实际上就是对数似然加上参数的先验分布。实际上，在参数服从⾼斯分布的情况下，上式的右边就对应着L2正则项；在Laplace的情况下，对应着L1的正则项；在均匀分布的情况下则为0，等价于MLE。（太绕了）

&emsp;&emsp;最大似然估计（MLE）是求$\theta$使得似然函数$P(x_{0}\mid \theta)$最大。最大后验概率估计（MAP）是求$\theta$使得函数$P(x_{0}\mid \theta)P(\theta)$最大。$\theta$自己出现的先验概率也最大。

#### 代码实现

### 监督学习算法

&emsp;&emsp;监督学习算法是给定一组输入$x$和输出$y$的训练集，学习如何关联输入和输出。在[人工智能是什么][6]一文中已经对监督学习和无监督学习进行了整理和划分。

<img src="/images/jiandusuanfa.png" width="500px"></img>

&emsp;&emsp;机器学习算法后续会专门整理、归纳、总结。

#### 支持向量机

#### 其他简单监督学习算法

### 无监督学习算法

#### 主成分分析

详见：[DeepLearning学习笔记-2.线性代数][4] 

#### k-均值聚类

### 随机梯度下降

详见：[DeepLearning学习笔记-4.数值计算][5] 

### 构建机器学习算法

&emsp;&emsp;几乎所有的深度学习算法都是一样的流程：特定的数据集、代价函数、优化过程和模型。

### 深度学习发展的挑战

&emsp;&emsp;高维数据在新样本上泛化苦难，传统机器学习中实现泛化机制不适合学习高维空间中的复杂函数，涉及到巨大空间问题，计算代价很大，深度学习则旨在客服这些以及其他一些难题。

##### 维数灾难

&emsp;&emsp;当数据的维数很高时，很多机器学习问题变得相当困难，这种现象被称为维数灾难（Curse of Dimensionality）。维数灾难带来的一个挑战是统计挑战。

#### 局部不变形和平滑正则化

&emsp;&emsp;为更好地泛化，机器学习算法需要由先验信念引导该学习什么类型的函数。先验信念还间接地体现在选择一些偏好某类函数的算法。其中使用最广泛的隐式先验是“平滑先验”和“局部不变性先验”，表明学习的函数不应在小区域内发生很大的变化。

#### 流行学习

&emsp;&emsp;**流行（Manifold）**是指连接在一起的区域。**流行学习（Manifold Learning）**算法通过一个假设来克服很多机器学习问题无望的障碍。该假设认为$R^{n}$中大部分区域都是无效的输入，有意义的输入只分布在包含少量数据点的子集构成的一组流行中，而学习函数的输出中，有意义的变化都沿着流行的方向或仅发生在我们切换到另一流行时。（难懂，先看应用场景，后续再研究）

&emsp;&emsp;主要用于图像降维<sup>[[7][7]]</sup>。


[1]:https://www.pianshen.com/article/459993619/
[2]:https://blog.csdn.net/u011508640/article/details/72815981
[3]:https://zhuanlan.zhihu.com/p/46737512
[4]:https://deeplearner.top/2021/12/09/DeepLearning%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0-2-%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0/#%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90
[5]:https://deeplearner.top/2022/01/04/DeepLearning%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0-4-%E6%95%B0%E5%80%BC%E8%AE%A1%E7%AE%97/#%E5%9F%BA%E4%BA%8E%E6%A2%AF%E5%BA%A6%E7%9A%84%E4%BC%98%E5%8C%96%E6%96%B9%E6%B3%95
[6]:https://deeplearner.top/2021/11/29/AI%E7%9F%A5%E8%AF%86%E4%BD%93%E7%B3%BB%E5%8F%8A%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/
[7]:https://zhuanlan.zhihu.com/p/40214106