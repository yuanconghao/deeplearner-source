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

&emsp;&emsp;**没有免费午餐定理（No Free Lunch Theorem）** ，在所有的数据生成分布上平均后，最先进的算法和简单的算法在性能上相差无异。通俗来讲，机器学习没有通用的学习算法来处理所有的概率分布问题，而是需要根据什么样的分布，采用什么样的机器学习算法在该数据分布上效果最好。

#### 正则化



### 超参数和验证集

#### 交叉验证

### 估计、偏差和方差

#### 点估计

#### 偏差

#### 方差和标准差

#### 权衡偏差和方差以最小化均方误差

#### 一致性

### 最大似然估计

#### 条件对数似然和均方误差

#### 最大似然的性质

### 贝叶斯统计

#### 最大后验估计

### 监督学习算法

#### 概率监督学习

#### 支持向量机

#### 其他简单监督学习算法

### 无监督学习算法

#### 主成分分析

#### k-均值聚类

### 随机梯度下降

### 构建机器学习算法

### 深度学习发展的挑战

##### 维数灾难

#### 局部不变形和平滑正则化

#### 流行学习




[1]:https://www.pianshen.com/article/459993619/