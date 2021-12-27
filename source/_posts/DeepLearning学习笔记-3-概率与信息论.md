---
title: DeepLearning学习笔记-3.概率与信息论
date: 2021-12-22 02:37:20
toc: true
mathjax: true
categories:
    - 深度学习

tags:
    - 深度学习笔记
    - 花书
---

概率论用于表示不确定性声明的数学框架。人工智能领域，主要用于推理和统计分析AI系统行为。

<!--more-->

### 为什么要使用概率

&emsp;&emsp;传统计算机编程，通过给CPU确定的指令，来完成相关工作。机器学习通常存在不确定量，如建模系统内在的随机性、不完全观测、不完全建模等，所以需要通过一种用于对不确定性进行表示和推理的方法，概率则可作为对不确定性的扩展逻辑，提供了一套形式化的规则，可在给定某些命题的真或假的假设下，判断另外一种命题是真还是假。

### 随机变量

&emsp;&emsp;**随机变量 (Random Variable)** :一个可能随机取不同值的变量。例如:抛掷一枚硬币，出现正面或者反面的结果。

### 概率分布

&emsp;&emsp;**概率分布（Probability Distribution）** 用来描述随机变量或一簇随机变量在每一个可能取到的状态的可能性大小。

#### 概率质量函数

&emsp;&emsp;**概率质量函数（Probability Mass Function,PMF）** 常用于描述离散型变量的概率分布。通常用大写字母$P$表示，如$P(x)$,

&emsp;&emsp;**联合概率分布（Joint Probability Distribution）** 常用于表示多个随机变量的概率分布，如$P(x,y)$

#### 概率密度函数

研究对象为连续型时，用**概率密度函数（）** 而不是概率质量函数来描述它的概率分布。

#### 累积分布函数

### 边缘概率

**边缘概率（Marginal Probability）** : 

### 条件概率

**条件概率（Conditional Probability）** : 

### 条件概率的链式法则

**条件概率的链式法则（Chain Rule of Conditional Probability）** : 

### 独立性和条件独立性

**独立性（Independence）** : 
**条件独立性（Conditional Independence）** : 

### 期望、方差和协方差

**期望（Expectation）** :

**方差（Variance）** :

**协方差（Covariance）** :

### 常用概率分布

#### Bernoulli分布

&emsp;&emsp;伯努利分布（Bernoulli Distribution），又叫两点分布。


#### Multinoulli分布

&emsp;&emsp;范畴分布（Multinoulli Distribution）

#### 高斯分布

&emsp;&emsp;高斯分布（Gaussian Distribution），实数上常用最常用的分布，又称正态分布（Normal Distribution）。

#### 指数分布

&emsp;&emsp;指数分布（Exponential Distribution），

#### 拉普拉斯分布

&emsp;&emsp;拉普拉斯分布（Laplace Distribution），

#### Dirac分布

#### 经验分布

&emsp;&emsp;经验分布（Empirical Distribution），

#### 混合分布

&emsp;&emsp;混合分布（Mixture Distribution），

### 常用函数的有用性质

#### Logistic sigmoid函数

#### softplus函数

### 贝叶斯规则

&emsp;&emsp;贝叶斯规则（Bayes rule）

### 信息论

### 结构化概率模型