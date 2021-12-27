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

&emsp;&emsp;**概率质量函数（Probability Mass Function,PMF）** 常用于描述离散型变量的概率分布。通常用大写字母$P$表示，如$P(x)$,定义一个随机变量，用$\sim $符号来说明它遵循的分布：$x\sim P(x)$ 。

&emsp;&emsp;**联合概率分布（Joint Probability Distribution）** 常用于表示多个随机变量的概率分布，如$P(x,y)$，表示$x,y$同时发生的概率。

&emsp;&emsp;如果一个函数$P$是随机变量$x$的概率质量函数(PMF)，则必须满足以下条件：
* $P$的定义域必须是$x$所有可能状态的集合。
* $\forall x\in X,0\leqslant P(x)\leqslant 1$ 。
* 归一化：$\sum_{x\in X}P(x)=1$ 。

#### 概率密度函数

&emsp;&emsp;连续型随机变量用，**概率密度函数（Probability Density Functino,PDF）** 来描述它的概率分布，而不是概率质量函数。函数$p$是概率密度函数，必须满足以下条件：

* $p$的定义域必须是$x$所有可能状态的集合。
* $\forall x\in X,p(x)\geqslant 0$ , 并不要求 $p(x)\leqslant 1$ 。
* $\int p(x)d(x)=1$ 。

#### 代码实现


### 边缘概率

&emsp;&emsp;一组变量的联合概率分布中的一个子集的概率分布称为**边缘概率（Marginal Probability）** 。例如：设离散型随机变量$x$和$y$，并且我们知道$P(x,y)$，可依据如下**求和法则**来计算$P(x)$:

<center>$\forall x\in X,P(X=x)=\sum _{y}P(X=x,Y=y)$</center></br>

&emsp;&emsp;连续型变量，可用积分替代求和：

<center>$p(x)=\int p(x,y)dy$</center></br>

### 条件概率

&emsp;&emsp;某个事件，在给定其他事件发生时出现的概率，这种概率称为**条件概率（Conditional Probability）** 。例如：我们给定$X=x,Y=y$发生的条件概率记为：$P(Y=y \mid X=x)$。计算公式：

<center>$P(Y=y\mid X=x)=\frac{P(Y=y,X=x)}{P(X=x)}$</center></br>

### 条件概率的链式法则

&emsp;&emsp;任何多维随机变量的联合概率分布，都可以分解成为只有一个变量的条件概率相乘的形式：

<center>$P(X^{(1)},...,X^{(n)})=P(X^{(1)})\prod ^{n}_{i=2}P(X^{(i)}\mid X^{(1)},...,X^{(i-1)})$</center></br>

&emsp;&emsp;这个规则被称为概率的**链式法则** ，如：

<center>$P(a,b,c)=P(a\mid b,c)P(b,c)$</center></br>
<center>$P(b,c)=P(b\mid c)P(c)$</center></br>
<center>$P(a,b,c)=P(a\mid b,c)P(b\mid c)P(c)$</center></br>

### 独立性和条件独立性

&emsp;&emsp;两个随机变量$x$和$y$，它们的概率分布可以表示成两个因子的乘积形式，并且一个因子只包含$x$，另一个因子只包含$y$，称这两个随机变量是**相互独立的（Independent）** ：

<center>$\forall x\in X, \forall y \in Y, P(X=x,Y=y)=P(X=x)P(Y=y)$</center></br>

&emsp;&emsp;如果关于$x$和$y$的条件概率分布对于$z$的每一个值都可以写成乘积形式，那么这两个随机变量$x$和$y$在给定随机变量$z$时是**条件独立的（Conditionally Independent）** ：

<center>$\forall x\in X, \forall y \in Y, z\in X,P(X=x,Y=y \mid Z=z)=P(X=x\mid Z=z)P(Y=y\mid Z=z)$</center></br>

### 期望、方差和协方差

&emsp;&emsp;**期望（Expectation）** :当$x$由$P$产生，$f$作用于$x$时，$f(x)$的平均值。

&emsp;&emsp;**离散型随机变量期望** ：
<center>$\mathbb{E}_{x\sim P[f(x)]}=\sum _{x}P(x)f(x)$</center></br>

&emsp;&emsp;**连续型随机变量期望** ：
<center>$\mathbb{E}_{x\sim p[f(x)]}=\int p(x)f(x)dx$</center></br

&emsp;&emsp;**方差（Variance）** : 依据$x$进行采样时，用于衡量随机变量$x$的函数值呈现多大的差异：

<center>$Var(f(x))=\mathbb{E}[(f(x)-\mathbb{E}[f(x)])^{2}]$</center></br>

&emsp;&emsp;当方差很小时，$f(x)$的值形成的簇比较接近它们的期望值。方差的平方根称为**标准差（Standard Deviation）** 。

&emsp;&emsp;**协方差（Covariance）**  : 给出了两个变量线性相关性的强度以及这些变量的尺度：

<center>$Cov(f(x),g(y))=\mathbb{E}[(f(x)-\mathbb{E}[f(x)])(g(y)-\mathbb{E}[g(y)])]$</center></br>

#### 代码实现

#### 手动推算

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