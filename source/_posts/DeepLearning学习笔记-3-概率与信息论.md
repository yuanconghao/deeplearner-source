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

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

# 生成样本
fix, ax = plt.subplots(1,1)
r = uniform.rvs(loc=0, scale=1, size=1000)
ax.hist(r, density=True, histtype='stepfilled', alpha=0.5)

# 均匀分布 pdf
x = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
ax.plot(x, uniform.pdf(x), 'r-', lw=5, alpha=0.8, label='uniform pdf')

# output:
```

<img src="/images/uniform.png"></img>


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

```python
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([9,8,7,6,5,4,3,2,1])

Mean = np.mean(x) # 平均值
Var = np.var(x)   # 默认总体方差
Var_unbias = np.var(x, ddof=1) # 样本方差（无偏方差）
Cov = np.cov(x,y) # 协方差
print("平均值：", Mean)
print("默认方差：", Var)
print("样本方差：", Var_unbias)
print("协方差：\n", Cov)

# output 
平均值： 5.0
默认方差： 6.666666666666667
样本方差： 7.5
协方差：
 [[ 7.5 -7.5]
 [-7.5  7.5]]
```

```python
def plot_distribution(X, axes=None):
    """ 给定随机变量，绘制 PDF，PMF，CDF"""
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    x_min, x_max = X.interval(0.99) 
    x = np.linspace(x_min, x_max, 1000)
    if hasattr(X.dist, 'pdf'): # 判断有没有 pdf，即是不是连续分布
        axes[0].plot(x, X.pdf(x), label="PDF")
        axes[0].fill_between(x, X.pdf(x), alpha=0.5) # alpha 是透明度，alpha=0 表示 100% 透明，alpha=100 表示完全不透明
    else: # 离散分布
        x_int = np.unique(x.astype(int))
        axes[0].bar(x_int, X.pmf(x_int), label="PMF") # pmf 和 pdf 是类似的
        axes[1].plot(x, X.cdf(x), label="CDF")
    for ax in axes:
        ax.legend()
    return axes


from scipy.stats import bernoulli
fig, axes = plt.subplots(1, 2, figsize=(10, 3)) # 画布
p = 0.3
X = bernoulli(p) # 伯努利分布
plot_distribution(X, axes=axes)

# output
```
<img src="/images/pmf_cdf.png"></img>


#### 手动推算

### 常用概率分布

#### Bernoulli分布

&emsp;&emsp;伯努利分布（Bernoulli Distribution），是**单个二值随机变量**的分布，又叫两点分布。由单个参数$\phi \in\left [ 0,1 \right ]$控制，$phi$给出了随机变量等于1的概率。表示一次试验结果要么成功要么失败。具有如下性质：

* $P(x=1)=\phi$
* $P(x=1)=1-\phi$
* $P(X=x)=\phi ^{x}(1-\phi)^{1-x}$

#### 代码实现

```python
# 产生成功的概率
possibility = 0.3
def trials(n_samples):
    samples = np.random.binomial(n_samples, possibility) # 成功的次数
    proba_zero = (n_samples-samples)/n_samples
    proba_one = samples/n_samples
    return [proba_zero, proba_one]

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
# 一次试验， 伯努利分布
n_samples = 1
axes[0].bar([0, 1], trials(n_samples), label="Bernoulli")
# n 次试验， 二项分布
n_samples = 1000
axes[1].bar([0, 1], trials(n_samples), label="Binomial")
for ax in axes:
    ax.legend()

# output
```
<img src="/images/bernoulli.png"></img>

#### 手动推算


#### Multinoulli分布

&emsp;&emsp;范畴分布（Multinoulli Distribution），是指在具有$k$个不同值的单个离散型随机变量上的分布，其中$k$是一个有限值。例如每次试验结果就可以记为一个$k$维的向量，只有此次试验的结果对应的维度记为1，其他记为0。公式：

<center>$p(X=x)=\prod _{i} \phi _{i}^{x_{i}}$</center>

#### 代码实现

```python
def k_possibilities(k):
    """
    随机产生一组 10 维概率向量
    """
    res = np.random.rand(k)
    _sum = sum(res)
    for i, x in enumerate(res):
        res[i] = x / _sum
    return res

fig, axes = plt.subplots(1, 2, figsize=(10, 3)) # 一次试验， 范畴分布
k, n_samples = 10, 1
samples = np.random.multinomial(n_samples, k_possibilities(k)) # 各维度“成功”的次数
axes[0].bar(range(len(samples)), samples/n_samples, label="Multinoulli")
n_samples = 1000 # n 次试验， 多项分布
samples = np.random.multinomial(n_samples, k_possibilities(k))
axes[1].bar(range(len(samples)), samples/n_samples, label="Multinomial")
for ax in axes:
    ax.legend()

# output
```
<img src="/images/random.png"></img>

#### 手动推算

#### 高斯分布

&emsp;&emsp;高斯分布（Gaussian Distribution），实数上常用最常用的分布，又称正态分布（Normal Distribution）:

<center>$N(x;\mu ,\sigma ^{2})=\sqrt{\frac{1}{2\pi \sigma ^{2}}}exp \left (-\frac{1}{2}\beta (x-\mu)^{2} \right )$ </center>

&emsp;&emsp;正态分布由两个参数控制，$\mu \in \mathbb{R}$和$\sigma \in (0,\propto )$。参数$\mu$给出了中心峰值的坐标，也是分布的均值。分布的标准差用$\sigma$表示，方差用$\sigma ^{2}$表示。当需要对概率密度函数求值时，需要对$\sigma$平方并且取倒数，$\beta = \frac{1}{\sigma ^{2}}$。如下图所示：

<img src="/images/normal_dis.png"></img>

&emsp;&emsp;其中$\mu=1,\sigma=1$ 称为**标准正态分布**。

&emsp;&emsp;中心极限定理说明很多独立随机变量的和近似服从正态分布，因此可认为噪声是属于正态分布的。

#### 代码实现

```python
from scipy.stats import norm
fig, axes = plt.subplots(1, 2, figsize=(10, 3)) # 画布
mu, sigma = 0, 1 
X = norm(mu, sigma) # 标准正态分布
plot_distribution(X, axes=axes)

# output
```
<img src="/images/pdf_cdf.png"></img>


#### 手动推算

#### 多元高斯分布（多元正态分布）

&emsp;&emsp;正态分布可以推广到$\mathbb{R}^{n}$空间，这种情况被称为**多维正态分布** ，形式如下：

<center>$N(x;\mu,\Sigma )=\sqrt{\frac{1}{(2\pi)^{n}det(\Sigma)}}exp \left (-\frac{1}{2}(x-\mu)^{\top} \Sigma^{-1}(x-\mu)\right )$</center>

&emsp;&emsp;对概率密度函数求值时，对$\Sigma$求逆，可使用一个精度矩阵$\beta$进行替代。$\beta=\frac{1}{\Sigma}$ 。

#### 代码实现

```python
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
x, y = np.mgrid[-1:1:.01, -1:1:.01]
pos = np.dstack((x, y))
fig = plt.figure(figsize=(4,4))
axes = fig.add_subplot(111)
mu = [0.5, -0.2] # 均值
sigma = [[2.0, 0.3], [0.3, 0.5]] # 协方差矩阵
X = multivariate_normal(mu, sigma)
axes.contourf(x, y, X.pdf(pos))

# output
```
<img src="/images/mul_normal.png"></img>

#### 手动推算

#### 指数分布

&emsp;&emsp;指数分布（Exponential Distribution），形式如下：

<center>$p(x;\lambda )=\lambda 1_{x\geqslant 0}exp(-\lambda x)$</center>

&emsp;&emsp;用于在$x=0$处取得临界点的分布，其中$\lambda > 0$是分布的一个参数，常被称为率参数。

#### 代码实现

```python
from scipy.stats import expon
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
# 定义 scale = 1 / lambda
X = expon(scale=1)
plot_distribution(X, axes=axes)

# output
```
<img src="/images/expon.png"></img>

#### 拉普拉斯分布

&emsp;&emsp;拉普拉斯分布（Laplace Distribution），形式如下：

<center>$Laplace(x;\mu , \gamma )=\frac{1}{2\gamma}exp\left ( -\frac{\left|x-\mu \right|}{\gamma } \right )$</center>

&emsp;&emsp;允许在任意一个点$\mu$处设置概率质量峰值。

#### 代码实现

```python
from scipy.stats import laplace
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
mu, gamma = 0, 1 
X = laplace(loc=mu, scale=gamma)
plot_distribution(X, axes=axes)

# output
```
<img src="/images/laplace.png"></img>

#### Dirac分布和经验分布

&emsp;&emsp;Dirac delta函数定义概率密度函数来实现：$p(x)=\delta (x-\mu)$ ，被定义成除了0以外的所有点的值都为0，但积分为1，是一个泛函数。常用于组成**经验分布（Empirical Distribution）** ：

<center>$\hat{p}(x)=\frac{1}{m}\sum_{m}^{i=1}\delta (x-x^{(i)})$</center>

#### 混合分布

&emsp;&emsp;混合分布（Mixture Distribution），通过组合一些简单的概率分布来定义新的概率分布。

### 常用函数的有用性质

&emsp;&emsp;深度学习模型中常用到的概率分布。

#### Logistic sigmoid函数

&emsp;&emsp;**logistic sigmoid函数**通常用来产生Bernoulli分布中的参数$\phi$ ，因为它的范围是$(0,1)$ ，在$\phi$的有效取值范围内。形式如下：

<center>$\sigma(x)=\frac{1}{1+exp(-x)}$</center>

&emsp;&emsp;下图给出了sigmoid函数的图示，sigmoid函数在变量取绝对值非常大的正值或负值时会出现**饱和（saturate）**现象，意味着函数会变得很平，并且对输入的微小改变会变得不敏感。

#### softplus函数

&emsp;&emsp;**softplus函数** :

<center>$\xi=log(1+exp(x))$</center>

可以用来产生正态分布的$\beta$和$\apha$参数，它的范围是$(0,\propto )$。当处理包含sigmoid函数的表达式时它也经常出现。softplus函数名来源于它是另外一个函数的平滑形式，函数如下：

<center>$x^{+}=max(0,x)$</center>

&emsp;&emsp;函数的一些性质如下：

* <font color="#ff0000">$\sigma(x)=\frac{exp(x)}{exp(x)+exp(0)}$</font>
* <font color="#ff0000">$\frac{d}{dx}\sigma (x)=\sigma(x)(1-\sigma(x))$</font>
* <font color="#ff0000">$1-\sigma(x)=\sigma(-x)$</font>
* <font color="#ff0000">$log\sigma(x)=-\xi (-x)$</font>
* <font color="#ff0000">$\frac{d}{dx}\xi (x)=\sigma (x)$</font>
* <font color="#ff0000">$\sigma ^{-1}(x)=log\left ( \frac{x}{1-x} \right ),\forall x\in (0,1)$</font>
* <font color="#ff0000">$\xi ^{-1}(x)=log(exp(x)-1),\forall x>0$</font>
* <font color="#ff0000">$\xi (x)=\int_{-\propto }^{x}\sigma (y)dy$</font>
* <font color="#ff0000">$\xi (x)-\xi (-x)=x$</font>

&emsp;&emsp;softplus函数被设计成**正部函数** ，$x^{+}=max{0,x}$ 和 **负部函数** ，$x^{-}=max(0,-x)$ 。

&emsp;&emsp;函数的图示如下：

#### 代码实现

#### 代码实现

```python
x = np.linspace(-10, 10, 100)
sigmoid = 1/(1 + np.exp(-x))
softplus = np.log(1 + np.exp(x))
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(x, sigmoid, label='sigmoid')
axes[1].plot(x, softplus, label='softplus')
for ax in axes:
    ax.legend()

# output
```
<img src="/images/sigmoid.png"></img>

### 贝叶斯规则

&emsp;&emsp;贝叶斯规则（Bayes rule），在已知$P(y\mid x)$时计算$P(x\mid y)$，计算公式如下：

<center>$P(x\mid y)=\frac{P(x)P(y\mid x)}{P(y)}$</center> 


### 信息论

&emsp;&emsp;**信息论**是应用数学的一个分支，主要研究的是对一个信号包含信息的多少进行量化。**信息论背后的思想：**一件不太可能的事件比一件比较可能的事件更有信息量。

&emsp;&emsp;**信息（Information）**需要满足三个条件：
* 非常可能发生的事件信息量比较少。
* 较不可能发生的事件具有更高的信息量。
* 独立事件应具有增量的信息。

&emsp;&emsp;**自信息（Self-Information）** ：对事件$X=x$，定义：

<center>$I(x)=-logP(x)$</center>

&emsp;&emsp;$log$为底为$e$的自然对数；$I(x)$单位为**奈特（nats）** 。

&emsp;&emsp;**香农熵（Shannon Entropy）** ：自信息只包含一个事件的信息，而对于**整个概率分布中的不确定性总量**可用香农熵进行量化：

<center>$H(x)=\mathbb{E}_{X\sim P}\left [ I(x) \right ]=-\mathbb{E}_{X\sim P}\left [ logP(x) \right ]$</center>

&emsp;&emsp;一个分布的香农熵是指这个分布的事件所产生的期望信息总量。香农熵是编码原理中最优编码长度。

### 结构化概率模型