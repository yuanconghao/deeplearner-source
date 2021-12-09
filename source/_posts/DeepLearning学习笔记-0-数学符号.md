---
title: DeepLearning学习笔记-0.数学符号
date: 2021-12-08 19:53:13
toc: true
mathjax: true
categories:
    - 深度学习

tags:
    - 深度学习笔记
    - 花书
---
《深度学习》本书所使用数学符号。

<!--more-->
### 数和数组

| 符号 | 含义 | LaTex表示 |
| --- | --- | --- |
| $\mathit{a}$ | 标量（整数或实数）| \mathit{a} | 
| $\vec{a}$ | 向量 | \vec{a} | 
| $\mathit{A}$ | 矩阵 | \mathit{A} | 
| $\mathbf{A}$ | 张量 | \mathbf{A} | 
| $\mathit{I}\mathit{n}$ | n行n列的单位矩阵 | \mathit{I}\mathit{n} | 
| $\mathit{I}$ | 维度蕴含于上下文的单位矩阵 | \mathit{I} | 
| $e^{(i)}$ | 标准基向量[0,...,0,1,0,...,0]，其中索引i处的值为1 | e^{(i)} | 
| $diag(a)$ | 对角方阵，其中对角元素由$\vec{a}$给定 | diag(a) | 
| a | 标量随机变量 | a | 
| a | 向量随机变量 | a | 
| A | 矩阵随机变量 | A | 

### 集合和图

| 符号 | 含义 | LaTex表示 |
| --- | --- | --- |
| $\mathbb{A}$ | 集合 | \mathbb{A} | 
| $\mathbb{R}$ | 实数集 | \mathbb{R} | 
| $\\{0,1\\}$ | 包含0和1的集合 | \\{0,1\\} | 
| $\\{0,1,...,n\\}$ | 包含0和n之间所有整数的集合 | \\{0,1,...,n\\} | 
| $\left [ a,b \right ]$ | 包含a和b的实数区间 | \left [ a,b \right ] | 
| $\left ( a,b \right ]$ | 不包含a但包含b的实数区间 | \left ( a,b \right ] | 
| $\mathbb{A}\backslash\mathbb{B}$ | 差集，即其元素包含于A，但不包含于B | \mathbb{A}\backslash\mathbb{B} | 
| $\mathit{G}$ | 图 | \mathit{G} | 
| $P_{ag}( x_{i})$ | 图 $\mathit{G}$ 中 $( x_{i})$ 的父节点 | P_{ag}( x_{i}) | 

### 索引

| 符号 | 含义 | LaTex表示 |
| --- | --- | --- |
| $\vec{a}_{i}$ | 向量a的第i个元素，其中索引从1开始 | \vec{a}_{i} | 
| $\vec{a}_{-i}$ | 除了第i个元素，a的所有元素 | \vec{a}_{-i} | 
| $A_{i,j}$ | 矩阵A的i,j元素 | A_{i,j} | 
| $A_{i,:}$ | 矩阵A的第i行 | A_{i,:} | 
| $A_{:,i}$ | 矩阵A的第i列 | A_{:,i} | 
| $\mathbf{A}_{i,j,k}$ | 三维张量A的(i,j,k)元素 | \mathbf{A}_{i,j,k} | 
| $\mathbf{A}_{:,:,i}$ | 三维张量的二维切片 | \mathbf{A}_{:,:,i} | 
| $a_{i}$ | 随机向量a的第i个元素 | a_{i} | 


### 线性代数中的操作

| 符号 | 含义 | LaTex表示 |
| --- | --- | --- |
| $A^{\top}$ | 矩阵A的转置 | A^{\top} | 
| $A^{+}$ | 矩阵A的Moore-Penrose伪逆 | A^{+} | 
| $A\odot B$ | A和B的遂元素乘积（Hadamard乘积） | A\odot B | 
| $\det(A)$ | A的行列式 | \det(A) | 

### 微积分

| 符号 | 含义 | LaTex表示 |
| --- | --- | --- |
| $\frac{\mathrm{d} y}{\mathrm{d} x}$ | y关于x的导数 | \frac{\mathrm{d} y}{\mathrm{d} x} | 
| $\frac{\partial y}{\partial x}$ | y关于x的偏导 | \frac{\partial y}{\partial x} |
| $\nabla_{x}y$ | y关于x的梯度 | \nabla_{x}y |
| $\nabla_{X}y$ | y关于X的矩阵导数 | \nabla_{X}y |
| $\nabla_{\mathbf{X}}y$ | y关于$\mathbf{A}$求导后的张量 | \nabla_{\mathbf{X}}y |
| $\frac{\partial f}{\partial x}$ | f:$\mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ 的Jacobian矩阵$J\in \mathbb{R}^{m\times n}$ | \frac{\partial f}{\partial x} |
| $\nabla_{2}^{x}f\(x\)\ or \ H\(f\)\(x\)$ | f在点x处的Hessian矩阵 | \nabla_{2}^{x}f\(x\)\ or \ H\(f\)\(x\) |
| $\int f\(x\)d_{x}$ | x整个域上的定积分 | \int f\(x\)d_{x} |
| $\int_{\mathbb{S}}f\(x\)d_{x}$ | 集合$\mathbb{S}$上关于x的定积分 | \int_{\mathbb{S}}f\(x\)d_{x} |

### 概率和信息论

| 符号 | 含义 | LaTex表示 |
| --- | --- | --- |
| $a\perp b$ | a和b相互独立的随机变量 | a\perp b | 
| $a\perp b \ \mid  \ c$ | 给定c后条件独立 | a\perp b \ \mid  \ c | 
| $P\(a\)$ | 离散变量上的概率分布 | P\(a\) | 
| $p\(a\)$ | 连续变量（或变量类型未指定时）上的概率分布 | p\(a\) | 
| $a\sim P$ | 具有分布P的随机变量a | a\sim P | 
| $\mathbb{E}_{x \sim P}[f(x)] \ or \ \mathbb{E}f(x)$ | f(x)关于P(x)的期望 | \mathbb{E}_{x \sim P}[f(x)] \ or \ \mathbb{E}f(x) | 
| $Var(f(x))$ | f(x)在分布P(x)下的方差 | Var(f(x)) | 
| $Cov(f(x),g(x))$ | f(x)和g(x)在分布P(x)下的协方差 | Cov(f(x),g(x)) | 
| $H(x)$ | 随机变量x的香农熵 | H(x) | 
| $D_{KL}(P\parallel Q)$ | P和的KL散度 | D_{KL}(P\parallel Q) | 
| $N(x;\mu ,\Sigma )$ | 均值为$\mu$,协方差为$\Sigma$,x上的高斯分布 | N(x;\mu ,\Sigma ) | 

### 函数

| 符号 | 含义 | LaTex表示 |
| --- | --- | --- |
| $f:\mathbb{A}\rightarrow \mathbb{B}$ | 定义域为$\mathbb{A}$，值域为$\mathbb{B}$的函数f | f:\mathbb{A}\rightarrow \mathbb{B} | 
| $f\circ g$ | f和g的组合 | f\circ g | 
| $f(x;\theta )$ | 由$\theta$参数化，关于x的函数（有时为简化表示，我们忽略$\theta$而记为f(x)） | f(x;\theta ) | 
| $\log x$ | x的自然对数 | \log x | 
| $\sigma (x)$ | Logistic sigmoid，$\frac{1}{1+exp(-x)}$ | \sigma (x) | 
| $\zeta (x)$ | Softplus，$\log(1+\exp(x))$ | \zeta (x) | 
| $\parallel x \parallel _{p}$ | x的$L^{p}$范数 | \parallel x \parallel _{p} | 
| $\parallel x \parallel$ | x的$L^{2}$范数 | \parallel x \parallel | 
| $x^{+}$ | x的正数部分，即max(0,x) | x^{+}$ | 
| $1_{condition}$ | 如果条件为真则为1，否则为0 | 1_{condition} | 

### 数据集合发布

| 符号 | 含义 | LaTex表示 |
| --- | --- | --- |
| $p_{data}$ | 数据生成分布 | p_{data} | 
| $\hat{p}_{train}$ | 由训练集定义的经验分布 | \hat{p}_{train} | 
| $\mathbb{X}$ | 训练样本的集合 | \mathbb{X} | 
| $x^{(i)}$ | 数据集的第i个样本（输入） | x^{(i)} | 
| $y^{(i)}$或<b>$y^{(i)}$</b> | 监督学习中与$x^{(i)}关联的目标$ | y^{(i)} | 
| $\mathit{X}$ | $m\times n$的矩阵，其中行$X_{i,:}$为输入样本$x^{(i)}$ | \mathit{X} | 



