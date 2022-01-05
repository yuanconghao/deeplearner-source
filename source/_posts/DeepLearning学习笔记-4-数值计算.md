---
title: DeepLearning学习笔记-4.数值计算
date: 2022-01-04 01:35:24
toc: true
mathjax: true
categories:
    - 深度学习

tags:
    - 深度学习笔记
    - 花书
---

机器学习算法需要大量数值计算，通常是指通过**迭代**过程更新解得估计值来解决数学问题的算法，而不是通过解析过程推导出公式来提供正确解的方法。常见的操作包括**优化**和**线性方程组**的求解。

<!--more-->

### 上溢和下溢

&emsp;&emsp;无限多的实数无法在计算机内精确保存，因此计算机保存实数时，几乎总会引入一些近似误差，单纯的舍入误差会导致一些问题，特别是当操作复合时，即使是理论上可行的算法，如果没有考虑最小化舍入误差的累积，在实践时也会导致算法失效。

&emsp;&emsp;**下溢（Underflow）** ： 当接近零的数被四舍五入为零时发生下溢。
&emsp;&emsp;**上溢（Overflow）** ： 当大量级的数被近似为$\propto $或$-\propto $时发生上溢。

&emsp;&emsp;必须对上溢和下溢进行数值稳定的一个例子是softmax函数（解决上溢和下溢）。softmax函数经常用于预测和范畴分布相关联的概率，定义为：

<center>$softmax(x)_{i}=\frac{exp(x_{i})}{\sum_{j=1}^{n}exp(x_{j})}$</center>

&emsp;&emsp;当所有$x_{i}$都等于某个常数$c$时，所有的输出都应该为$\frac{1}{n}$，当$$是很小的负数，$exp(c)$就会下溢，函数分母会变为0，所以结果是未定义的。当$c$是非常大的正数时，$exp(c)$的上溢再次导致整个表达式未定义。
&emsp;&emsp;两个困难都通过计算$softmax(z)$同时解决，其中$z=x-max_{i}x_{i}$ 。

#### 代码实现

```python
import numpy as np
import numpy.linalg as la
```

```python
x = np.array([1e7, 1e8, 2e5, 2e7])
y = np.exp(x) / sum(np.exp(x))
print("上溢：", y)
x = x - np.max(x) # 减去最大值
y = np.exp(x) / sum(np.exp(x))
print("上溢处理：", y)

# output
上溢： [nan nan nan nan]
上溢处理： [0. 1. 0. 0.]
```

```python
x = np.array([-1e10, -1e9, -2e10, -1e10])
y = np.exp(x) / sum(np.exp(x))
print("下溢：", y)
x = x - np.max(x) # 减去最大值
y = np.exp(x) / sum(np.exp(x))
print("下溢处理：", y)
print("log softmax(x):", np.log(y))
# 对 log softmax 下溢的处理：
def logsoftmax(x):
    y = x - np.log(sum(np.exp(x)))
    return y
print("logsoftmax(x):", logsoftmax(x))

# output
下溢： [nan nan nan nan]
下溢处理： [0. 1. 0. 0.]
log softmax(x): [-inf   0. -inf -inf]
logsoftmax(x): [-9.0e+09  0.0e+00 -1.9e+10 -9.0e+09]
```

#### 手动推算

### 病态矩阵与条件数

&emsp;&emsp;在求解方程组时，如果对数据进行较小的扰动，则结果有很大的波动，这样的矩阵称为**病态矩阵** 。病态矩阵是一种特殊矩阵。指条件数很大的非奇异矩阵。病态矩阵的逆和以其为系数矩阵的方程组的界对微小扰动十分敏感，对数值求解会带来很大困难。<font color="#ff0000"><sup>[[1][1]]</sup></font>

例如：现在有线性方程组，$Ax = b$， 解方程：

<center>
$$\begin{bmatrix}
 400& -201 \\\\
 -800& 401
\end{bmatrix}
\begin{bmatrix}
x_{1} \\\\
x_{2}
\end{bmatrix} = 
\begin{bmatrix}
200 \\\\
-200
\end{bmatrix}$$
</center>

很容易得到解为：$x1 = -100, x2 = -200$。如果在样本采集时存在一个微小的误差，比如，将$A$矩阵的系数**400**改变成**401**：则得到一个截然不同的解： $x1 = 40000, x2 = 79800$.

<center>
$$\begin{bmatrix}
 401& -201 \\\\
 -800& 401
\end{bmatrix}
\begin{bmatrix}
x_{1} \\\\
x_{2}
\end{bmatrix} = 
\begin{bmatrix}
200 \\\\
-200
\end{bmatrix}$$
</center>

当解集 $x$ 对 $A$ 和 $b$ 的系数高度敏感，那么这样的方程组就是病态的 (ill-conditioned)。

&emsp;&emsp;**条件数** ：判断矩阵是否病态以及衡量矩阵的病态程度通常看矩阵$A$的条件数$K(A)$的大小：

<center>$K(A)=\frac{\left\|A^{-1}\right\|}{\left\|A \right\|}$</center>

&emsp;&emsp;$K(A)$称为$A$的条件数，很大时，称$A$为病态，否则称良态；$K(A)$俞大，$A$的病态程度俞严重。

### 基于梯度的优化方法

&emsp;&emsp;深度学习算法都涉及到某种形式的优化，优化指改变$x$以最小化或最大化某个函数$f(x)$的任务。


#### Jacobian矩阵和Hessian矩阵

### 约束优化

### 线性最小二乘



[1]:https://www.pianshen.com/article/459993619/