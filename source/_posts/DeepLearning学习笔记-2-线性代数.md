---
title: DeepLearning学习笔记-2.线性代数
date: 2021-12-09 00:38:44
toc: true
mathjax: true
categories:
    - 深度学习

tags:
    - 深度学习笔记
    - 花书
---

线性代数，主要面向连续数学，而非离散数学。

<!--more-->

### 标量、向量、矩阵和张量

#### 标量（scalar）

&emsp;&emsp;表示一个**单独的数**，通常用*斜体*小写字母表示，如：$\mathit{s}\in \mathbb{R},\mathit{n}\in \mathbb{N}$。

#### 向量（vector）
&emsp;&emsp;表示**一列数**，这些数是有序列的，并且可以通过下标索引获取对应值，通常用**粗体**小写字母表示，如：$\mathbf{x}\in \mathbb{R}^{n}$，表示元素取实数，且有$n$个元素，第一个元素：$x_{1}$，第n个元素：$x_{n}$。向量写成列形式：

$$\begin{bmatrix}
x_{1}\\\\
x_{2}\\\\
...\\\\
x_{n}
\end{bmatrix}$$

#### 矩阵（matrix）

&emsp;&emsp;表示一个**二维数组**，每个元素下标由两个数字确定，通常用**大写粗体**字母表示，如：$\mathbf{x}\in \mathbb{R}^{m\times n}$，表示元素取实数的$m$行$n$列矩阵，有$m\times n$个元素，可表示为$A_{1,1},A_{m,n}$，$A_{i:}$表示为第$i$行，$A_{:j}$表示为第$j$列。矩阵形式：

$$\begin{bmatrix}
A_{1,1} & A_{1,2} \\\\
A_{2,1} & A_{2,2}
\end{bmatrix}$$

&emsp;&emsp;矩阵逐元素操作：将函数$f$应用到**A**的所有元素上，用$f(A)_{i,j}$表示。

#### 张量（tensor）

&emsp;&emsp;**超过二维的数组**，表示同矩阵，如：**A**表示三维张量，$A_{i,j,k}$表示其元素。

#### 转置（transpose）

&emsp;&emsp;矩阵转置相当于**沿着对角线翻转**，定义如下：$A_{i,j}^{\top } = A_{j,i}$。
* <font color="#ff0000">矩阵转置的转置等于矩阵本身：$\left ( A^{\top} \right )^{\top} = A$</font>
* <font color="#ff0000">$(A+B)^{\top} = A^{\top} + B^{\top}$</font>
* <font color="#ff0000">$(\lambda A)^{\top} = \lambda A^{\top}$</font>
* <font color="#ff0000">$(AB)^{\top} = B^{\top}A^{\top}$</font>

```python
# 矩阵转置
import numpy as np

A = np.array([[1,2],[1,0],[2,3]])  # 矩阵A
A_t = A.transpose() # 矩阵转置
A_t_t = A.transpose().transpose() # 矩阵转置的转置

print("A:\n", A)
print("A的转置:\n", A_t)
print("A转置的转置:\n", A_t_t)

# output:
A:
 [[1 2]
 [1 0]
 [2 3]]
A的转置:
 [[1 1 2]
 [2 0 3]]
A转置的转置:
 [[1 2]
 [1 0]
 [2 3]]
```

#### 矩阵加法

&emsp;&emsp;加法即**对应元素相加**，要求两个矩阵形状一样：</br>
<center>$C = A + B,C_{i,j} = A_{i,j} + B_{i,j}$</center></br>

&emsp;&emsp;**数乘**即一个**标量**与**矩阵每个元素相乘**：</br>
<center>$D=a\cdot B+c,D_{i,j}=a\cdot B_{i,j}+c$</center></br>

&emsp;&emsp;广播是矩阵和向量相加，得到一个矩阵，将$b$加到了$A$的每一行上，本质上是构造了一个将$b$按行复制的一个新矩阵。</br>
<center>$C=A+b,C_{i,j}=A_{i,j}+b_{j}$</center>

```python
# 矩阵加法
import numpy as np 
a = 2     # 标量a
b = np.array([1,1]) # 向量b
A = np.array([[1,2],[3,4]]) # 矩阵A
B = np.array([[5,6],[7,8]]) # 矩阵B

C = A + B # 矩阵相加
D = a * B # 数乘
E = A + b # 广播

print("矩阵相加:\n", C)
print("数乘:\n", D)
print("广播:\n", E)

# output:
矩阵相加:
 [[ 6  8]
 [10 12]]
数乘:
 [[10 12]
 [14 16]]
广播:
 [[2 3]
 [4 5]]
```

### 矩阵和向量相乘

&emsp;&emsp;两个矩阵相乘得到第三个矩阵，$A_{m\times n},B_{n\times p}$，相乘得到矩阵$C_{m\times p}$：
<center>$C=AB$</center></br>
<center>$$</center></br>
矩阵相乘不是对应元素相乘，元素对应相乘又叫Hadamard乘积，记作$$。

### 单位矩阵和逆矩阵

### 线性相关和生成子空间

### 范数

### 特殊类型的矩阵和向量

### 特征分解

### 奇异值分解

### Moore-Penrose伪逆

### 迹运算

### 行列式

### 主成分分析
