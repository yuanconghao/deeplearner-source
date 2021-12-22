---
title: DeepLearning学习笔记-2.线性代数
date: 2021-12-09 00:38:44
updated: 2021-12-21 00:38:44
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

&emsp;&emsp;具体定义为：
<center>$C_{i,j}=\sum_{k}A_{i,k}B_{k,j}$</center></br>

&emsp;&emsp;矩阵相乘不是对应元素相乘，元素对应相乘又叫Hadamard乘积，记作$A\odot B$。</br>
&emsp;&emsp;向量可看作列为1的矩阵，两个相同维数的向量$x$和$y$的点乘（Dot Product）或者内积，可以表示为$x^{\top }y$。</br>

&emsp;&emsp;矩阵乘积运算满足**分配率**和**结合律**：
<center><font color="#ff0000">$A\left ( B+C \right )=AB+AC$</font></center></br>
<center><font color="#ff0000">$A\left ( BC \right )=\left ( AB \right )C$</font></center></br>

&emsp;&emsp;不满足**交换律**：
<center><font color="#ff0000">$AB=BA$，情况并非总满足</font></center></br>
&emsp;&emsp;乘积的转置：
<center><font color="#ff0000">$\left ( AB \right )^{\top }=B^{\top}A^{\top}$</font></center>

```python
# 矩阵乘法
import numpy as np
A = np.array([[1,2],[3,4]]) # 矩阵A
B = np.array([[5,6],[7,8]]) # 矩阵B

x = np.array([1,2]) # 向量x
y = np.array([3,4]) # 向量y

C = np.dot(A, B) # 矩阵相乘
D = np.multiply(A, B) # 矩阵逐元素相乘，又叫Hadamard乘积，同 A*B

z = np.dot(x, y) # 向量点乘或内积, 同x的转置乘y 

print("矩阵相乘:\n", C)
print("矩阵逐元素相乘:\n", D)
print("向量内积:\n", z)

# output
矩阵相乘:
 [[19 22]
 [43 50]]
矩阵逐元素相乘:
 [[ 5 12]
 [21 32]]
向量内积:
 11
```

### 单位矩阵和逆矩阵

&emsp;&emsp;**单位矩阵**（Identity Matrix）为乘以任意一个向量等于这个向量本身，记为：$I_{n}$，为保持$n$维向量不变的单位矩阵：
<center>$I_{n}\in \mathbb{R}^{n\times n},\forall x\in \mathbb{R}^{n},I_{n}x=x$</center></br>

&emsp;&emsp;单位矩阵结构如：

$$\begin{bmatrix}
1 & 0 & 0\\\\
0 & 1 & 0\\\\
0 & 0 & 1
\end{bmatrix}$$

&emsp;&emsp;**逆矩阵（Inverse Matrix）**：对于$n$阶矩阵$A$，如果有一个$n$阶矩阵$B$，使$AB=BA=I^{n}$，则矩阵$A$可逆，$B$为$A$的**逆矩阵**，$B=A^{-1}$。

&emsp;&emsp;如果$A^{-1}$存在，则线性方程组$Ax=b$的解为：
<center>$A^{-1}Ax=I_{n}x=x=A^{-1}b$</center></br>

* <font color="#ff0000">定理1：若矩阵$A$可逆，则$|A|\neq 0$。</font>
* <font color="#ff0000">定理2：若$|A|\neq 0$，则矩阵$A$可逆，且$A^{-1}=\frac{1}{|A|}A^{\ast }$，$A^{\ast}$为矩阵$A$的**伴随矩阵**。</font>
* <font color="#ff0000">定理3：当$|A|=0$时，称$A$为**奇异矩阵**。</font>
* <font color="#ff0000">定理4：$\left ( \lambda A \right )^{-1}=\frac{1}{\lambda}A^{-1}$</font>
* <font color="#ff0000">定理5：$\left ( AB \right )^{-1}=B^{-1}A^{-1}$</font>

```python
# 单位矩阵
I3 = np.identity(3)    # 单位矩阵
print("单位矩阵:\n",I3)

# 逆矩阵
A = np.array([[1,2,3],[2,2,1],[3,4,3]]) # 矩阵A
A_inv = np.linalg.inv(A)  # 矩阵A的逆矩阵
print("A的逆矩阵:\n", A_inv)

# output:
单位矩阵:
 [[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
A的逆矩阵:
 [[ 1.   3.  -2. ]
 [-1.5 -3.   2.5]
 [ 1.   1.  -1. ]]
```

#### 手动推算
<img src="/images/inverse_matrix.jpg" width="600px"></img>

### 范数

&emsp;&emsp;**范数（norm）**用来衡量向量的大小，向量$\mathbb{L}^{p}$范数定义为：
<center>$\left \| x \right \|_{p}=\left ( \sum_{i}\left | x_{i} \right |^{p} \right )^{\frac{1}{p}},p\in \mathbb{R},p\geqslant 1$</center></br>

&emsp;&emsp;$L^{2}$范数，也称为欧几里得范数（Euclidean norm），是**向量$x$到原点的欧几里得距离**，$L^{2}$范数不一定适用于所有情况，当区别0和非常小但非0值的情况，$L^{1}$范数是一个比较好的选择。
&emsp;&emsp;$L^{1}$范数，在所有方向上的速率是一样的，定义为：
<center>$\left \| x \right \|_{1}=\sum_{i}\left | x_{i} \right |$</center></br>

&emsp;&emsp;<font color="#ff0000">经常用于<b>区分0</b>和<b>非0元素</b>的情形中。</font></br>
&emsp;&emsp;$L^{0}$范数，可用于衡量向量中非0元素的个数，但它并不是一个范数。

&emsp;&emsp;$L^{\infty }$范数，向量元素绝对值的最大值，也叫做（Max norm）：
<center>$\left \| x \right \|_{\infty }=\underset{i}{max}\left | x_{i} \right |$</center></br>
&emsp;&emsp;机器学习中常用的$F$范数（Frobenius norm），定义为：
<center>$\left \| A \right \|_{F}=\sqrt{\sum_{i,j}A_{i,j}^{2}}$</center>

```python
# 范数
a = np.array([1,2,3])  # 向量a
L1 = np.linalg.norm(a, ord=1) # 向量1范数
L2 = np.linalg.norm(a, ord=2) # 向量2范数（欧几里得范数）
Ln = np.linalg.norm(a, ord=np.inf) # 向量无穷范数（最大范数）

A = np.array([[1,2],[3,4]]) # 矩阵A
A_f = np.linalg.norm(A, ord="fro")

print(L1)
print(L2)
print(Ln)
print(A_f)

# output：
6.0
3.7416573867739413
3.0
5.477225575051661
```

#### 手动推算
<img src="/images/norm.jpg" width="400px"></img>


### 特殊类型的矩阵和向量

&emsp;&emsp;**对角矩阵（diagonal matrix）**：只在主对角线含有非零元素，其余位置为零。如：

$$\begin{bmatrix}
1 & 0 & 0\\\\ 
0 & 2 & 0\\\\ 
0 & 0 & 3
\end{bmatrix}$$

并非所有的对角矩阵都是方阵，长方形矩阵也可能为对角矩阵，但没有逆矩阵。

&emsp;&emsp;**对称矩阵（symmetric matrix）**：是转置和自己相等的矩阵。即：

<center>$A=A^{\top}$</center></br>

&emsp;&emsp;**单位向量（unit ventor）**：是具有**单位范数（unit norm）**的向量，即：

<center>$\left \| x_{2} \right \|=1$</center></br>

若$x^{\top}y=0$，则向量$x$和向量$y$互相**正交（orthogonal）**。正交且范数为1，则称为**标准正交**，即：$A^{\top}A=AA^{\top}=I$，$A^{-1}=A^{\top}$。

### 特征分解

&emsp;&emsp;**矩阵分解（eigendecompostion）**是使用最广的矩阵分解之一，即将矩阵分解成一组**特征向量**和**特征值**。

&emsp;&emsp;方阵$A$的**特征向量（eigenvector）**是指与$A$相乘后相当于对该向量进行缩放的非零向量$v$:

<center>$Av=\lambda v$</center></br>

&emsp;&emsp;其中标量$\lambda$称为这个特征向量对应的**特征值（eigenvalue）**。

&emsp;&emsp;如果一个$n\times n$矩阵$A$有$n$组线性无关的单位特征向量$\\{ v^{(1)},...,v^{(n)} \\}$，以及对应的特征值$\lambda _{1},...,\lambda _{n}$。将这些特征向量按列拼接成一个矩阵$V=\left [ v^{(1)},...,v^{(n)} \right]$，并将对应的特征值拼接成一个向量：$\lambda=\left [\lambda _{1},...,\lambda _{n}\right ]$。
&emsp;&emsp;$A$的特征值分解为：
<center>$A=V_{diag}(\lambda)V^{-1}$</center></br>

&emsp;&emsp;所有特征值都是正数的矩阵称为**正定（positive definite）**；所有特征值都是非负数的矩阵称为**半正定（positive semidefinite）**；所有特征值都是负数的矩阵称为**负定（negative definite）**；所有特征值都是非正数的矩阵称为**半负定（negative semidefinite）**；

注意：
* 不是所有的矩阵都有特征分解。
* 在某些情况下，实矩阵的特征值分解可能会得到复矩阵。


```python
# 特征值和特征向量
A=np.array([[-1,1,0],[-4,3,0],[1,0,2]]) # 矩阵A

# 计算特征值
A_eig = np.linalg.eigvals(A)

# 计算特征值和特征向量
A_eig,A_eigvector = np.linalg.eig(A) 

print("特征值：\n", A_eig)
print("特征向量：\n", A_eigvector)

# output:
特征值：
 [2. 1. 1.]
特征向量：
 [[ 0.          0.40824829  0.40824829]
 [ 0.          0.81649658  0.81649658]
 [ 1.         -0.40824829 -0.40824829]]
```

#### 手动推算
<img src="/images/eigvector.jpg" width="600px"></img>


### 奇异值分解

&emsp;&emsp;**奇异值分解（singular value decomposition,SVD）**将矩阵分解为**奇异向量（singular vector）**和**奇异值（singular value）**。与特征值分解相⽐，奇异值分解更加通⽤，所有的实矩阵都可以进⾏奇异值分解，⽽特征值分解只对某些⽅阵可以。

&emsp;&emsp;奇异值分解的形式为：
<center>$A=UDV^{\top}$</center></br>

&emsp;&emsp;若$A$是 $m \times n$ 的，那么 $U$ 是 $m \times m$ 的，其列向量称为左奇异向量，⽽ $V$ 是 $n \times n$ 的，其列向量称为右奇异向量，⽽ $D$ 是 $m \times n$ 的⼀个对⾓矩阵，其对⾓元素称为矩阵 $A$ 的奇异值。左奇异向量是 $AA^{\top}$ 的特征向量，⽽右奇异向量是 $A^{\top}A$ 的特征向量，⾮ 0 奇异值的平⽅是 $A^{\top}A$ 的⾮ 0 特征值。

```python
# 奇异值分解
A = np.array([[0,1],[1,1],[1,0]]) # 矩阵A

U,D,V = np.linalg.svd(A)

print("矩阵U:\n", U)
print("矩阵D:\n", D)
print("矩阵V:\n", V)

# output:
矩阵U:
 [[-4.08248290e-01  7.07106781e-01  5.77350269e-01]
 [-8.16496581e-01  2.64811510e-17 -5.77350269e-01]
 [-4.08248290e-01 -7.07106781e-01  5.77350269e-01]]
矩阵D:
 [1.73205081 1.        ]
矩阵V:
 [[-0.70710678 -0.70710678]
 [-0.70710678  0.70710678]]
```

#### 手动推算

<img src="/images/svd1.jpg" width="600px"></img>
<img src="/images/svd2.jpg" width="600px"></img>

### 迹运算

&emsp;&emsp;**迹运算**返回的是矩阵对角元素的和：

<center>$Tr(A)=\sum_{i}A_{i,j}$</center></br>

### 行列式

&emsp;&emsp;**行列式**，$det(A)$是将一个方阵$A$映射到实数的函数。行列式等于矩阵特征值的乘积。行列式的绝对值可以用来衡量矩阵参与矩阵乘法后空间扩大或者缩小了多少。

### 主成分分析

&emsp;&emsp;花书中对于PCA的讲解不是很清晰，理论性很强，很难根据书中公式进行推导。于是查阅相关资料，总结出一套清晰的推导步骤：

#### 背景与作用
&emsp;&emsp;在研究与应用中，需要对收集的大量数据进行分析，随着数据集变量增多，且变量间可能存在相关性，便加大了问题分析的复杂性，如果对每个指标进行分析，往往分析是孤立的，不能完全利用数据中的信息，而盲目的减少指标会损失很多有用的信息，产生错误的结论。
&emsp;&emsp;因此，需要找到一种合理的方法，对指标进行降维，既要减少分析的指标，又要达到对所收集的数据进行全面分析的目的。降维算法如：**奇异值分解(SVD)** 、**主成分分析(PCA)** 、**因子分析(FA)** 、**独立成分分析(ICA)**。
&emsp;&emsp;降维是一种对高维度特征数据预处理方法。是将高维度的数据保留下最重要的一些特征，去除噪声和不重要特征，实现提升数据处理速度的目的。在实际生产和应用中，降维在一定的信息损失范围内，可帮助我们节省大量时间和成本。降维的优点：

* 使数据集更易使用
* 降低算法的计算开销
* 去除噪声
* 使得结果容易理解

#### PCA概念
&emsp;&emsp;PCA(Principal Component Analysis)，即主成分分析方法，是一种使用最广泛的数据降维算法。PCA的主要思想是将n维特征映射到k维上，这k维是全新的正交特征也被称为主成分，是在原有n维特征的基础上重新构造出来的k维特征。PCA的工作就是从原始的空间中顺序地找一组相互正交的坐标轴，新的坐标轴的选择与数据本身是密切相关的。其中，第一个新坐标轴选择是原始数据中方差最大的方向，第二个新坐标轴选取是与第一个坐标轴正交的平面中使得方差最大的，第三个轴是与第1,2个轴正交的平面中方差最大的。依次类推，可以得到n个这样的坐标轴。通过这种方式获得的新的坐标轴，我们发现，大部分方差都包含在前面k个坐标轴中，后面的坐标轴所含的方差几乎为0。于是，我们可以忽略余下的坐标轴，只保留前面k个含有绝大部分方差的坐标轴。事实上，这相当于只保留包含绝大部分方差的维度特征，而忽略包含方差几乎为0的特征维度，实现对数据特征的降维处理。

#### 






