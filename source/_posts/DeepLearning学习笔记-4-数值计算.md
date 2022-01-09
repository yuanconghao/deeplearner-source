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

&emsp;&emsp;深度学习算法都涉及到某种形式的优化，优化指改变$x$以最小化或最大化某个函数$f(x)$的任务。**目标函数（Objective Function）** ：把要最小化或最大化的函数称为目标函数。对其进行最小化时，称为**损失函数（Loss Function）** 或 **误差函数（Error Function）**。梯度下降时无约束优化最常用的方法之一，另一种为最小二乘法。

&emsp;&emsp;**梯度下降（Gradient Descent）** <font color="#ff0000"><sup>[[2][2]]</sup></font><font color="#ff0000"><sup>[[3][3]]</sup></font>： 梯度下降简单来说就是一种寻找目标函数最小化的方法，**导数**对于最小化一个函数很有用，代表更改$x$来略微改善$y$，因此可以将$x$往导数的反方向移动一小步来减小$f(x)$，这种技术称为梯度下降。如图所示，梯度下降算法，沿着函数的下坡方向（导数反方向）直到最小。

<img src="/images/gradient_descent.png"></img>

* 对于$x>0$，有$f'(x)>0$，左移来减小$f$。
* 对于$x<0$，有$f'(x)<0$，右移来减小$f$。
* 当$f'(x)=0$时，导数无法提供往哪个方向移动，$f'(x)=0$的点称为**临界点（Critical Point）** 。

&emsp;&emsp;临界点分局部极小点、局部极大点、鞍点。当存在多个局部极小点或平坦区域时，优化算法可能无法找到**全局最小点**，在深度学习背景下，即使找到的解不是真正最小的，但只要对应于代价函数显著低的值，通常可以接受这样的解。

&emsp;&emsp;针对多维输入的函数，需要用到**偏导数（Partial Detrivatice）**，**梯度（Gradient）**是相对于一个向量求导的导数：$f$的梯度是包含所有偏导数的向量，记为$\bigtriangledown_{x}f(x)$。在多维情况下，临界点是梯度中所有元素都为零点。

&emsp;&emsp;梯度下降建议新的点为:
<center>$x'=x-\epsilon \bigtriangledown_{x}f(x)$</center>

&emsp;&emsp;其中$\epsilon$为**学习率（Learning Rate）**，是一个确定步长大小的正标量。$\epsilon$的选择方法：选择一个小常数；根据几个$\epsilon$计算$x'$，选择能产生最小目标函数值的$\epsilon$，称为在线搜索。

#### 代码实现

```python
# 上图中函数公式 f(x)=0.5x^2 梯度下降算法演示
def f(x): # f(x)
    return 0.5 * x**2

def df(x): # f(x)导数
    return x

def gradient_descent(x, epsilon, iteration):
    iter_num = 0
    f_change = f_current = f(x)
    while(iter_num < iteration and f_change > 1e-10):
        iter_num += 1
        x = x - epsilon * df(x)
        f_new = f(x)
        f_change = abs(f_current - f_new)
        f_current = f_new
        GD_X.append(x)
        GD_Y.append(f_new)
        print("第%d次迭代：x=%f,f(x)=%f,df(x)=%f" %(iter_num,x,f_new,df(x)))
    return x

x = 2 # 初始点
epsilon = 0.1 # 学习率
GD_X = []
GD_Y = []
x_g = gradient_descent(x, epsilon, 100)
print("最终优化参数:%.10f" %x_g)

# output:
第1次迭代：x=1.800000,f(x)=1.620000,df(x)=1.800000
第2次迭代：x=1.620000,f(x)=1.312200,df(x)=1.620000
第3次迭代：x=1.458000,f(x)=1.062882,df(x)=1.458000
第4次迭代：x=1.312200,f(x)=0.860934,df(x)=1.312200
第5次迭代：x=1.180980,f(x)=0.697357,df(x)=1.180980
第6次迭代：x=1.062882,f(x)=0.564859,df(x)=1.062882
第7次迭代：x=0.956594,f(x)=0.457536,df(x)=0.956594
第8次迭代：x=0.860934,f(x)=0.370604,df(x)=0.860934
第9次迭代：x=0.774841,f(x)=0.300189,df(x)=0.774841
第10次迭代：x=0.697357,f(x)=0.243153,df(x)=0.697357
第11次迭代：x=0.627621,f(x)=0.196954,df(x)=0.627621
第12次迭代：x=0.564859,f(x)=0.159533,df(x)=0.564859
第13次迭代：x=0.508373,f(x)=0.129222,df(x)=0.508373
第14次迭代：x=0.457536,f(x)=0.104670,df(x)=0.457536
第15次迭代：x=0.411782,f(x)=0.084782,df(x)=0.411782
第16次迭代：x=0.370604,f(x)=0.068674,df(x)=0.370604
第17次迭代：x=0.333544,f(x)=0.055626,df(x)=0.333544
第18次迭代：x=0.300189,f(x)=0.045057,df(x)=0.300189
第19次迭代：x=0.270170,f(x)=0.036496,df(x)=0.270170
第20次迭代：x=0.243153,f(x)=0.029562,df(x)=0.243153
第21次迭代：x=0.218838,f(x)=0.023945,df(x)=0.218838
第22次迭代：x=0.196954,f(x)=0.019395,df(x)=0.196954
第23次迭代：x=0.177259,f(x)=0.015710,df(x)=0.177259
第24次迭代：x=0.159533,f(x)=0.012725,df(x)=0.159533
第25次迭代：x=0.143580,f(x)=0.010308,df(x)=0.143580
第26次迭代：x=0.129222,f(x)=0.008349,df(x)=0.129222
第27次迭代：x=0.116299,f(x)=0.006763,df(x)=0.116299
第28次迭代：x=0.104670,f(x)=0.005478,df(x)=0.104670
第29次迭代：x=0.094203,f(x)=0.004437,df(x)=0.094203
第30次迭代：x=0.084782,f(x)=0.003594,df(x)=0.084782
第31次迭代：x=0.076304,f(x)=0.002911,df(x)=0.076304
第32次迭代：x=0.068674,f(x)=0.002358,df(x)=0.068674
第33次迭代：x=0.061806,f(x)=0.001910,df(x)=0.061806
第34次迭代：x=0.055626,f(x)=0.001547,df(x)=0.055626
第35次迭代：x=0.050063,f(x)=0.001253,df(x)=0.050063
第36次迭代：x=0.045057,f(x)=0.001015,df(x)=0.045057
第37次迭代：x=0.040551,f(x)=0.000822,df(x)=0.040551
第38次迭代：x=0.036496,f(x)=0.000666,df(x)=0.036496
第39次迭代：x=0.032846,f(x)=0.000539,df(x)=0.032846
第40次迭代：x=0.029562,f(x)=0.000437,df(x)=0.029562
第41次迭代：x=0.026606,f(x)=0.000354,df(x)=0.026606
第42次迭代：x=0.023945,f(x)=0.000287,df(x)=0.023945
第43次迭代：x=0.021551,f(x)=0.000232,df(x)=0.021551
第44次迭代：x=0.019395,f(x)=0.000188,df(x)=0.019395
第45次迭代：x=0.017456,f(x)=0.000152,df(x)=0.017456
第46次迭代：x=0.015710,f(x)=0.000123,df(x)=0.015710
第47次迭代：x=0.014139,f(x)=0.000100,df(x)=0.014139
第48次迭代：x=0.012725,f(x)=0.000081,df(x)=0.012725
第49次迭代：x=0.011453,f(x)=0.000066,df(x)=0.011453
第50次迭代：x=0.010308,f(x)=0.000053,df(x)=0.010308
第51次迭代：x=0.009277,f(x)=0.000043,df(x)=0.009277
第52次迭代：x=0.008349,f(x)=0.000035,df(x)=0.008349
第53次迭代：x=0.007514,f(x)=0.000028,df(x)=0.007514
第54次迭代：x=0.006763,f(x)=0.000023,df(x)=0.006763
第55次迭代：x=0.006087,f(x)=0.000019,df(x)=0.006087
第56次迭代：x=0.005478,f(x)=0.000015,df(x)=0.005478
第57次迭代：x=0.004930,f(x)=0.000012,df(x)=0.004930
第58次迭代：x=0.004437,f(x)=0.000010,df(x)=0.004437
第59次迭代：x=0.003993,f(x)=0.000008,df(x)=0.003993
第60次迭代：x=0.003594,f(x)=0.000006,df(x)=0.003594
第61次迭代：x=0.003235,f(x)=0.000005,df(x)=0.003235
第62次迭代：x=0.002911,f(x)=0.000004,df(x)=0.002911
第63次迭代：x=0.002620,f(x)=0.000003,df(x)=0.002620
第64次迭代：x=0.002358,f(x)=0.000003,df(x)=0.002358
第65次迭代：x=0.002122,f(x)=0.000002,df(x)=0.002122
第66次迭代：x=0.001910,f(x)=0.000002,df(x)=0.001910
第67次迭代：x=0.001719,f(x)=0.000001,df(x)=0.001719
第68次迭代：x=0.001547,f(x)=0.000001,df(x)=0.001547
第69次迭代：x=0.001392,f(x)=0.000001,df(x)=0.001392
第70次迭代：x=0.001253,f(x)=0.000001,df(x)=0.001253
第71次迭代：x=0.001128,f(x)=0.000001,df(x)=0.001128
第72次迭代：x=0.001015,f(x)=0.000001,df(x)=0.001015
第73次迭代：x=0.000914,f(x)=0.000000,df(x)=0.000914
第74次迭代：x=0.000822,f(x)=0.000000,df(x)=0.000822
第75次迭代：x=0.000740,f(x)=0.000000,df(x)=0.000740
第76次迭代：x=0.000666,f(x)=0.000000,df(x)=0.000666
第77次迭代：x=0.000599,f(x)=0.000000,df(x)=0.000599
第78次迭代：x=0.000539,f(x)=0.000000,df(x)=0.000539
第79次迭代：x=0.000485,f(x)=0.000000,df(x)=0.000485
第80次迭代：x=0.000437,f(x)=0.000000,df(x)=0.000437
第81次迭代：x=0.000393,f(x)=0.000000,df(x)=0.000393
第82次迭代：x=0.000354,f(x)=0.000000,df(x)=0.000354
第83次迭代：x=0.000319,f(x)=0.000000,df(x)=0.000319
第84次迭代：x=0.000287,f(x)=0.000000,df(x)=0.000287
第85次迭代：x=0.000258,f(x)=0.000000,df(x)=0.000258
第86次迭代：x=0.000232,f(x)=0.000000,df(x)=0.000232
第87次迭代：x=0.000209,f(x)=0.000000,df(x)=0.000209
第88次迭代：x=0.000188,f(x)=0.000000,df(x)=0.000188
第89次迭代：x=0.000169,f(x)=0.000000,df(x)=0.000169
第90次迭代：x=0.000152,f(x)=0.000000,df(x)=0.000152
第91次迭代：x=0.000137,f(x)=0.000000,df(x)=0.000137
第92次迭代：x=0.000123,f(x)=0.000000,df(x)=0.000123
第93次迭代：x=0.000111,f(x)=0.000000,df(x)=0.000111
第94次迭代：x=0.000100,f(x)=0.000000,df(x)=0.000100
第95次迭代：x=0.000090,f(x)=0.000000,df(x)=0.000090
第96次迭代：x=0.000081,f(x)=0.000000,df(x)=0.000081
第97次迭代：x=0.000073,f(x)=0.000000,df(x)=0.000073
第98次迭代：x=0.000066,f(x)=0.000000,df(x)=0.000066
第99次迭代：x=0.000059,f(x)=0.000000,df(x)=0.000059
第100次迭代：x=0.000053,f(x)=0.000000,df(x)=0.000053
最终优化参数:0.0000531228

# 最终优化参数x并非真实的0,而是无限逼近0，这是梯度下降缺点之一，没法非常精确，但在可接受范围内。
```

```python
# 可视化梯度下降过程
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(-2, 2, 0.05)
Y = f(X)
Y = np.array(Y)
plt.plot(X,Y)
plt.scatter(GD_X, GD_Y)
plt.title("$f(x) = 0.5x^2$")
plt.show()
```
<img src="/images/gradient_descent1.png"></img>

```python
# 调整学习率为1.5
plt.plot(GD_X, GD_Y)
plt.title("$f(x) = 0.5x^2$")
plt.show()
```
<img src="/images/gradient_descent2.png"></img>

#### 手动推算
<img src="/images/gradient_descent3.png" width="800px"></img>

#### Jacobian矩阵和Hessian矩阵

&emsp;&emsp;**Jacobian矩阵** ：有时我们需要计算输入和输出都为向量的函数的所有偏导数，包含所有这样的偏导数的矩阵称为Jacobian矩阵。定义：有一个函数$f:\mathbb{R}^{m}\rightarrow \mathbb{R}^{n}$，$f$的Jacobian矩阵$J\in \mathbb{R}^{n\times m}$，定义为$J_{i,j}=\frac{\partial }{\partial x_{j}}f(x)_{i}$。

&emsp;&emsp;**Hessian矩阵**：当函数具有多维输入时，二阶导数有很多。将这些导数合并为一个矩阵，称为Hessian矩阵，定义为：
<center>$H(f)(x)_{i,j}=\frac{\partial ^{2}}{\partial x_{i}\partial x_{j}}f(x)$</center>

&emsp;&emsp;Hessian矩阵等价于梯度的Jacobian矩阵。

### 牛顿法

&emsp;&emsp;**牛顿法（Newton's Method）**：基于宇哥二阶泰勒展开来近似$x^{(0)}$附近的$f(x)$：
<center>$f(x)\approx f(x^{(0)})+(x-x^{(0)})^{\top}\bigtriangledown _{x}f(x^{(0)})+\frac{1}{2}(x-x^{(0)})^{\top }H(f)(x^{(0)})(x-x^{(0)})$</center>

&emsp;&emsp;接着通过计算，可以得到这个函数的临界点：
<center>$x^{*}=x^{(0)}-H(f)(x^{(0)})^{-1}\bigtriangledown _{x}f(x^{(0)})$</center>

&emsp;&emsp;牛顿法迭代更新近似函数和跳到近似函数的最小点可以比梯度下降法更快地到达临界点。这在接近全局极小时是一个特别有用的性质，但在鞍点附近是有害的。
&emsp;&emsp;针对上述实例，计算得到：$H=A^{\top}A$，进一步计算得到最优解：
<center>$x^{*}=x^{(0)}-(A^{\top}A)^{-1}(A^{\top}Ax^{(0)}-A^{\top}b)=(A^{\top}A)^{-1}A^{\top}b$</center>

#### 手动推算

### 约束优化
&emsp;&emsp;**约束优化（Constrained Optimization）** : $x$在某些集合$S$中找$f(x)$的最大值和最小值，而非在所有值下的最大和最小值，这称为约束优化。
&emsp;&emsp;通过$m$个函数$g^{(i)}$和$n$个函数$h^{(j)}$描述$S$，那么$S$可以表示为$S=\left\\{x\mid \forall i,g^{(i)}(x)=0 and \forall j,h^{(j)}(x)\leqslant 0 \right\\}$。其中涉及$g^{(i)}$的等式称为等式约束，涉及$h^{(j)}$的不等式称为不等式约束。

&emsp;&emsp;为每个约束引入新的变量$\lambda _{i}$和$\alpha _{j}$，这些新变量被称为KKT乘子。广义拉格朗日式定义：
<center>$L(x,\lambda ,\alpha )=f(x)+\sum_{i}\lambda _{i}g^{(i)}(x)+\sum_{j}\lambda _{j}h^{(j)}(x)$</center>

&emsp;&emsp;可以通过优化无约束的广义拉格朗日式解决约束最小化问题：
<center>$\underset{x}{min}\ \underset{\lambda}{max}\ \underset{\alpha ,\alpha \geqslant 0}{max}L(x,\lambda,\alpha)$</center>

&emsp;&emsp;优化该式与下式等价：
<center>$\underset{m\in S}{min}\ f(x)$</center>

&emsp;&emsp;针对上述实例，约束优化：$x^{\top}x\leqslant 1$

&emsp;&emsp;引入广义拉格朗日式：
<center>$L(x,\lambda)=f(x)+\lambda (x^{\top}x-1)$</center>

&emsp;&emsp;解决以下问题：
<center>$\underset{x}{min}\ \underset{\lambda,\lambda \geqslant 0}{max}L(x,\lambda)$</center>

&emsp;&emsp;关于$x$对于Lagrangian微分，我们得到方程：
<center>$A^{\top}Ax-A^{\top}b+2\lambda x=0$</center>

&emsp;&emsp;得到解的形式：
<center>$x=(A^{\top}A+2\lambda I)^{-1}A^{\top}b$</center>

&emsp;&emsp;$\lambda$的选择必须使结果服从约束，可以对$\lambda$梯度上升找到这个值：
<center>$\frac{\partial }{\partial \lambda}L(x,\lambda)=x^{\top}x-1$</center>

#### 手动推算

### 线性最小二乘

&emsp;&emsp;**最小二乘法**：<font color="#ff0000"><sup>[[4][4]]</sup></font><font color="#ff0000"><sup>[[5][5]]</sup></font><font color="#ff0000"><sup>[[6][6]]</sup></font>用来做函数拟合或者求函数极值的方法，在机器学习中，在回归模型中较为常见。

&emsp;&emsp;例如：引入实例：
<center>$f(x)=\frac{1}{2}\left\| Ax-b\right\|^{2}_{2}$</center>

&emsp;&emsp;假设我们希望找到最⼩化该式的$x$值，计算梯度得到：
<center>$\bigtriangledown _{x}f(x)=A^{\top}(Ax-b)=A^{\top}Ax-A^{\top}b$</center>

#### 手动推算

#### 代码实现

```python
import numpy as np
import numpy.linalg as la

def matmul_chain(*args):
    if len(args) == 0: return np.nan
    result = args[0]
    for x in args[1:]:
        result = result@x
    return result

"""
牛顿法
"""
def newton(x, A, b, delta):
    x = matmul_chain(np.linalg.inv(matmul_chain(A.T, A)), A.T, b)
    return x

"""
梯度下降法
"""
def gradient_decent(x, A, b, epsilon, delta):
    while la.norm(matmul_chain(A.T, A, x) - matmul_chain(A.T, b)) > delta:
        x -= epsilon*(matmul_chain(A.T, A, x) - matmul_chain(A.T, b))
    return x

"""
约束优化
"""
def constrain_opti(x, A, b, delta):
    k = len(x)
    lamb = 0
    # delta 设为 5e-2，最优设为 0 x = matmul_chain(np.linalg.inv(matmul_chain(A.T, A)+2*lamb*np.identity(k)), A.T, b)
    while np.abs(np.dot(x.T, x) - 1) > 5e-2: 
        lamb += np.dot(x.T, x) - 1
    return x


x0 = np.array([1.0, 1.0, 1.0])
A = np.array([[1.0, -2.0, 1.0], [0.0, 2.0, -8.0], [-4.0, 5.0, 9.0]])
b = np.array([0.0, 8.0, -9.0])
epsilon = 0.01
delta = 1e-3

print("牛顿法：", newton(x0, A, b, delta))
print("梯度下降法：", gradient_decent(x0, A, b, epsilon, delta))
print("约束优化：", constrain_opti(x0, A, b, delta))

# output:
牛顿法： [29. 16.  3.]
梯度下降法： [27.82277014, 15.34731055, 2.83848939]
约束优化： [0.23637902, 0.05135858, -0.94463626]
```


[1]:https://www.pianshen.com/article/459993619/
[2]:https://www.cnblogs.com/pinard/p/5970503.html
[3]:https://www.zhihu.com/question/305638940
[4]:https://www.pianshen.com/article/2053395762
[5]:https://zhuanlan.zhihu.com/p/140377384
[6]:https://zhuanlan.zhihu.com/p/38128785