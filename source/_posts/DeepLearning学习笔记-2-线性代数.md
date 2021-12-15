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

{% codeblock Javascript Array Syntax lang:js http://j.mp/pPUUmW MDN Documentation %}
var arr1 = new Array(arrayLength);
var arr2 = new Array(element0, element1, ..., elementN);
{% endcodeblock %}

#### 张量（tensor）

&emsp;&emsp;**超过二维的数组**，表示同矩阵，如：**A**表示三维张量，$A_{i,j,k}$表示其元素。



* 转置（transpose）：

### 矩阵和向量相乘

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
