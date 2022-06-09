---
title: Python-2.数据分析numpy使用
date: 2022-05-19 11:54:47
toc: true
mathjax: true
categories:
    - Python

tags:
    - Python
    - 数据分析
    - numpy
---

[Numpy][1]（Numerical Python）官方定义：The fundamental package for scientific computing with Python. 

Numpy是python的科学计算库，在矩阵乘法与数组性状处理上，Numpy有很好的性能，处理速度快。优点总结如下：
* Python科学计算基础库
* 可对数组进行高效的数学运算
* ndarray对象可以用来构建多维数组
* 能够执行傅里叶变换与重塑多维数组性状
* 提供了线性代数，以及随机数生成的内置函数

<!--more-->

### NumPy数据类型

| [数据类型][3] | 描述 |
| --- | --- |
| bool_ | 布尔型数据类型（True 或者 False）|
| int_ | 默认整数类型，类似于C语言中的long，取值为int32或int64 |
| int8/int16/int32/int64 | 代表1个字节/2个字节/4个字节/8个字节的整数 |
| uint8/uint16/uint32/uint64 | 代表1个字节/2个字节/4个字节/8个字节的无符号整数 |
| float_ | float64 类型的简写 |
| float16 | 半精度浮点数，包括：1 个符号位，5 个指数位，10个尾数位 |
| float32 | 单精度浮点数，包括：1 个符号位，8 个指数位，23个尾数位 |
| float64 | 双精度浮点数，包括：1 个符号位，11 个指数位，52个尾数位 |
| complex_ | 复数类型，与complex128类型相同 |
| complex64 | 表示实部和虚部共享32位的复数 |
| complex128 | 表示实部和虚部共享64位的复数 |
| str_/string_ | 表示字符串类型 |


### 子类化ndarray
&emsp;&emsp;ndarray对象采用数组的索引机制，将数组中的每个元素映射到内存上，并按照一定布局对内存块进行排列。通过NumPy的内置函数array()可以创建ndarray对象，语法格式如下：

```python
'''
@array
object    必选，表示一个数组序列
dtype     可选，可更改数组的数据类型
copy      可选，数组能否被复制，默认True
order     可选，以哪种内存布局创建数组，可选值C(行序列)/F(列序列)/A(默认)
ndmin     可选，指定数组维度
'''
numpy.array(object, dtype=None, copy=True, order=None, ndmin=0)

# 数组属性
'''
shape     返回值一个由数组维度构成的元组，如(2,3) 表示2行3列二维数组
'''
ndarray.shape()    

'''
reshape   调整数组形状
'''
ndarray.reshape()    

'''
ndmin     返回数组维数
'''
ndarray.ndmin

'''
itemsize  返回数组每个元素的大小，单位字节
'''
ndarray.itemsize

'''
flags     返回ndarray数组的内存信息
'''
ndarray.flags
```

```python
# import numpy package
import numpy as np

# 使用列表构建一维数组
a = np.array([1, 2, 3])
# 使用列表构建二维数组
b = np.array([[1, 2, 3], [4, 5, 6]])
# 使用dtype改变数组元素数据类型
c = np.array([1, 0, 5], dtype=bool)
# ndmin维度改变：一维改为二维
d = np.array([1, 2, 3,4,5], ndmin = 2)
# reshape数组变维，将数组2行3列转换为3行2列
e = b.reshape(3, 2)

# 打印
print("----------------")
print("a:", a)
print("----------------")
print("b:", b)
print("----------------")
print("c:", c)
print("----------------")
print("d:", d)
print("----------------")
print("e:", e)
print("----------------")

# output
'''
----------------
a: [1 2 3]
----------------
b: [[1 2 3]
 [4 5 6]]
----------------
c: [ True False  True]
----------------
d: [[1 2 3 4 5]]
----------------
e: [[1 2]
 [3 4]
 [5 6]]
----------------
'''
```

### NumPy创建数组
&emsp;&emsp;创建数组有5种常用方法：
1. Python其他结构（列表，元组）转换
2. numpy原生数组的创建（arange、ones、zeros等）
3. 磁盘读取，标准格式或自定义格式
4. 使用字符串或缓冲区从原始字节创建数组
5. 使用特殊库函数，如random

#### 原生数组的创建

```python
'''
@array
object    必选，表示一个数组序列
dtype     可选，可更改数组的数据类型
copy      可选，数组能否被复制，默认True
order     可选，以哪种内存布局创建数组，可选值C(行序列)/F(列序列)/A(默认)
ndmin     可选，指定数组维度
'''
numpy.array(object, dtype=None, copy=True, order=None, ndmin=0)

'''
@array
start   起始值，默认是 0。
stop    终止值，注意生成的数组元素值不包含终止值。
step    步长，默认为 1。
dtype   可选参数，指定 ndarray 数组的数据类型。
'''
numpy.arange(start, stop, step, dtype)

'''
@linspace
start       数值区间的起始值；
stop        数值区间的终止值；
num         数值区间内要生成多少个均匀的样本。默认值为 50；
endpoint    默认为 True，表示数列包含 stop 终止值，反之不包含；
restep      默认为 True，表示生成的数组中会显示公差项，反之不显示；
dtype       数组元素值的数据类型。
'''
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)

'''
@logspace
start       序列的起始值：base**start。
stop        序列的终止值：base**stop。
num         数值范围区间内样本数量，默认为50。
endpoint    默认为True包含终止值，反之不包含。
base        对数函数的 log 底数，默认为10。
dtype       数组元素值的数据类型。
'''
np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)

'''
@empty
shape   数组的形状
dtype   数组元素的数据类型，默认值为float
order   数组元素在内存的顺序，默认C，行优先
'''
numpy.empty(shape, dtype = float, order = 'C')

'''
@zeros
shape   数组的形状大小
dtype   可选，数组元素的数据类型
order   数组元素在内存的顺序，默认C，行优先
'''
numpy. zeros(shape,dtype=float,order="C")

'''
@ones
shape   数组的形状大小
dtype   可选，数组元素的数据类型
order   数组元素在内存的顺序，默认C，行优先
'''
numpy.ones(shape, dtype = None, order = 'C')

'''
@asarray
sequence    接受一个 Python 序列，可以是列表或者元组
dtype       可选，数组元素的数据类型
order       数组元素在内存的顺序，默认C，行优先
'''
numpy.asarray(sequence，dtype = None ，order = None)

'''
@frombuffer
buffer    将任意对象转换为流的形式读入缓冲区；
dtype     返回数组的数据类型，默认是float32；
count     要读取的数据数量，默认为-1表示读取所有数据；
offset    读取数据的起始位置，默认为0。
'''
numpy.frombuffer(buffer, dtype = float, count = -1, offset = 0)

'''
@fromiter
iterable  可迭代对象。
dtype     返回数组的数据类型
count     读取的数据数量，默认为-1，读取所有数据。
'''
numpy.fromiter(iterable, dtype, count = -1)
```

```python


```

### Numpy索引和切片

### Numpy数据相关操作

#### 数组变维

#### 数组转置

#### 数组连接

#### 数组分割

#### 数组元素增删改查


### Numpy常用统计方法

#### 数学函数

#### 算术运算

#### 统计函数

#### 矩阵运算

#### Matrix矩阵库

#### 线性代数

#### 矩阵乘法


### Numpy输入输出(IO)



[1]:https://numpy.org/
[2]:https://www.numpy.org.cn/user/basics/types.html#%E6%95%B0%E7%BB%84%E7%B1%BB%E5%9E%8B%E4%B9%8B%E9%97%B4%E7%9A%84%E8%BD%AC%E6%8D%A2
[3]:https://www.numpy.org.cn/user/basics/types.html#%E6%95%B0%E7%BB%84%E7%B1%BB%E5%9E%8B%E4%B9%8B%E9%97%B4%E7%9A%84%E8%BD%AC%E6%8D%A2