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
@array  创建给定数值范围的数组
start   起始值，默认是 0。
stop    终止值，注意生成的数组元素值不包含终止值。
step    步长，默认为 1。
dtype   可选参数，指定 ndarray 数组的数据类型。
'''
numpy.arange(start, stop, step, dtype)

'''
@linspace   在指定的数值区间内，返回均匀间隔的一维等差数组，默认均分50份
start       数值区间的起始值；
stop        数值区间的终止值；
num         数值区间内要生成多少个均匀的样本。默认值为 50；
endpoint    默认为 True，表示数列包含 stop 终止值，反之不包含；
restep      默认为 True，表示生成的数组中会显示公差项，反之不显示；
dtype       数组元素值的数据类型。
'''
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)

'''
@logspace   返回一个ndarray数组，它用于创建等比数组
start       序列的起始值：base**start。
stop        序列的终止值：base**stop。
num         数值范围区间内样本数量，默认为50。
endpoint    默认为True包含终止值，反之不包含。
base        对数函数的 log 底数，默认为10。
dtype       数组元素值的数据类型。
'''
np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)

'''
@empty  创建未初始化的数组
shape   数组的形状
dtype   数组元素的数据类型，默认值为float
order   数组元素在内存的顺序，默认C，行优先
'''
numpy.empty(shape, dtype = float, order = 'C')

'''
@zeros  创建元素均为0的数组
shape   数组的形状大小
dtype   可选，数组元素的数据类型
order   数组元素在内存的顺序，默认C，行优先
'''
numpy. zeros(shape,dtype=float,order="C")

'''
@ones   创建元素均为1的数组
shape   数组的形状大小
dtype   可选，数组元素的数据类型
order   数组元素在内存的顺序，默认C，行优先
'''
numpy.ones(shape, dtype = None, order = 'C')

'''
@asarray    与array()类似，可将python序列或元组转化为ndarray对象
sequence    接受一个Python序列，可以是列表或者元组
dtype       可选，数组元素的数据类型
order       数组元素在内存的顺序，默认C，行优先
'''
numpy.asarray(sequence，dtype = None ，order = None)

'''
@frombuffer 使用指定的缓冲区创建数组
buffer      将任意对象转换为流的形式读入缓冲区；
dtype       返回数组的数据类型，默认是float32；
count       要读取的数据数量，默认为-1表示读取所有数据；
offset      读取数据的起始位置，默认为0。
'''
numpy.frombuffer(buffer, dtype = float, count = -1, offset = 0)

'''
@fromiter   把迭代对象转换为ndarray数组
iterable    可迭代对象。
dtype       返回数组的数据类型
count       读取的数据数量，默认为-1，读取所有数据。
'''
numpy.fromiter(iterable, dtype, count = -1)
```

```python
# import numpy package
import numpy as np

# numpy.array()
a = np.array([1, 2, 3])
b = np.array([1, 0, 2], dtype=bool)

print("----------------")
print(a)
print(b)

# numpy.arange()
# 长度为6
c = np.arange(6)
# start=1，end=10，step=2
d = np.arange(1, 10, 2)

print("----------------")
print(c)
print(d)

# numpy.linespace()
# start=1,end=10,默认num=50
e = np.linspace(1, 10)
# [1, 10]
f = np.linspace(1, 10, 10, endpoint=True)
# [1, 10)
g = np.linspace(1, 10, 10, endpoint=False)

print("----------------")
print(e)
print(f)
print(g)

# numpy.logspace()
# 2^1 2^2 2^3 ... 2^10
h = np.logspace(1, 10, num=10, base=2)
print("----------------")
print(h)

# numpy.empty()
i = np.empty((2,3), dtype=int)
print("----------------")
print(i)

# numpy.zeros()
j = np.zeros((2, 3))
print("----------------")
print(j)

# numpy.ones()
k = np.ones((3, 2))
print("----------------")
print(k)

# numpy.asarray()
# 列表
data_l = [1, 2, 3, 4]
# 元组
data_t = (5, 6, 7, 8)
l = np.asarray(data_l)
m = np.asarray(data_t)
print("----------------")
print(l)
print(m)
print("----------------")

# output
'''
----------------
[1 2 3]
[ True False  True]
----------------
[0 1 2 3 4 5]
[1 3 5 7 9]
----------------
[ 1.          1.18367347  1.36734694  1.55102041  1.73469388  1.91836735
  2.10204082  2.28571429  2.46938776  2.65306122  2.83673469  3.02040816
  3.20408163  3.3877551   3.57142857  3.75510204  3.93877551  4.12244898
  4.30612245  4.48979592  4.67346939  4.85714286  5.04081633  5.2244898
  5.40816327  5.59183673  5.7755102   5.95918367  6.14285714  6.32653061
  6.51020408  6.69387755  6.87755102  7.06122449  7.24489796  7.42857143
  7.6122449   7.79591837  7.97959184  8.16326531  8.34693878  8.53061224
  8.71428571  8.89795918  9.08163265  9.26530612  9.44897959  9.63265306
  9.81632653 10.        ]
[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
[1.  1.9 2.8 3.7 4.6 5.5 6.4 7.3 8.2 9.1]
----------------
[   2.    4.    8.   16.   32.   64.  128.  256.  512. 1024.]
----------------
[[0 1 2]
 [3 4 5]]
----------------
[[0. 0. 0.]
 [0. 0. 0.]]
----------------
[[1. 1.]
 [1. 1.]
 [1. 1.]]
----------------
[1 2 3 4]
[5 6 7 8]
----------------
'''
```

### Numpy索引和切片

&emsp;&emsp;Numpy内置函数slice()来构造切片。slice(start, stop, step)；**切片还可以使用省略号“…”，如果在行位置使用省略号，那么返回值将包含所有行元素，反之，则包含所有列元素。**

```python
# import numpy package
import numpy as np

a = np.arange(10)
# 坐标从2到8，不包含8 [2, 8)
b = a[2:8:3]

c = np.array([[1,2,3],[3,4,5],[4,5,6]])
# 从[1:]索引处开始切割
d = c[1:]
# 返回数组的第二列
e = c[...,1]
# 返回数组的第二行
f = c[1,...]
# 返回第二列后的所有项
g = c[...,1:]

print("----------------")
print("a", a)
print("b", b)
print("c", c)
print("d", d)
print("e", e)
print("f", f)
print("g", g)
print("----------------")

# output
'''
----------------
a [0 1 2 3 4 5 6 7 8 9]
b [2 5]
c [[1 2 3]
 [3 4 5]
 [4 5 6]]
d [[3 4 5]
 [4 5 6]]
e [2 4 5]
f [3 4 5]
g [[2 3]
 [4 5]
 [5 6]]
----------------
'''
```

### Numpy数据相关操作

#### 数组变维

```python

'''
在不改变数组元素的条件下，修改数组的形状。
'''
numpy.ndarray.reshape()

'''
返回是一个迭代器，可以用for循环遍历其中的每一个元素。
'''
numpy.ndarray.flat

'''
以一维数组的形式返回一份数组的副本，对副本的操作不会影响到原数组。
'''
numpy.ndarray.flatten()

'''
返回一个连续的扁平数组（即展开的一维数组），与flatten不同，它返回的是数组视图（修改视图会影响原数组）。
'''
numpy.ravel()
```

```python
# import numpy package
import numpy as np

# reshape 将一维数组变为二维3行3列数组
a = np.arange(9).reshape((3, 3))
print(a)

# flat返回一个迭代器，for可以遍历矩阵中的每个元素
for element in a.flat:
    print(element, end=" ")
print("\n")

# flatten，以一维数组的形式返回一份数组的副本
# 默认行展开
b = a.flatten()
# 列展开
c = a.flatten(order='F')
print(b)
print(c)

# revel 将多维数组以一维数组形式展开，与flatten区别，如果修改会影响原始数组
print(np.ravel(a))

# output
'''
[[0 1 2]
 [3 4 5]
 [6 7 8]]
0 1 2 3 4 5 6 7 8 

[0 1 2 3 4 5 6 7 8]
[0 3 6 1 4 7 2 5 8]
[0 1 2 3 4 5 6 7 8]
'''
```

#### 数组转置

```python

'''
将数组的维度值进行对换，比如二维数组维度(2,4)使用该方法后为(4,2)。
'''
numpy.transpose()

'''
与transpose方法相同。
'''
ndarray.T

'''
沿着指定的轴向后滚动至规定的位置。
'''
numpy.rollaxis()

'''
对数组的轴进行对换。
'''
numpy.swapaxes()
```

```python
# import numpy package
import numpy as np

# reshape 将一维数组变为二维3行3列数组
a = np.arange(8).reshape((2, 4))
print("a", a)

# transpose T
b = np.transpose(a)
# b = a.transpose()
# b = a.T
print("b", b)

# output
'''
a [[0 1 2 3]
 [4 5 6 7]]
b [[0 4]
 [1 5]
 [2 6]
 [3 7]]
'''
```

#### 数组连接

```python

'''
沿指定轴连接两个或者多个相同形状的数组
a1, a2, ...     表示一系列相同类型的数组；
axis            沿着该参数指定的轴连接数组，默认为0。
'''
numpy.concatenate((a1, a2, ...), axis)

'''
沿着新的轴连接一系列数组
'''
numpy.stack()

'''
水平连接
'''
numpy.hstack()

'''
垂直连接
'''
numpy.vstack()

```

```python
# import numpy package
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print("--------------")
print("a", a)
print("b", b)

# axis默认0，沿x轴连接
c = np.concatenate((a, b), axis=0)
# 沿y轴连接
d = np.concatenate((a, b), axis=1)
print("--------------")
print("c", c)
print("d", d)

# stack 沿新轴连接
e = np.stack((a, b))
print("--------------")
print("e", e)

# output
'''
--------------
a [[1 2]
 [3 4]]
b [[5 6]
 [7 8]]
--------------
c [[1 2]
 [3 4]
 [5 6]
 [7 8]]
d [[1 2 5 6]
 [3 4 7 8]]
--------------
e [[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
'''
```

#### 数组分割

```python

'''
@split 将一个数组分割为多个子数组
ary     被分割的数组
indices_or_sections     若是一个整数，代表用该整数平均切分，若是一个数组，则代表沿轴切分的位置（左开右闭）；
axis    默认为0，表示横向切分；为1时表示纵向切分。
'''
numpy.split(ary, indices_or_sections, axis)

'''
将一个数组水平分割为多个子数组（按列）
'''
numpy.hsplit()

'''
将一个数组垂直分割为多个子数组（按行）
'''
numpy.vsplit()
```

```python
# import numpy package
import numpy as np

a = np.arange(6)
print(a)

# 将a切割为2个数组
b = np.split(a, 2)
print(b)

# 将a按照坐标在[3, 4)部分切割
c = np.split(a, [3, 4])
print(c)

# output
'''
[0 1 2 3 4 5]
[array([0, 1, 2]), array([3, 4, 5])]
[array([0, 1, 2]), array([3]), array([4, 5])]
'''
```

#### 数组元素增删改查

```python

'''
返回指定形状的新数组。
'''
numpy.resize(arr, shape)

'''
将元素值添加到数组的末尾。
'''
numpy.append(arr, values, axis=None)

'''
沿规定的轴将元素值插入到指定的元素前。
'''
numpy.insert(arr, obj, values, axis)

'''
删掉某个轴上的子数组，并返回删除后的新数组。
'''
numpy.delete(arr, obj, axis)

'''
返回数组内符合条件的元素的索引值。
'''
numpy.argwhere()

'''
用于删除数组中重复的元素，并按元素值由大到小返回一个新数组。
'''
numpy.unique(arr, return_index, return_inverse, return_counts)
```

### Numpy常用统计方法

#### 数学函数

| 函数 | 运算符号 | 说明 |
| --- | --- | --- |
| bitwise_and | & | 计算数组元素之间的按位**与**运算 |
| bitwise_or | | | 计算数组元素之间的按位**或**运算 |
| invert | ~ | 计算数组元素之间的按位**取反**运算 |
| left_shift | << | 将二进制数的位数向左移 |
| right_shift | >> | 将二进制数的位数向右移 |
| sin |  | 正弦 |
| cos |  | 余弦 |
| tan |  | 正切 |

#### 算术运算

```python

'''
@around     返回一个十进制值数，并将数值四舍五入到指定的小数位上
a           代表要输入的数组；
decimals    要舍入到的小数位数。它的默认值为0，如果为负数，则小数点将移到整数左侧。
'''
numpy.around(a,decimals)

'''
@floor      表示对数组中的每个元素向下取整数，即返回不大于数组中每个元素值的最大整数。
'''
numpy.floor()

'''
@ceil       数与floor函数相反，表示向上取整。
'''
numpy.ceil()

'''
@add        数组相加
'''
numpy.add()

'''
@subtract   数组相减
'''
numpy.subtract()

'''
@multiple   数组元素相乘（内积，这里是对应元素相乘，不是矩阵相乘，矩阵相乘为A*B）
'''
numpy.multiple()

'''
@devide     数组元素相除
'''
numpy.devide()

'''
@reciprocal 数组中的每个元素取倒数，并以数组的形式将它们返回
'''
numpy.reciprocal()

'''
@power      将a数组中的元素作为底数，把b数组中与a相对应的元素作幂，最后以数组形式返回两者的计算结果
'''
numpy.power()

'''
@mod        返回两个数组相对应位置上元素相除后的余数，它与numpy.remainder()的作用相同
'''
numpy.mod()

```

```python
# import numpy package
import numpy as np

a = np.array([1.234, 5.678, 9.012, 3.456])
print("-------------")
print(a)

# 四舍五入到两位小数
b = np.around(a, 2)
print("-------------")
print(b)

# 向下取整
c = np.floor(a)
print("-------------")
print(c)

# 向上取整
d = np.ceil(a)
print("-------------")
print(d)

aa = np.arange(9).reshape(3, 3)
bb = np.array([10, 10, 10])

# 加法
cc = np.add(aa, bb)
# 减法
dd = np.subtract(aa, bb)
# 乘法
ee = np.multiply(aa, bb)
# 除法
ff = np.divide(aa, bb)
print("-------------")
print(aa)
print("-------------")
print(bb)
print("-------------")
print(cc)
print("-------------")
print(dd)
print("-------------")
print(ee)
print("-------------")
print(ff)

# 倒数
a = np.array([0.25, 1.33, 1, 0, 100])
b = np.reciprocal(a)
print("-------------")
print(a)
print("-------------")
print(b)

# 幂
a = np.array([10, 100, 1000])
b = np.power(a, 2)
c = np.power(a, [1, 2, 3])
print("-------------")
print(a)
print("-------------")
print(b)
print("-------------")
print(c)

# 取模
a = np.array([11,22,33])
b = np.array([3,5,7])
#a与b相应位置的元素做除法
c = np.mod(a,b)
# np.remainder(a,b)
print("-------------")
print(c)

# output
'''
-------------
[1.234 5.678 9.012 3.456]
-------------
[1.23 5.68 9.01 3.46]
-------------
[1. 5. 9. 3.]
-------------
[ 2.  6. 10.  4.]
-------------
[[0 1 2]
 [3 4 5]
 [6 7 8]]
-------------
[10 10 10]
-------------
[[10 11 12]
 [13 14 15]
 [16 17 18]]
-------------
[[-10  -9  -8]
 [ -7  -6  -5]
 [ -4  -3  -2]]
-------------
[[ 0 10 20]
 [30 40 50]
 [60 70 80]]
-------------
[[0.  0.1 0.2]
 [0.3 0.4 0.5]
 [0.6 0.7 0.8]]
-------------
[  0.25   1.33   1.     0.   100.  ]
-------------
[4.        0.7518797 1.              inf 0.01     ]
-------------
[  10  100 1000]
-------------
[    100   10000 1000000]
-------------
[        10      10000 1000000000]
-------------
[2 2 5]
'''
```

#### 统计函数

```python

'''
沿指定的轴，查找数组中元素的最小值，并以数组形式返回；
'''
numpy.amin()

'''
沿指定的轴，查找数组中元素的最大值，并以数组形式返回。
'''
numpy.amax()

'''
用于计算数组元素中最值之差值，也就是（最大值 - 最小值）。
'''
numpy.ptp()

'''
百分位数，是统计学中使用的一种度量单位。该函数表示沿指定轴，计算数组中任意百分比分位数
a       输入数组；
q       要计算的百分位数，在 0~100 之间；
axis    沿着指定的轴计算百分位数。
'''
numpy.percentile(a, q, axis)

'''
计算a数组元素的中位数（中值）
'''
numpy.median()

'''
沿指定的轴，计算数组中元素的算术平均值（即元素之总和除以元素数量）
'''
numpy.mean()

'''
a = [1, 2, 3, 4]
weights = [4, 3, 2, 1]
加权平均值=（1 * 4 + 2 * 3 + 3 * 2 + 4 * 1）/（4 + 3 + 2 + 1）

a           数组
weights     权重数组
returned    可选，为True返回元组（加权平均值，权重和）
'''
numpy.average(a, weights, returned)

'''
方差
'''
numpy.var()

'''
标准差
'''
numpy.std()
```

#### 矩阵运算

&emsp;&emsp;矩阵库模块，numpy.matlib。

#### Matrix矩阵库

```python

'''
返回一个空矩阵，所以它的创建速度非常快。
shape   以元组的形式指定矩阵的形状。
dtype   表示矩阵的数据类型。
order   有两种选择，C（行序优先）或者 F（列序优先）
'''
numpy.matlib.empty(shape, dtype, order)

'''
创建一个以0填充的矩阵
'''
numpy.matlib.zeros()

'''
创建一个以1填充的矩阵
'''
numpy.matlib.ones()

'''
返回一个对角线元素为1，而其他元素为0的矩阵
n       返回矩阵的行数
M       返回矩阵的列数，默认为n
k       对角线的索引
dtype   矩阵中元素数据类型
'''
numpy.matlib.eye(n,M,k, dtype)

'''
返回一个给定大小的单位矩阵，矩阵的对角线元素为1，而其他元素均为0
'''
numpy.matlib.identity()

'''
创建一个以随机数填充，并给定维度的矩阵
'''
numpy.matlib.rand()
```

#### 线性代数

&emsp;&emsp;Numpy提供了numpy.linalg模块，包含了一些常用的线性代数计算方法。
| 函数 | 说明 |
| --- | --- |
| dot | 两个数组的点积 |
| vdot | 两个向量的点积 |
| inner | 两个数组的内积 |
| matmul | 两个数组的矩阵积 |
| det | 计算输入矩阵的行列式 |
| solve | 求解线性矩阵方程 |
| inv | 计算矩阵的逆矩阵，逆矩阵与原始矩阵相乘，会得到单位矩阵 |


#### 矩阵乘法

##### 逐元素矩阵乘法

```python

# numpy.multiply()

```

##### 矩阵乘积运算

```python

# numpy.matmul()

```

##### 矩阵点积

```python

# numpy.dot()

```

### Numpy输入输出(IO)

&emsp;&emsp;Numpy从磁盘的文件中加载ndarray对象，可处理二进制文件和普通文本文件。IO操作方法如下：
| 文件类型 | 处理方法 |
| --- | --- |
| 二进制文件 | load()和save() |
| 普通文本文件 | loadtxt()和savetxt() |


[1]:https://numpy.org/
[2]:https://www.numpy.org.cn/user/basics/types.html#%E6%95%B0%E7%BB%84%E7%B1%BB%E5%9E%8B%E4%B9%8B%E9%97%B4%E7%9A%84%E8%BD%AC%E6%8D%A2
[3]:https://www.numpy.org.cn/user/basics/types.html#%E6%95%B0%E7%BB%84%E7%B1%BB%E5%9E%8B%E4%B9%8B%E9%97%B4%E7%9A%84%E8%BD%AC%E6%8D%A2