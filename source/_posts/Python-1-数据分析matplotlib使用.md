---
title: Python-1.数据分析matplotlib使用
date: 2022-05-19 11:54:20
toc: true
mathjax: true
categories:
    - Python

tags:
    - Python
    - 数据分析
    - matplotlib
---

[Matplotlib][1]官方定义：Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. Matplotlib makes easy things easy and hard things possible.

<!--more-->

### 图像基础知识

#### 英寸（inches）
&emsp;&emsp;尺寸，比如图像尺寸、屏幕尺寸；比如一寸照片、经典iPhone4s屏幕尺寸3.5英寸；**1英寸=2.54厘米**；

#### 像素（pixel）
&emsp;&emsp;像素，是指图像的最小方格，是图像中不可分割的元素，是图像中的最小单位；像素可由RGB颜色组合而成；

#### 分辨率（resolution）
&emsp;&emsp;分辨率，是指在图像/屏幕水平方向和垂直方向上的像素个数。如100*80，水平方向含有100个像素点，垂直方向含有80个像素点；

#### DPI（dot per inch）
&emsp;&emsp;像素密度，又叫屏幕密度，指的是每英寸上的像素数，数值越大，图像/屏幕越清晰；


### matplotlib图形组成

&emsp;&emsp;matplotlib主要有Figure（图形/画布）、Axis（坐标轴）、Axes（坐标系）、Artist（绘制对象）组成，如图所示：

<img src="/images/matplotlib_1.png" width="250px"></img>

&emsp;&emsp;<font color="##ff0000">用代码画图跟艺术家用笔画图原理是类似的，选一张什么尺寸、颜色、材料等的纸，在纸张上进行构图布局，然后再进行作画，画什么内容，用什么颜色。同样，matplotlib，要先创建一张画布（Figure），可指定大小、背景颜色、边框等，可设置坐标轴（Axis），建立坐标系（Axes），在画布上生成图像元素（Artist）。</font>

### matplotlib基础绘图

```python
# pyplot import method
import matplotlib.pyplot as plt

# 准备数据
x = range(2, 26, 2)
y = [15, 13, 14.5, 17, 20, 25, 26, 26, 27, 22, 18, 15]

# 绘制图形
plt.plot(x, y)

# 图片展示
plt.show()

# 图片保存
plt.savefig("./matploglib_default.png")
```

&emsp;&emsp;<font color="##ff0000">图一，该图为Python运行后，生成的默认图片</font>
<img src="/images/matplotlib_basic_py.png" width="400px"></img>
&emsp;&emsp;<font color="##ff0000">图二，该图为在jupyter内运行生成的图片，两者还是有差别的，经过仔细对比研究，发现jupyter的图像分辨率长宽为原生Python运行的一半。</font>
<img src="/images/matplotlib_basic.png" width="400px"></img>

&emsp;&emsp;从这几行代码画出的图像中（图一），可以看出以下几点：
* 图像画布默认的大小为<b>[6.4, 4.8]</b>英寸；分辨率dpi为<b>100.0</b>；背景颜色为<b>白色</b>；边框颜色为<b>白色</b>；折线的默认颜色为<b>'#1f77b4'</b>;经查阅[matplotlib官方文档][2]；
<img src="/images/matplotlib_figure_args.png" width="600px"></img>
* 这里有个dpi和分辨率的换算，水平分辨率为$6.4\times 100 = 640$，垂直分辨率为$4.8 \times 100 = 480$ ，图片分辨率为$640 \times 480$。
* <font color="##ff0000">可通过figure类的参数设置，来改变图片大小、分辨率等。</font>
* <font color="##ff0000">可调整x轴和y轴的间距与设置描述信息</font>
* <font color="##ff0000">可调整x轴和y轴的间距与设置描述信息</font>

#### 设置图片大小

#### 设置描述信息（x、y轴信息）

#### 调整x轴和y轴间距

#### 设置线条样式

#### 标记特殊的点

#### 设置显示中文

#### 设置图形信息

### matplotlib折线图

### matplotlib散点图

### matplotlib条形图

### matplotlib多次条形图

### matplotlib直方图




[1]:https://matplotlib.org/
[2]:https://matplotlib.org/stable/api/figure_api.html


