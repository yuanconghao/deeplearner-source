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

<img src="/images/matplotlib/matplotlib_1.png" width="250px"></img>

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
<img src="/images/matplotlib/matplotlib_basic_py.png" width="400px"></img>
&emsp;&emsp;<font color="##ff0000">图二，该图为在jupyter内运行生成的图片，两者还是有差别的，经过仔细对比研究，发现jupyter的图像分辨率长宽为原生Python运行的一半。</font>
<img src="/images/matplotlib/matplotlib_basic.png" width="400px"></img>

&emsp;&emsp;从这几行代码画出的图像中（图一），可以看出以下几点：
* 图像画布默认的大小为<b>[6.4, 4.8]</b>英寸；分辨率dpi为<b>100.0</b>；背景颜色为<b>白色</b>；边框颜色为<b>白色</b>；折线的默认颜色为<b>'#1f77b4'</b>;经查阅[matplotlib官方文档][2]；
<img src="/images/matplotlib/matplotlib_figure_args.png" width="600px"></img>
* 这里有个dpi和分辨率的换算，水平分辨率为$6.4\times 100 = 640$，垂直分辨率为$4.8 \times 100 = 480$ ，图片分辨率为$640 \times 480$。
* <font color="##ff0000">可通过figure类的参数设置，来改变图片大小、分辨率等。</font>
* <font color="##ff0000">可调整x轴和y轴的间距与设置描述信息</font>

#### 设置图片大小、分辨率、颜色

```python
# pyplot import method
import matplotlib.pyplot as plt

# 创建图形图像（画布）
'''
figsize    指定画布的大小，(宽度,高度)，单位为英寸。
dpi        分辨率，每英寸多少个像素
facecolor  背景颜色
linewidth  边框宽度
edgecolor  边框颜色
frameon    是否显示边框
'''
fig = plt.figure(figsize=[5, 3], dpi=200, facecolor='#0000ff', edgecolor='#ff0000', linewidth=1, frameon=True)

# 准备数据
x = range(2, 26, 2)
y = [15, 13, 14.5, 17, 20, 25, 26, 26, 27, 22, 18, 15]

# 绘制图形
plt.plot(x, y)

# 图片展示
plt.show()

# 图片保存
plt.savefig("./matplotlib_figure.png")
```
<img src="/images/matplotlib/matplotlib_figure.png"></img>

* <font color="##ff0000">图片尺寸，水平宽度为5英尺，垂直高度为3英尺；DPI为200，分辨率为$1000 \times 600$ ;</font>
* <font color="##ff0000">图片背景颜色为蓝色(blue)；边框宽度为1；边框颜色为红色(red)</font>

#### 设置坐标轴间距与描述信息

```python
# pyplot import method
import matplotlib.pyplot as plt

# 创建图形图像（画布）
'''
@figure
figsize    指定画布的大小，(宽度,高度)，单位为英寸。
dpi        分辨率，每英寸多少个像素
facecolor  背景颜色
linewidth  边框宽度
edgecolor  边框颜色
frameon    是否显示边框
'''
fig = plt.figure(figsize=[4, 2], dpi=100, edgecolor='#ff0000', linewidth=2, frameon=True)

# 坐标轴间距与描述信息
'''
@add_axes
rect       新坐标轴尺寸，四个数值[left, bottom, width, height]，一般介于0-1之间，为分数，代表百分比。
'''
ax = fig.add_axes([0.1, 0.1, 1, 1])
# 设置x轴范围
ax.set_xlim(0, 30)
# 设置y轴范围
ax.set_ylim(0, 30)

# 设置图片标题
ax.set_title("Line Chart")
# 设置x轴描述信息
ax.set_xlabel("x axis")
# 设置y轴描述信息
ax.set_ylabel("y axis")

# 准备数据
x = range(2, 26, 2)
y = [15, 13, 14.5, 17, 20, 25, 26, 26, 27, 22, 18, 15]

# 绘制图形
ax.plot(x, y)

# 图片展示
plt.show()

# 图片保存
#plt.savefig("./matplotlib_axes.png")
```

<img src="/images/matplotlib/matplotlib_axes.png"></img>

* <font color="##ff0000">add_axes的rect参数代表显示的百分比部分，可以看到left为从10%开始展示，最后侧边框的红色部分没有展示出来。</font>

#### 设置线条样式
| 颜色代码 | 含义 |
| --- | --- |
| 'b' | 蓝色 |
| 'g' | 绿色 |
| 'r' | 红色 |
| 'c' | 青色 |
| 'm' | 品红色 |
| 'y' | 黄色 |
| 'k' | 黑色 |
| 'w' | 白色 |

| 线条表示符号 | 含义 |
| --- | --- |
| '-' | 实线 |
| '--' | 虚线 |
| '-.' | 点划线 |
| ':' | 虚线 |

| [标记符号][3] | 含义 |
| --- | --- |
| '.' | 点 |
| 'o' | 圆圈 |
| 'x' | X |
| 'D' | 钻石 |
| 'H' | 六角形 |
| 's' | 正方形 |
| '+' | 加号 |
| 'v' | 倒三角 |
| '^' | 三角 |
| '<' | 左三角 |
| '>' | 右三角 |

```python
# pyplot import method
import matplotlib.pyplot as plt

# 准备数据
x = range(2, 26, 2)
y = [15, 13, 14.5, 17, 20, 25, 26, 26, 27, 22, 18, 15]

# 创建图形图像（画布）
'''
@figure
figsize    指定画布的大小，(宽度,高度)，单位为英寸。
dpi        分辨率，每英寸多少个像素
facecolor  背景颜色
linewidth  边框宽度
edgecolor  边框颜色
frameon    是否显示边框
'''
fig = plt.figure(figsize=[4, 2], dpi=100, edgecolor='#ff0000', linewidth=2, frameon=True)

# 坐标轴间距与描述信息
'''
@add_axes
rect       新坐标轴尺寸，四个数值[left, bottom, width, height]，一般介于0-1之间，为分数，代表百分比。
'''
ax = fig.add_axes([0, 0, 1, 1])
# 设置x轴范围
ax.set_xlim(0, 30)
# 设置y轴范围
ax.set_ylim(0, 30)

# 设置图片标题
ax.set_title("Line Chart")
# 设置x轴描述信息
ax.set_xlabel("x axis")
# 设置y轴描述信息
ax.set_ylabel("y axis")

# 设置线条样式
ax.plot(x, y, 'go--')
# 绘制图例
'''
@legend
handles     所有线型实例，序列
labels      标签名称，字符串序列
loc         图例位置：
(Best/0/自适用；upper right/1/右上；upper left/2/左上；lower left/3/左下；lower right/4/右下；)
(right/5/右侧；center left/6/居中靠左；center right/7/居中靠右；lower center/8/底部居中；upper center/9/上部居中；center/10/中部；)
'''
ax.legend(labels=('GDP'), loc='lower right')

# 图片展示
plt.show()

# 图片保存
#plt.savefig("./matplotlib_axes1.png")
```
<img src="/images/matplotlib/matplotlib_axes1.png"></img>

#### 设置显示中文

&emsp;&emsp;环境用的是mac，anaconda，首先要查看环境支持那些中文字体；
```
conda install fontconfig
fc-list :lang=zh
```

```python
# pyplot import method
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams["font.sans-serif"]=["Arial Black"]
# 该语句解决图像中的“-”负号的乱码问题
plt.rcParams["axes.unicode_minus"]=False


# 准备数据
x = range(2, 26, 2)
y = [15, 13, 14.5, 17, 20, 25, 26, 26, 27, 22, 18, 15]

# 创建图形图像（画布）
'''
@figure
figsize    指定画布的大小，(宽度,高度)，单位为英寸。
dpi        分辨率，每英寸多少个像素
facecolor  背景颜色
linewidth  边框宽度
edgecolor  边框颜色
frameon    是否显示边框
'''
fig = plt.figure(figsize=[4, 2], dpi=100, edgecolor='#ff0000', linewidth=2, frameon=True)

# 坐标轴间距与描述信息
'''
@add_axes
rect       新坐标轴尺寸，四个数值[left, bottom, width, height]，一般介于0-1之间，为分数，代表百分比。
'''
ax = fig.add_axes([0, 0, 1, 1])
# 设置x轴范围
ax.set_xlim(0, 30)
# 设置y轴范围
ax.set_ylim(0, 30)

# 设置图片标题
ax.set_title("折线图")
# 设置x轴描述信息
ax.set_xlabel("x轴")
# 设置y轴描述信息
ax.set_ylabel("y轴")

# 设置线条样式
ax.plot(x, y, 'go--')
# 绘制图例
'''
@legend
handles     所有线型实例，序列
labels      标签名称，字符串序列
loc         图例位置：
(Best/0/自适用；upper right/1/右上；upper left/2/左上；lower left/3/左下；lower right/4/右下；)
(right/5/右侧；center left/6/居中靠左；center right/7/居中靠右；lower center/8/底部居中；upper center/9/上部居中；center/10/中部；)
'''
ax.legend(labels=('GDP'), loc='lower right')

# 图片展示
plt.show()

# 图片保存
#plt.savefig("./matplotlib_zh.png")

```

<img src="/images/matplotlib/matplotlib_zh.png"></img>


### matplotlib折线图

```python
import matplotlib.pyplot as plt

# 对比两天内同一时刻温度的变化情况
x = [5, 8, 12, 14, 16, 18, 20]
y1 = [18, 21, 29, 31, 26, 24, 20]
y2 = [15, 18, 24, 30, 31, 25, 24]

#绘制折线图，添加数据点，设置点的大小
plt.plot(x, y1, 'c',marker='o', markersize=5)
plt.plot(x, y2, 'g', marker='o',markersize=5)

# 折线图标题
plt.title('Line Chart Of Temperature Change')  
# x轴标题
plt.xlabel('Time(h)')  
# y轴标题
plt.ylabel('Temperature(C)')  

#给图像添加注释，并设置样式
for a, b in zip(x, y1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
for a, b in zip(x, y2):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

#绘制图例
plt.legend(['First Day', 'Second Day'])

#显示图像
plt.show()
```
<img src="/images/matplotlib/matplotlib_line.png"></img>

### matplotlib散点图

```python
import matplotlib.pyplot as plt

# 准备数据
girls_grades = [89, 90, 70, 89, 100, 80, 90, 100, 80, 34]
boys_grades = [30, 29, 49, 48, 100, 48, 38, 45, 20, 30]
grades_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 创建画布
fig=plt.figure()

#添加绘图区域
ax=fig.add_axes([0, 0, 1, 1])

ax.scatter(grades_range, girls_grades, color='g', label="girls")
ax.scatter(grades_range, boys_grades, color='r', label="boys")

# 设置标题
ax.set_title('scatter plot')
# 设置x轴标签
ax.set_xlabel('Grades Range')
# 设置y轴标签
ax.set_ylabel('Grades Scored')

# 添加图例
plt.legend()

# 展示图像
plt.show()
```
<img src="/images/matplotlib/matplotlib_scatter.png"></img>

### matplotlib柱状图
```python
import matplotlib.pyplot as plt

# 创建图形对象
fig = plt.figure()

# 添加子图区域，参数值表示[left, bottom, width, height ]
ax = fig.add_axes([0, 0, 1, 1])

# 准备数据
langs = ['C', 'C++', 'Java', 'Python', 'PHP']
students = [23,17,35,29,12]

# 绘制柱状图
ax.bar(langs,students)
plt.show()
```
<img src="/images/matplotlib/matplotlib_bar.png"></img>

### matplotlib多次柱状图

```python
import numpy as np
import matplotlib.pyplot as plt

# 准备数据
data = [[30, 25, 50, 20], [40, 23, 51, 17], [35, 22, 45, 19]]
X = np.arange(4)

# 创建画布
fig = plt.figure()

# 添加子图区域
ax = fig.add_axes([0, 0, 1, 1])

# 绘制柱状图
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
```

<img src="/images/matplotlib/matplotlib_multi_bar.png"></img>

### matplotlib直方图
```python
from matplotlib import pyplot as plt
import numpy as np

#创建图形对象和轴域对象
fig,ax = plt.subplots(1,1)

# 准备数据
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])

# 绘制直方图
ax.hist(a, bins = [0,25,50,75,100])

# 设置标题
# 数学表达式，放到$$内
ax.set_title("histogram of result " + r'$\alpha > \beta$')

# 设置坐标轴
ax.set_xticks([0,25,50,75,100])

# 设置标签
ax.set_xlabel('marks')
ax.set_ylabel('no.of students')
plt.show()
```
<img src="/images/matplotlib/matplotlib_hist.png"></img>


### matplotlib饼图
```python
from matplotlib import pyplot as plt
import numpy as np

# 准备数据
langs = ['C', 'C++', 'Java', 'Python', 'PHP']
students = [23,17,35,29,12]

# 添加图形对象
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

# 使得X/Y轴的间距相等
ax.axis('equal')

#绘制饼状图
ax.pie(students, labels = langs,autopct='%1.2f%%')
plt.show()
```
<img src="/images/matplotlib/matplotlib_pie.png"></img>

[1]:https://matplotlib.org/
[2]:https://matplotlib.org/stable/api/figure_api.html
[3]:https://matplotlib.org/stable/api/_as_gen/matplotlib.markers.MarkerStyle.html#matplotlib.markers.MarkerStyle


