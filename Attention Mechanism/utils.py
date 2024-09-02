import os, hashlib, requests, zipfile, tarfile, torch, random, time, math, collections
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils import data

class Accumulator:  # 累加多个变量的实用程序类
    def __init__(self, n):
        self.data = [0.0]*n

    def add(self, *args):  # 在data的对应位置加上对应的数
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class ResVisualization:
    def __init__(self, xlist: tuple | list, ylist: tuple | list, legend_names: tuple | list, is_grid=None,
                 xlabel: str = None, ylabel: str = None, title: str = None,
                 xlim: list = None, ylim: list = None, line_style: str = '-') -> None:
        """
        xlist : 二维数组,每一行代表一个曲线的x坐标\n
        ylist : 二维数组,每一行代表一个曲线的y坐标\n
        legend_names : 列表，代表每条曲线的名字\n
        is_grid : 是否显示网格\n
        xlabel : x轴的名字\n
        ylabel : y轴的名字\n
        title : 图的名字\n
        xlim : x轴的范围\n
        ylim : y轴的范围\n
        line_style : 曲线的样式\n
        """
        self.res_dict = {name: (x, y) for name, x, y 
                        in zip(legend_names, xlist, ylist)}
        self.is_grid = is_grid
        self.xlim = xlim
        self.ylim = ylim
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.line_style = line_style

    def add(self, x_val, y_val, name):
        """向名为name的曲线中添加一个(x_val, y_val)数据对"""
        self.res_dict[name][0].append(x_val)
        self.res_dict[name][1].append(y_val)

    def plot_res(self):
        for name, xy_pair in self.res_dict.items():
            plt.plot(xy_pair[0], xy_pair[1], label=name,
                     linestyle=self.line_style)
        if self.is_grid:
            plt.grid()
        if self.title is not None:
            plt.title(self.title)
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)
        if self.xlim is not None:
            plt.xlim(self.xlim)
        if self.ylim is not None:
            plt.ylim(self.ylim)
        plt.legend()
        plt.show()

class Timer:
    """一个计时器类,含有start, stop, get_elapesd_time方法"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.elapsed_time_sum = 0

    def start(self):
        self.start_time = time.time()
        self.end_time = None
        self.elapsed_time = None

    def stop(self):
        if self.start_time is None:
            raise ValueError("计时器还没有开始计时")
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.elapsed_time_sum += self.elapsed_time

    def get_elapsed_time(self):
        if self.elapsed_time is None:
            raise ValueError("计时器未被停止计时")
        return self.elapsed_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        # print(f"Elapsed time: {self.get_elapsed_time():.4f} seconds")

def sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None] # 应用广播机制进行比较
        # 这里, torch.arange获取了一个从0到maxlen-1的整数序列并用[None, :]升为shape=(1, maxlen)的二维矩阵。
        # valid_len通过[:, None]升为shape=(len(valid_len), 1)的二维矩阵。 
        # 这里的通过None升维的操作和unsqueeze()是一样的
        # 之后进行比较 通过广播机制, arange数组中长度<valid_len的部分得到True, 其他为False。 得到了一张布尔表mask
        # 通过布尔表, 进行X[~mask]=value即可通过切片把X对应的位置替换为value指定的值。
        X[~mask] = value
        return X

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5,2.5), cmap='Reds'):
    """
    显示矩阵热图的函数\n
    参数:\n
        matrices : 一个四维矩阵列表, shape=(要显示的行数，要显示的列数，查询的数目，键的数目)\n
    """
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                            sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)): # 遍历每一行矩阵和它们对应的子图
        for j, (axis, matrix) in enumerate(zip(row_axes, row_matrices)): # 从每一行矩阵遍历每个矩阵和对应的子图
            pcm = axis.imshow(matrix.detach().numpy(), cmap=cmap) # 绘制矩阵热图并返回Pseudocolor Map
            if i == num_rows-1:
                axis.set_xlabel(xlabel)
            if j == 0:
                axis.set_ylabel(ylabel)
            if titles:
                axis.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6) # 利用pcm绘制色条