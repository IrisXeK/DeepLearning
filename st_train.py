import sys
import os
import torch
import torchvision
import torch.nn as nn
from torch.utils import data
import matplotlib.pyplot as plt
from torchvision import transforms


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
    def __init__(self, legend_name: tuple, num_epochs) -> None:
        self.res_dict = {name: [] for name in legend_name}
        self.num_epochs = num_epochs

    def plot_res(self):
        for legend_name, data in self.res_dict.items():
            plt.plot(list(range(self.num_epochs)), data, label=legend_name)
        plt.title("Result")
        plt.xlabel("num_epochs")
        plt.ylabel("ResultValue")
        plt.legend()
        plt.show()

# 获取训练集中的数据对应的标签 labels参数传入 MINST_train.train_labels


def std_get_MINST_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def std_accuracy(y_hat, y):  # 计算预测正确的数量
    """
    如果y_hat存储的是矩阵,假定第二个维度存储每个类的预测分数
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # 取出y_hat中每一行中的最大的那个概率的索引
    # 比较y_hat和y中的每个位置相不相等, 注意这之前要先把它们的类型转换为一样的 .type(dtype)函数表示将这个tensor的类型转为dtype
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def std_evaluate_accuracy(net, data_iter):  # 对于任何data_iter可访问的数据集 都可以评估模型的精度
    if isinstance(net, nn.Module):
        net.eval()  # 模型设置为评估模式
    metric = Accumulator(2)  # 2个位置为 正确预测数和预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(std_accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def std_train_epoch(net, train_set, loss_function, updater):  # 模型在训练周期中的一次训练
    """
    updater是更新模型参数的函数,接收批量大小作为参数
    updater可以是sgd函数 也可以是框架内的内置函数
    """
    if isinstance(net, nn.Module):
        net.train()
    metric = Accumulator(3)  # 三个位置为 训练损失总和 训练准确数 样本数
    for X, y in train_set:
        y_hat = net(X)  # 给出一次预测
        loss = loss_function(y_hat, y)  # 计算损失
        if isinstance(updater, torch.optim.Optimizer):  # updater为Pytorch框架的内置优化器
            updater.zero_grad()  # 将grad置为0 因为pytorch计算梯度时会累加
            loss.mean().backward()  # 计算梯度
            updater.step()  # 由计算出的梯度更新参数
        else:  # 使用的是定制的优化器和损失函数
            loss.sum().backward()
            updater(X.shape[0])
    metric.add(float(loss.sum()), std_accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]  # 返回训练损失和训练精度


def train(net, train_set, test_set, loss_function, num_epochs, updater, Res: ResVisualization):  # 训练模型
    for epoch in range(num_epochs):
        train_metrics = std_train_epoch(net, train_set, loss_function, updater)
        print(
            f"Epoch:{epoch},训练平均损失:{train_metrics[0] :.4f}, 训练准确度:{train_metrics[1]:.3f}")
        test_accurancy = std_evaluate_accuracy(net, test_set)
        Res.res_dict['train_loss'].append(train_metrics[0])
        Res.res_dict['train_acc'].append(train_metrics[1])
        Res.res_dict['test_acc'].append(test_accurancy)
    train_loss, train_accuracy = train_metrics
    assert train_loss < 0.7, train_loss
    assert train_accuracy <= 1 and train_accuracy > 0.7, train_accuracy
    assert test_accurancy <= 1 and test_accurancy > 0.7, test_accurancy


def std_prediction(net, test_set, n=6):
    for X, y in test_set:
        _, axes = plt.subplots(1, n, figsize=(8, 8))
        true_labels = std_get_MINST_labels(y)
        pred_labels = std_get_MINST_labels(net(X).argmax(axis=1))
        titles = ['T:' + true + '\n' + 'P:' + pred for true,
                  pred in zip(true_labels, pred_labels)]
        for i in range(n):
            axes[i].imshow(X[i].reshape((28, 28)))
            axes[i].set_title(titles[i])
            axes[i].axis('off')
