import torch
import matplotlib.pyplot as plt

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5,2.5), cmap='Reds'):
    """
    显示矩阵热图的函数\n
    参数:\n
        matrices : 一个四维矩阵列表, shape=(要显示的行数，要显示的列数，查询的数目，键的数目)\n
                    一个矩阵是一个注意力头的权重矩阵
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

if __name__ == '__main__':
    attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
    show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
