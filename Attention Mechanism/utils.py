import os, hashlib, requests, zipfile, tarfile, torch, random, time, math, collections
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils import data


DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

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

class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，queries的形状:(batch_size，查询的个数，1，num_hidden)
        # key的形状:(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1) # 求和后shape:(batch_size,查询的个数，“键-值”对的个数，num_hiddens)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        scores = self.w_v(features).squeeze(-1) # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        self.attention_weights = masked_softmax(scores, valid_lens) # --> 形状：(batch_size, 查询个数, “键－值”对的个数)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class Vocabulary:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None) -> None:
        """"
        tokens : 词元列表
        min_freq : 最小的词元出现次数
        reserved_tokens : 保留的词元列表
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = self.count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x:x[1], reverse=True) # 按出现频率降序排序
        self.idx_to_token = ['<unk>'] + reserved_tokens # 未知的词元索引为0
        self.token_to_idx = {token:idx for idx, token in enumerate(self.idx_to_token)}  # 单词到索引的映射
        for token, freq in self._token_freqs: # 根据min_freq规则,过滤部分tokens中的词 剩下的添加到单词表
            if freq < min_freq:
                break
            if token not in self.token_to_idx: # 如果当前单词不在保留单词中,则添加到单词表中
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def count_corpus(self, tokens):
        """
        统计词元的出现频率
        tokens : 1D或2D列表
        """
        if len(tokens) == 0 or isinstance(tokens[0], list): # 若是空列表或二维列表,则展平为一维列表
            # 将二维列表展平成一个一维列表
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list,tuple)): # 如果不是列表或元组,则直接以单个键的方式返回一个token的索引
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens] # 以多个键方式返回若干个token的索引列表
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def num_tokens(self):
        return len(self)
    
    @property
    def unk(self):
        return 0 # 未知的单词索引为0
    
    @property
    def token_freqs(self):
        return self._token_freqs

class Encoder(nn.Module):
    """编码器:接受一个长度可变的序列作为输入，并将其转换为具有固定形状的编码状态。"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError

class Seq2SeqEncoder(Encoder):
    """
    用于序列到序列学习的循环神经网络编码器。
    使用了嵌入层(embedding layer)来获得输入序列中每个词元的特征向量。
    嵌入层的权重是一个矩阵,其行数等于输入词表的大小(vocab_size),其列数等于特征向量的维度(embed_size)。
    对于任意输入词元的索引i,嵌入层获取权重矩阵的第i行(从0开始)以返回其特征向量。
    选择了一个多层门控循环单元来实现编码器。
    """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size) # 嵌入层
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X).permute(1,0,2) # 转置后输出shape:(num_steps, batch_size, embed_size) 第一个轴对应时间步
        output, state = self.rnn(X)
        # 输出output的形状为 (num_steps, batch_size, num_hiddens)
        # 输出state的形状为  (num_layers, batch_size, num_hiddens)
        return output, state # 输出和隐状态 最后一次更新的隐状态即为要传递给Decoder的上下文变量

class MaskedSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    """带屏蔽不相关项的交叉熵损失函数"""
    def sequence_mask(self, X, valid_len, value=0):
        """
        在序列中将不相关的项(如用于填充的"<pad>")替换为value值的函数\n
        参数:\n
            X : 未经转置的原始数据 shape=(batch_size, num_steps, num_features)\n
            valid_len : 一个列表,指示X的每个batch的有效长度\n
            value : 用于替换不相关项的值\n
        
        例如,如果两个序列的有效长度(不包括填充词元)分别为1和2,\n
        则第一个序列的第一项和第二个序列的前两项之后的剩余项将被清除为value指定的值(默认为0)。\n
        即:将填充词元的预测排除在损失函数的计算之外
        """
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None] # 应用广播机制进行比较
        # 这里, torch.arange获取了一个从0到maxlen-1的整数序列并用[None, :]升为shape=(1, maxlen)的二维矩阵。
        # valid_len通过[:, None]升为shape=(len(valid_len), 1)的二维矩阵。 
        # 这里的通过None升维的操作和unsqueeze()是一样的
        # 之后进行比较 通过广播机制, arange数组中长度<valid_len的部分得到True, 其他为False。 得到了一张布尔表mask
        # 通过布尔表, 进行X[~mask]=value即可通过切片把X对应的位置替换为value指定的值。
        X[~mask] = value
        return X

    def forward(self, pred, label, valid_len) -> torch.Tensor:
        """
        参数:\n
            pred : Tensor, shape=(batch_size, num_steps, vocab_size)\n
            label : Tensor, shape=(batch_size, num_steps)\n
            valid_len : Tensor, shape=(batch_size,)\n
        返回:\n
            weighted_loss : Tensor, shape = (batch_size,)
        """
        weights = torch.ones_like(label, device=label.device)
        weights = self.sequence_mask(weights, valid_len)  # 将weights中不相关的位置置0后与原始损失相乘即可在序列中屏蔽不相关的项
        self.reduction ='none'
        unweighted_loss = super().forward(pred.permute((0, 2, 1)), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)  # 计算加权后的平均损失
        return weighted_loss

class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_inputs, decoder_inputs, *args):
        encoder_outputs = self.encoder(encoder_inputs, *args)
        decoder_state = self.decoder.init_state(encoder_outputs, *args)
        return self.decoder(decoder_inputs, decoder_state)

class PositionalEncoding(nn.Module):
    """
    位置编码:
    在完全依赖于注意力机制的模型中， 为了并行计算, 序列的位置信息会被忽略
    位置编码的目的就是给予模型词元的位置信息
    """
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        # P是位置编码矩阵, 存储序列中每个位置的位置编码信息, 这些编码之后会被添加到输入embedding中

        self.P = torch.zeros((1, max_len, num_hiddens)) # 长度为1的第一维为了广播机制
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X) # 奇数索引位置编码
        self.P[:, :, 1::2] = torch.cos(X) # 偶数索引位置编码

    def forward(self, X):
        """
        X : shape=(batch_size, seq_len, num_hiddens)
        """
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

def masked_softmax(X, valid_lens=None):
    """
    带遮蔽的softmax函数, 通过在最后一个轴上遮蔽元素来执行softmax操作\n
    Parameters:
        X : 3D张量
        valid_lens : 1D或2D张量
    """
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6) # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        return nn.functional.softmax(X.reshape(shape), dim=-1)

def bleu(pred_seq, label_seq, k):
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred)) # 初始惩罚项,用于降低较短序列(因为容易预测)的分数 公式的前半部分
    for n in range(1, min(k, len_pred)+1):
        num_matches, label_ngram_cnt = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1): # 初始化标签序列的n元语法表
            label_ngram_cnt[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1): # 查找预测序列中有多少和标签序列表匹配的n元语法
            if label_ngram_cnt[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_ngram_cnt[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n)) # p_n * 1/2^n 公式的后半部分
    return score

def sequence_mask(X, valid_lens, value=0):
    """
    将X中的每个序列(行)中在valid_lens指定位置以后的元素设为value
    Parameters:
        X : 2D张量, 每一行是一个序列
        valid_lens : 1D张量, 每一个元素表示X中每一行的有效长度
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_lens[:, None] # 应用广播机制进行比较
    # 这里, torch.arange获取了一个从0到maxlen-1的整数序列并用[None, :]升为shape=(1, maxlen)的二维矩阵。
    # valid_lens通过[:, None]升为shape=(len(valid_lens), 1)的二维矩阵。 
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

def load_array(data_arrays, batch_size, is_train=True):
    """
    将data_arrays中的array打包成TensorDataset后加载到DataLoader中\n
    参数:\n
        data_arrays : tuple(tuple)\n 每一个array的第一维长度必须一致
    返回:\n
        一个DataLoader类的data_iter
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def try_gpu(i=0):
    """如果存在,返回gpu(i), 否则返回cpu()"""
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def grad_clipping(net, theta):
    """
    进行梯度裁剪的函数\n
    参数:\n
    net : 训练过程中需要进行梯度裁剪的神经网络\n
    theta : 一个阈值,如果梯度梯度的L2范数超过了这个阈值,就将梯度缩放到这个阈值\n
    作用:防止梯度爆炸\n

    梯度爆炸:
    在训练过程中，由于网络的深度较深，反向传播时梯度在每个时间步都会累积。如果在某些层的权重初始化得不当，或者激活函数没有选择好，梯度可能会在反向传播过程中逐渐增大。
    梯度变得非常大，导致参数更新时步长过大。这会使得模型参数发生剧烈的变化，甚至导致模型无法收敛，损失函数变得无限大。
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def download(DATA_HUB, name, save_folder_name: str):
    """
    下载一个DATA_HUB中的name文件并返回本地文件名\n
    参数:\n
        save_folder_name指定存储在当前目录下的data/save_folder_name下
    """
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    cache_dir = os.path.join('data', save_folder_name)
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()  # 计算给定字符串的SHA-1哈希值
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)  # 参数:读取1MB内容
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:  # 检查哈希值判定文件是否已经存在
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(DATA_HUB, name, save_folder_name):
    """下载并解压zip/tar文件"""
    compressed_file = download(DATA_HUB, name, save_folder_name) # 下载压缩包
    base_dir = os.path.dirname(compressed_file)  # basedir为压缩包所在相对路径 dirname()获取文件的路径
    data_dir, ext = os.path.splitext(compressed_file) # splitext()将文件名与文件后缀(如.zip)分割为具有两元素的元组
    if ext == '.zip':
        fp = zipfile.ZipFile(compressed_file, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(compressed_file, 'r')
    else:
        raise ValueError('只有zip/tar文件可以被解压缩')
    fp.extractall(base_dir)  # 将压缩的文件解压到base_dir路径下
    return data_dir

def download_all(DATA_HUB):
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)

def seq2seq_train(net, data_iter, lr, num_epochs, target_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                # nn.GRU的_flat_weights_names是参数名字的集合, 例:['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0']
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_function = MaskedSoftmaxCrossEntropyLoss()
    timer = Timer()
    res = ResVisualization(xlist=[[]], ylist=[[]], legend_names=['loss in epoch'],
                                is_grid=True, title='Result', xlabel='epoch', ylabel='loss')
    net.train()
    sum_tokens = 0
    with timer:
        for epoch in range(num_epochs):
            metric = Accumulator(2) # 训练损失总和, 词元数量
            for batch in data_iter: # batch:(src_array, src_valid_len, tgt_array, tgt_valid_len)
                optimizer.zero_grad()
                X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
                # 给每组文本列添加起始标志'<bos>'
                bos = torch.tensor([target_vocab['<bos>']]*Y.shape[0], device=device).reshape(-1, 1)
                decoder_input = torch.cat([bos, Y[:, :-1]], dim=1) # 使用强制教学方法 将原始输出序列(而不是预测结果)输入解码器
                Y_hat, _ = net(X, decoder_input, X_valid_len)
                loss = loss_function(Y_hat, Y, Y_valid_len)
                loss.sum().backward()
                grad_clipping(net, 1)
                num_tokens = Y_valid_len.sum()
                optimizer.step()
                with torch.no_grad():
                    metric.add(loss.sum(), num_tokens)
                    sum_tokens += num_tokens
            if (epoch+1) % 10 == 0:
                res.add(epoch+1, metric[0]/metric[1], 'loss in epoch')
                print(f"epoch:{epoch+1}, loss:{metric[0]/metric[1]:.3f}")
    print(f"损失:{metric[0]/metric[1]:.3f}, {sum_tokens/timer.get_elapsed_time():.1f} tokens/秒 在 {str(device)}上")
    res.plot_res()

def seq2seq_predict(net, src_sentence, src_vocab, tgt_vocab, 
                    num_steps, device, save_attention_weights=False):
    """序列到序列模型的预测"""

    def truncate_pad(line, num_steps, padding_token):
            """
            通过截断(truncation)和 填充(padding)方式实现一次只处理一个小批量的文本序列。\n
            假设同一个小批量中的每个序列都应该具有相同的长度num_steps,那么如果文本序列的词元数目少于num_steps时,\n
            我们将继续在其末尾添加特定的“<pad>”词元, 直到其长度达到num_steps。\n
            反之,我们将截断文本序列时,只取其前num_steps 个词元，并且丢弃剩余的词元。\n
            这样，每个文本序列将具有相同的长度，以便以相同形状的小批量进行加载。\n
            参数:\n
                line : 一个文本序列\n
                num_steps : 同一个小批量中所有文本序列的最大长度。\n
                padding_token : 用于填充的特殊词元。\n
            返回:\n
                list : 截断或填充后的文本序列。\n
            """
            if len(line) > num_steps:
                return line[:num_steps] # 截断
            else:
                return line + [padding_token] * (num_steps - len(line)) # 填充
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']] # 给src_tokens加上'<eos>'
    encoder_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>']) # 对源词元进行截断和填充
    encoder_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0) # 添加批量轴
    encoder_outputs = net.encoder(encoder_X, encoder_valid_len)
    decoder_state = net.decoder.init_state(encoder_outputs, encoder_valid_len)
    decoder_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0) # 添加批量轴
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, decoder_state = net.decoder(decoder_X, decoder_state) # Y.shape=(1, 1, tgt_vocab_size)
        decoder_X = Y.argmax(dim=2) # 使用具有最高可能性的词元, 作为解码器在下一时间步的输入
        pred = decoder_X.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weights: # 保存注意力权重
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']: # 一旦结束词元被预测,输出序列的生成就完成了
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq