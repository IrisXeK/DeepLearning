import os, hashlib, requests, zipfile, tarfile, torch, random, text_pretreatment, time, math
import matplotlib.pyplot as plt
from torch import nn

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
    def __init__(self, xlist:tuple|list, ylist:tuple|list, legend_names:str|list, is_grid=None,
                xlabel:str=None, ylabel:str=None, title:str=None,
                xlim:list=None, ylim:list=None, line_style:str='-') -> None:
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
        if isinstance(legend_names, (tuple, list)):
            self.res_dict = {name:(x, y) for name, x, y in zip(legend_names, xlist, ylist)}
        else:
            self.res_dict = {legend_names:(xlist,ylist)}
        self.is_grid = is_grid
        self.xlim = xlim
        self.ylim  = ylim
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title  = title
        self.line_style = line_style

    def add(self, x_val, y_val, name):
        """向名为name的曲线中添加一个(x_val, y_val)数据对"""
        self.res_dict[name][0].append(x_val)
        self.res_dict[name][1].append(y_val)

    def plot_res(self):
        for name, xy_pair in self.res_dict.items():
            plt.plot(xy_pair[0], xy_pair[1], label=name, linestyle=self.line_style)
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
            raise ValueError("Timer has not been started.")
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self.elapsed_time_sum += self.elapsed_time

    def get_elapsed_time(self):
        if self.elapsed_time is None:
            raise ValueError("Timer has not been stopped yet.")
        return self.elapsed_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        # print(f"Elapsed time: {self.get_elapsed_time():.4f} seconds")

def download(DATA_HUB, name, save_folder_name:str): # save_folder_name指定存储在当前目录下的data/save_folder_name下
    """下载一个DATA_HUB中的文件并返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    cache_dir=os.path.join('data', save_folder_name)
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1() # 计算给定字符串的SHA-1哈希值
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576) # 参数:读取1MB内容
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash: # 检查哈希值判定文件是否已经存在
            return fname # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname) # dirname(name)查询name文件所在的文件夹的路径
    data_dir, ext = os.path.splitext(fname) # splitext 将文件名与文件后缀(如.zip)分割为具有两元素的元组
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir) # 将压缩的文件解压到base_dir路径下
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all(DATA_HUB):
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)

def try_gpu(i=0):
    """如果存在,返回gpu(i), 否则返回cpu()"""
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def sgd(params:list, lr, batch_size):
    """
    小批量梯度下降优化函数\n
    参数:\n
    params : 模型的所有可学习的参数的列表\n
    lr : 学习率\n
    batch_size : 批量大小\n
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size # 更新参数
            param.grad.zero_() # 清除累积的梯度

def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

class SeqDataLoader:
    """
    加载序列数据的迭代器
    """
    def __init__(self, batch_size, num_steps, max_tokens, use_random_iter) -> None:
        if use_random_iter: 
            self.data_iter_fn = self.get_random_batch_seq
        else: 
            self.data_iter_fn = self.get_sequential_batch_seq
        self.corpus, self.vocab = text_pretreatment.load_time_machine_corpus(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
    
    def get_random_batch_seq(self, corpus, batch_size, num_steps):
        """
        使用随机抽样生成一个样本批量\n
        参数:\n
        corpus : 语料库
        batch_size : 一个小批量中有多少个子序列样本
        num_steps : 每个序列预定义的时间步\n
        返回:\n
        X : 特征, shape=(batch_size, num_steps)
        Y : 标签, shape=(batch_size, num_steps)
        """
        def get_seq(pos):
            """
            返回从pos位置开始的长度为num_steps的序列\n
            pos : 一个偏移量
            """
            return corpus[pos: pos + num_steps]

        corpus = corpus[random.randint(0, num_steps - 1):] # 随机选择起始分区的偏移量,随机范围包括num_steps-1  减去1是因为需要考虑标签
        num_subseqs = (len(corpus) - 1) // num_steps # 整个语料库可划分出的子序列的数量
        initial_indices = list(range(0, num_subseqs * num_steps, num_steps)) # 长度为num_step的每个子序列的起始索引
        random.shuffle(initial_indices) #  在随机抽样的迭代过程中,来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
        num_batches = num_subseqs // batch_size # 所有子序列可被分成的小批量的数量 即以batch_size个样本为一批,可分出多少批
        for i in range(0, batch_size * num_batches, batch_size): # 迭代小批量
            initial_indices_per_batch = initial_indices[i : i+batch_size] # 得到一批中所有样本的起始索引
            X = [get_seq(j) for j in initial_indices_per_batch] # 根据起始索引依次获得一批中的样本 X.shape=(batch_size, num_steps)
            Y = [get_seq(j+1) for j in initial_indices_per_batch] # 根据起始索引依次获得一批中的样本的标签
            yield torch.tensor(X), torch.tensor(Y) # 特征 和 对应的标签

    def get_sequential_batch_seq(self, corpus, batch_size, num_steps):
        """
        使用顺序分区生成一个样本批量\n
        参数:\n
        corpus : 语料库
        batch_size : 一个小批量中有多少个子序列样本
        num_steps : 每个序列预定义的时间步数\n
        返回:\n
        X : 特征, shape=(batch_size, num_steps)
        Y : 标签, shape=(batch_size, num_steps)
        """
        offset = random.randint(0, num_steps-1) # 用随机偏移量划分序列
        num_tokens = ((len(corpus)-offset-1) // batch_size) * batch_size # 得到正好的token数, 将不能完整组成一批的token舍弃
        Xs = torch.tensor(corpus[offset : offset + num_tokens])
        Ys = torch.tensor(corpus[offset+1 : offset + num_tokens + 1])
        Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
        num_batches = Xs.shape[1] // num_steps # 小批量的个数(纵向分割出一个个batch)
        for i in range(0, num_batches*num_steps, num_steps):
            X = Xs[: ,i:i + num_steps] # 特征
            Y = Ys[: ,i:i + num_steps] # 标签
            yield X, Y # shape=(batch_size, num_steps)
    
def load_time_machine_data(batch_size, num_steps, 
                           max_tokens=10000, use_random_iter=False):
    """
    返回时光机器数据集的 迭代器、词表
    """
    data_iter = SeqDataLoader(batch_size, num_steps, max_tokens, use_random_iter)
    return data_iter, data_iter.vocab

def predict_rnn(prefix, num_preds, net, vocab, device):
    """
    这个函数用于在prefix后面生成新字符\n
    prefix : 一个用户提供的包含多个字符的字符串\n
    在循环遍历prefix中的开始字符时,不断地将隐状态传递到下一个时间步，但是不生成任何输出。称为预热(warm-up)期,
    在此期间模型会自我更新(例如，更新隐状态),但不会进行预测。
    预热期结束后，隐状态的值通常比刚开始的初始值更适合预测，从而预测字符并输出它们。
    """
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]: # 预热期
        _, state = net(get_input(), state) # 更新隐状态
        outputs.append(vocab[y])
    for _ in range(num_preds): # 预测num_preds步
        y, state = net(get_input(), state) # 预测y并更新隐状态 相当于(batch_size, num_step)=(1,1)的单步预测
        outputs.append(int(y.argmax(dim=1).reshape(1))) # argmax输出的是列表 所以需要reshape
    return  ''.join([vocab.idx_to_token[i] for i in outputs])

def rnn_train_epoch(net, train_iter, loss_function, updater, device, use_random_iter):
    """
    训练模型的一个迭代周期\n
    当使用顺序分区时，只在每个迭代周期的开始位置初始化隐状态。
    由于下一个小批量数据中的第i个子序列样本与当前第i个子序列样本相邻,
    因此当前小批量数据最后一个样本的隐状态，将用于初始化下一个小批量数据第一个样本的隐状态。
    这样，存储在隐状态中的序列的历史信息可以在一个迭代周期内流经相邻的子序列。
    然而，在任何一点隐状态的计算，都依赖于同一迭代周期中前面所有的小批量数据，这使得梯度计算变得复杂。
    为了降低计算量，在处理任何一个小批量数据之前，要先分离梯度，使得隐状态的梯度计算总是限制在一个小批量数据的时间步内。
    (当使用随机抽样时,因为每个样本都是在一个随机位置抽样的,因此需要为每个迭代周期重新初始化隐状态。)
    """
    state, timer = None, Timer()
    metric = Accumulator(2) # 训练损失之和 与 词元数量
    with timer:
        for X,Y in train_iter:
            if state is None or use_random_iter: # 在第一次迭代或使用随机抽样时初始化state
                state = net.begin_state(batch_size=X.shape[0], device=device)
            else: # 剥离梯度
                if isinstance(net, nn.Module) and not isinstance(state, tuple): # state对于nn.GRU是个张量
                    state.detach_()
                else: # state对于nn.LSTM或从头实现的模型是一个张量
                    for s in state: s.detach_()
            y = Y.T.reshape(-1) # 展平为len=num_steps*batch_size的向量,以便nn.CrossEntropyLoss处理
            X, y = X.to(device), y.to(device)
            y_hat, state = net(X, state) # y_hat.shape=(num_steps*batch_size, vocab_size)
            loss = loss_function(y_hat, y.long()).mean() # .long将tensor的类型转化为torch.int64
            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                loss.backward()
                grad_clipping(net, theta=1) # 梯度裁减
                updater.step()
            else:
                loss.backward()
                grad_clipping(net, theta=1)
                updater(batch_size=1)
            metric.add(loss*y.numel(), y.numel())
    return math.exp(metric[0]/metric[1]), metric[1]/timer.elapsed_time # 返回一次迭代的 困惑度 和 训练速度

def rnn_train(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """训练模型"""
    loss_function = nn.CrossEntropyLoss()
    res = ResVisualization(xlist=[], ylist=[], legend_names=('train'),
                           xlabel='epoch', ylabel='perplexity', title='train_res')
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size : sgd(net.params, lr, batch_size)
    predict = lambda prefix : predict_rnn(prefix, 50, net, vocab, device)    
    for epoch in range(num_epochs):
        perplexity, train_speed = rnn_train_epoch(net, train_iter, loss_function, updater, device, use_random_iter)
        if (epoch+1) % 100 == 0:
            print(f"epoch: {epoch+1}, 对'time traveller'的预测:{predict(prefix='time traveller')}")
        res.add(epoch+1, perplexity, 'train')
    print(f"困惑度{perplexity:.2f}, {train_speed:.1f}词元/秒 在{str(device)}上")
    print(f"对prefix为'time traveller'的预测:{predict(prefix='time traveller')}")
    print(f"对prefix为'traveller'的预测:{predict(prefix='traveller')}")
    res.plot_res()