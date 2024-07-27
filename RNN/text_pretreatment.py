import re # Python正则表达式库
import utils, collections

DATA_HUB = dict() # 存储文件路径及文件对应的SHA-1哈希值
# 读取数据集
DATA_HUB['time_machine'] = (utils.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')
def read_time_machine(): # 将time_machine的内容加载到文本的行的列表(文本的一行是列表的一个元素)中
    with open(utils.download(DATA_HUB, name='time_machine', save_folder_name='time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines] # 忽略标点符号和字母大写

# 词元(token, 是文本的基本单位)化
def tokenize(lines:list, token='word'): 
    """将文本拆分为单词词元或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        raise ValueError("未知词元类型:" + token)

# 构建方便模型使用的词表
"""
构建一个字典,通常也叫做词表(vocabulary)用来将字符串类型的词元映射到从0开始的数字索引中。
先将训练集中的所有文档合并在一起,对它们的唯一词元进行统计,得到的统计结果称之为语料(corpus)。
然后根据每个唯一词元的出现频率，为其分配一个数字索引。很少出现的词元通常被移除，这可以降低复杂性。
另外，语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元“<unk>”。
可以选择增加一个列表，用于保存那些被保留的词元，例如：填充词元"<pad>", 序列开始词元("<bos>"), 序列结束词元"<eos>"）。
"""
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
        return [self.__getitem__(token) for token in tokens] # 以多个键方式返回若干个tokend的索引列表
    
    @property
    def num_tokens(self):
        return len(self)
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self):
        return 0 # 未知的单词索引为0
    
    @property
    def token_freqs(self):
        return self._token_freqs

def load_time_machine_corpus(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, token='char') #  为简化后面的训练,使用字符实现文本词元化
    vocab = Vocabulary(tokens) # 使用time_machine数据集作为语料库构建词表
    # 因为《时光机器》数据集中的每个文本行不一定是一个句子或一个段落，所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens] # 截断为前max_tokens个词元
    return corpus, vocab