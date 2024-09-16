import torch, math
from torch import nn
from utils import masked_softmax

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

class MultiHeadAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens,
                 num_heads, dropout, use_bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention_function = ScaledDotProductAttention(dropout=dropout)
        self.W_qh = nn.Linear(query_size, num_hiddens, bias=use_bias)
        self.W_kh = nn.Linear(key_size, num_hiddens, bias=use_bias)
        self.W_vh = nn.Linear(value_size, num_hiddens, bias=use_bias)
        self.W_out = nn.Linear(num_hiddens, num_hiddens, bias=use_bias)

    def forward(self, queries, keys, values, valid_lens):
        """
        queries, keys, values的形状: (batch_size, 查询或者“键－值”对的个数, num_hiddens)
        valid_lens的形状: (batch_size, )或(batch_size, 查询的个数)
        """
        # 经过变换后, 输出的queries, keys, values　的形状:
        # (batch_size*num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
        queries = self.transpose_qkv(self.W_qh(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_kh(keys), self.num_heads)
        values = self.transpose_qkv(self.W_vh(values), self.num_heads)
        if valid_lens is not None:
            # 在轴0, 将第一项（标量或者矢量）复制num_heads次, 
            # 然后如此复制第二项, 然后诸如此类。
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        # output的形状:(batch_size*num_heads, 查询的个数, num_hiddens/num_heads)
        output = self.attention_function(queries, keys, values, valid_lens)
        # output_concat的形状:(batch_size, 查询的个数, num_hiddens)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_out(output_concat)

    def transpose_qkv(self, X, num_heads):
        """
        为了多注意力头的并行计算而变换形状
        输入X的形状:(batch_size, 查询或者“键－值”对的个数, num_hiddens)
        """
        # 输出X的形状:(batch_size, 查询或者“键－值”对的个数, num_heads, num_hiddens/num_heads)
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1) # 为每个注意力头分配隐藏单元
        # 输出X的形状:(batch_size, num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3) # 为能在每个头上并行计算更改形状
        # 最终输出的形状:(batch_size*num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])
    
    def transpose_output(self, X, num_heads):
        """逆转transpose_qkv函数的操作, 将多个注意力头结合成一个tensor"""
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2]) # 分离num_heads和batch_size维度
        X = X.permute(0, 2, 1, 3) # 调换num_heads和查询或“键－值”对的维度
        return X.reshape(X.shape[0], X.shape[1], -1) # 结合注意力头