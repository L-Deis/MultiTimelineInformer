import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        # freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        # d_inp = freq_map[freq]
        # self.embed = nn.Linear(d_inp, d_model)
        
        # Base frequency map
        freq_map = {
            's': 6,   # seconds
            'min': 5, # minutes
            't': 5,   # time (minutes)
            'h': 4,   # hours
            'd': 3,   # days
            'b': 3,   # business days
            'w': 2,   # weeks
            'm': 1,   # months
            'a': 1,   # years (annual)
        }
        
        # Parse the freq string, e.g., "5min" -> "min"
        if isinstance(freq, str):
            # Extract letters (unit part)
            import re
            match = re.search(r'[a-zA-Z]+', freq)
            if match:
                base_freq = match.group(0)
            else:
                raise ValueError(f"Invalid freq format: {freq}")
        else:
            raise ValueError(f"freq must be a string, got {type(freq)}")
        
        # Normalize common abbreviations
        base_freq = base_freq.lower()
        
        d_inp = freq_map[base_freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)
    

class ConditionEmbedding(nn.Module):
    def __init__(self, n_cond_num_in, n_cond_num_out, n_cond_cat_in, n_cond_cat_out):
        super(ConditionEmbedding, self).__init__()

        self.n_cond_num_in = n_cond_num_in
        self.n_cond_num_out = n_cond_num_out
        self.n_cond_cat_in = [int(n) for n in n_cond_cat_in.split(',')]
        self.n_cond_cat_out = [int(n) for n in n_cond_cat_out.split(',')]
        self.n_in = n_cond_num_in + sum(self.n_cond_cat_in)
        self.n_out = n_cond_num_out + sum(self.n_cond_cat_out)

        self.cond_num = nn.Linear(n_cond_num_in, n_cond_num_out) # numerical features
        self.cond_cat = nn.ModuleList([nn.Embedding(n_in, n_out) for n_in, n_out in zip(self.n_cond_cat_in, self.n_cond_cat_out)]) # categorical features

    def forward(self, x): # x: all the first values are numerical features, and the last values are categorical features
        x_num = self.cond_num(x[:, :self.n_cond_num_in])
        x_cat = torch.cat([cond(x[:, self.n_cond_num_in+i].long()) for i, cond in enumerate(self.cond_cat)], dim=-1)

        return torch.cat([x_num, x_cat], dim=-1) #Shape: (batch_size, n_cond_num_out+sum(n_cond_cat_out))