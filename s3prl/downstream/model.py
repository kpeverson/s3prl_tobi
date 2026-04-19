import torch
import torch.nn as nn
import torch.nn.functional as F


def get_downstream_model(input_dim, output_dim, config):
    model_cls = eval(config['select'])
    model_conf = config.get(config['select'], {})
    model = model_cls(input_dim, output_dim, **model_conf)
    return model


class FrameLevel(nn.Module):
    def __init__(self, input_dim, output_dim, hiddens=None, activation='ReLU', **kwargs):
        super().__init__()
        latest_dim = input_dim
        self.hiddens = []
        if hiddens is not None:
            for dim in hiddens:
                self.hiddens += [
                    nn.Linear(latest_dim, dim),
                    getattr(nn, activation)(),
                ]
                latest_dim = dim
        self.hiddens = nn.Sequential(*self.hiddens)
        self.linear = nn.Linear(latest_dim, output_dim)

    def forward(self, hidden_state, features_len=None):
        hidden_state = self.hiddens(hidden_state)
        logit = self.linear(hidden_state)

        return logit, features_len


class UtteranceLevel(nn.Module):
    def __init__(self,
        input_dim,
        output_dim,
        pooling='MeanPooling',
        activation='ReLU',
        pre_net=None,
        post_net={'select': 'FrameLevel'},
        **kwargs
    ):
        super().__init__()
        latest_dim = input_dim
        self.pre_net = get_downstream_model(latest_dim, latest_dim, pre_net) if isinstance(pre_net, dict) else None
        self.pooling = eval(pooling)(input_dim=latest_dim, activation=activation)
        self.post_net = get_downstream_model(latest_dim, output_dim, post_net)

    def forward(self, hidden_state, features_len=None):
        if self.pre_net is not None:
            hidden_state, features_len = self.pre_net(hidden_state, features_len)

        pooled, features_len = self.pooling(hidden_state, features_len)
        logit, features_len = self.post_net(pooled, features_len)

        return logit, features_len

class UtteranceLevelBeforeAfter(nn.Module):
    def __init__(self,
        input_dim,
        output_dim,
        pooling='MeanPooling',
        activation='ReLU',
        pre_net=None,
        post_net={'select': 'FrameLevel'},
        **kwargs
    ):
        super().__init__()
        latest_dim = input_dim
        self.pre_net_before = get_downstream_model(latest_dim, latest_dim, pre_net) if isinstance(pre_net, dict) else None
        self.pre_net_after = get_downstream_model(latest_dim, latest_dim, pre_net) if isinstance(pre_net, dict) else None
        self.pooling_before = eval(pooling)(input_dim=latest_dim, activation=activation)
        self.pooling_after = eval(pooling)(input_dim=latest_dim, activation=activation)
        self.post_net = get_downstream_model(latest_dim*2, output_dim, post_net)

    def forward(self, hidden_state, features_len):
        assert isinstance(hidden_state, (list, tuple)) and len(hidden_state) == 2, \
            f"hidden_state should be a list or tuple of length 2, got {type(hidden_state)} with length {len(hidden_state)}"
        assert isinstance(features_len, (list, tuple)) and len(features_len) == 2, \
            f"features_len should be a list or tuple of length 2, got {type(features_len)} with length {len(features_len)}"
        if self.pre_net_before is not None and self.pre_net_after is not None:
            hidden_state[0], features_len[0] = self.pre_net_before(hidden_state[0], features_len[0])
            hidden_state[1], features_len[1] = self.pre_net_after(hidden_state[1], features_len[1])

        pooled_before, features_len[0] = self.pooling_before(hidden_state[0], features_len[0])
        pooled_after, features_len[1] = self.pooling_after(hidden_state[1], features_len[1])
        pooled = torch.cat([pooled_before, pooled_after], dim=-1)
        features_len = [f1+f2 for f1, f2 in zip(features_len[0], features_len[1])]
        logit, features_len = self.post_net(pooled, features_len)

        return logit, features_len

class MeanPooling(nn.Module):

    def __init__(self, **kwargs):
        super(MeanPooling, self).__init__()

    def forward(self, feature_BxTxH, features_len, **kwargs):
        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            features_len  - [B] of feature length
        '''
        # zero out padded positions
        device = feature_BxTxH.device
        len_masks = torch.lt(torch.arange(features_len.max()).unsqueeze(0).to(device), features_len.unsqueeze(1))
        len_masks = len_masks.unsqueeze(-1).float()
        feature_BxTxH = feature_BxTxH * len_masks

        # sum and divide by lengths
        feature_mean = torch.sum(feature_BxTxH, dim=1) / features_len.unsqueeze(-1).float()
        return feature_mean, torch.ones(len(feature_BxTxH)).long()

class SelfAttentionCLSPooling(nn.Module):
    ''' Self Attention Pooling consisting of a single multi-head self-attention layer and a CLS token '''

    def __init__(self, input_dim, **kwargs):
        super(SelfAttentionCLSPooling, self).__init__()
        self.cls_embed = nn.Parameter(
            torch.FloatTensor(input_dim).uniform_()
        )
        self.mha = nn.MultiheadAttention(
            input_dim, 1, batch_first=True
        )

    def forward(self, feature_BxTxH, att_mask):
        # append cls embedding to beginning
        feature_BxTxH = torch.cat(
            [
                self.cls_embed.expand(feature_BxTxH.size(0), 1, -1),
                feature_BxTxH,
            ], dim=1
        )
        # pass thru mha layer
        feature = self.mha(
            feature_BxTxH, feature_BxTxH, feature_BxTxH,
        )[0][:, 0, :]

        return feature, torch.ones(len(feature_BxTxH)).long()

class AttentivePooling(nn.Module):
    ''' Attentive Pooling module incoporate attention mask'''

    def __init__(self, input_dim, activation, fixed=True, **kwargs):
        super(AttentivePooling, self).__init__()
        self.sap_layer = AttentivePoolingModule(input_dim, activation)

    def forward(self, feature_BxTxH, features_len):
        ''' 
        Arguments
            feature_BxTxH - [BxTxH]   Acoustic feature with shape 
            features_len  - [B] of feature length
        '''
        device = feature_BxTxH.device
        len_masks = torch.lt(torch.arange(features_len.max()).unsqueeze(0).to(device), features_len.unsqueeze(1))
        sap_vec, _ = self.sap_layer(feature_BxTxH, len_masks)

        return sap_vec, torch.ones(len(feature_BxTxH)).long()

class AttentivePoolingModule(nn.Module):
    """
    Implementation of AttentivePoolingModule with correct masking
    """
    def __init__(self, input_dim, activation='ReLU', **kwargs):
        super(AttentivePoolingModule, self).__init__()
        self.W_a = nn.Linear(input_dim, input_dim)
        self.W = nn.Linear(input_dim, 1)
        self.act_fn = getattr(nn, activation)()
        self.softmax = nn.functional.softmax
        
    def forward(self, batch_rep, att_mask):
        """
        input:
            batch_rep : size (B, T, H), B: batch size, T: sequence length, H: Hidden dimension
            att_mask : size (B, T), with True on valid positions and False on padded positions
            
        return:
            utter_rep : size (B, H)
            att_w : size (B, T, 1)
        """
        att_logits = self.W(self.act_fn(self.W_a(batch_rep))).squeeze(-1)
        att_logits = att_logits.masked_fill(~att_mask, float('-inf'))
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep, att_w