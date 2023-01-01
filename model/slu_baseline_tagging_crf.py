#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torchcrf import CRF

class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNCRFDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        embed = self.word_embed(input_ids)
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        hiddens = self.dropout_layer(rnn_out)
        tag_output, loss = self.output_layer(hiddens, tag_mask, tag_ids)

        return tag_output, loss

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        tag_output, loss = self.forward(batch)
        predictions = []
        for i in range(batch_size):
            pred = tag_output[i]
            
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            # print(batch.utt[i])
            # pred = pred[:len(batch.utt[i])]
            # print(len(pred))
            # print(len(batch.utt[i]))
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                # print(tag)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    # print(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                # print(f'{slot}-{value}')
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if loss == None:
            return predictions, labels, None
        return predictions, labels, loss.cpu().item()


class TaggingFNNCRFDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNCRFDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def loss_func(self, logits, labels, mask):
        # print(self.crf.forward(logits, labels, mask, reduction='mean'))

        return -self.crf.forward(logits, labels, mask, reduction='mean')
        
        
        
    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        pred = self.crf.decode(logits, mask)
        # print(len(pred), len(pred[0]))  bsize x seqlen 列表里的长度和句子本身的长度是一一对应的
        
        if labels is not None:
            loss = self.loss_func(logits, labels, mask)
            return pred, loss
        return pred, None
