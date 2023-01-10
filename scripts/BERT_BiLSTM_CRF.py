import torch
from transformers import BertTokenizer, AutoModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchcrf import CRF
import sys, os, time, gc

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)
from utils.vocab import *
from utils.evaluator import Evaluator
from utils.args import init_args
from utils.initialization import *
from utils.vocab import PAD

# 选择要加载的预训练语言模型
checkpoint = "hfl/chinese-bert-wwm-ext"
#加载预训练字典和分词方法
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path=checkpoint,
    cache_dir=None,
    force_download=False,
)


############
# 构建数据集 #
############
class MyExample():
    @classmethod 
    def load_label_vocab(cls, root):
        cls.label_vocab = LabelVocab(root)
    
    @classmethod
    def load_dataset(cls, data_path):
        datas = json.load(open(data_path, 'r'))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict):
        super(MyExample, self).__init__()
        self.ex = ex
        self.utt = ex['asr_1best'] #这个是句子
        self.slot = {}
        if 'semantic' in ex.keys():
            for label in ex['semantic']:
                act_slot = f'{label[0]}-{label[1]}'
                if len(label) == 3:
                    self.slot[act_slot] = label[2]
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        l = MyExample.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags] # 这个是句子中每个词对应的 label 的 id
        
class SLUDataSet(Dataset):
    def __init__(self, data_path):
        self.dataset = MyExample.load_dataset(data_path)
    def __getitem__(self, index):
        return {'utt' : self.dataset[index].utt, 'labels' : self.dataset[index].tag_id}
    def __len__(self):
        return len(self.dataset)

############
# 初始化参数 #
############
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()    
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
MyExample.load_label_vocab(args.dataroot)
train_dataset = SLUDataSet(train_path)
dev_dataset = SLUDataSet(dev_path)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

word_vocab = tokenizer.get_vocab()
args.vocab_size = len(word_vocab)
args.pad_idx = 0 # word_vocab['[PAD]'] = 0
args.num_tags = MyExample.label_vocab.num_tags
args.tag_pad_idx = MyExample.label_vocab.convert_tag_to_idx(PAD)

#################
# 定义数据整理函数 #
#################
def collate_fn(data):
    utt_batch = [i['utt'] for i in data]
    labels_batch = [i['labels'] for i in data]

    inputs = tokenizer.batch_encode_plus(utt_batch,
                                         padding=True,
                                         return_tensors='pt')

    lens = inputs['input_ids'].shape[1]

    for i in range(len(labels_batch)):
        labels_batch[i] = [74] + labels_batch[i]
        labels_batch[i] += [74] * lens
        labels_batch[i] = labels_batch[i][:lens]

    return inputs, torch.LongTensor(labels_batch)

#数据加载器
loader = DataLoader(dataset=train_dataset,
                    batch_size=args.batch_size,
                    collate_fn=collate_fn,
                    shuffle=True,
                    drop_last=True)

#查看数据样例
# for i, (inputs, labels) in enumerate(loader):
#     break

# print(len(loader)) #一共有多少个 mini-batch
# for i in range(32):
#     print(tokenizer.decode(inputs['input_ids'][i]))
#     print(labels[i])

# for k, v in inputs.items():
#     print(k, v.shape)

label_vocab = MyExample.label_vocab
label_vocab.tag2idx['[SPE-TOKEN]'] = 74
label_vocab.idx2tag[74] = '[SPE-TOKEN]'
args.num_tags += 1

#加载预训练模型
pretrained = AutoModel.from_pretrained(checkpoint,
                                       id2label=label_vocab.idx2tag,
                                       label2id=label_vocab.tag2idx).to(device)

#统计参数量
# print(sum(i.numel() for i in pretrained.parameters()) / 10000)

#模型试算
# [b, lens] -> [b, lens, 768]
# embeddings = pretrained(**inputs).last_hidden_state
# print(embeddings.shape)

##############
# 定义下游模型 #
##############
class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tuneing = False
        self.pretrained = None

        self.config = config
        self.cell = config.encoder_cell
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNCRFDecoder(config.hidden_size, config.num_tags) 
    
    def forward(self, inputs, labels=None):
        
        if self.tuneing:
            embeddings =  self.pretrained(**inputs).last_hidden_state
        else:
            with torch.no_grad():
                embeddings =  pretrained(**inputs).last_hidden_state
        
        mask = inputs['attention_mask'].type(torch.uint8)
        hiddens, h_t_c_t = self.rnn(embeddings)  # bsize x seqlen x dim
        pred, loss = self.output_layer(hiddens, mask, labels)

        return pred, loss
    
    def fine_tuneing(self, tuneing):
        self.tuneing = tuneing
        if tuneing:
            for i in pretrained.parameters():
                i.requires_grad = True
            pretrained.train()
            self.pretrained = pretrained
        else:
            for i in pretrained.parameters():
                i.requires_grad_(False)
            pretrained.eval()
            self.pretrained = None
            
    def decode(self, tag_outputs, utts_list):
        predictions = []
        for i in range(self.config.batch_size):
            pred = tag_outputs[i]
            
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([utts_list[i][j] for j in idx_buff])
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
                value = ''.join([utts_list[i][j] for j in idx_buff])
                # print(f'{slot}-{value}')
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        return predictions
    
class TaggingFNNCRFDecoder(nn.Module):
    def __init__(self, input_size, num_tags):
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
        # print(len(pred), len(pred[4])) # bsize x seqlen 列表里的长度和句子本身的长度是一一对应的
        if labels is not None:
            loss = self.loss_func(logits, labels, mask)
            return pred, loss
        return pred, None
    
def compute_metrics(predictions, labels):
    evaluator = Evaluator()
    metrics = evaluator.acc(predictions, labels)
    return {
        "precision": metrics["fscore"]["precision"],
        "recall": metrics["fscore"]["recall"],
        "f1": metrics["fscore"]["fscore"],
        "accuracy": metrics["acc"],
    }

# 把 labels 的 padding 移出
def remove_pad(labels, mask):
    new_labels = []
    for i in range(mask.shape[0]):
        selector = mask[i] == 1
        # print(selector)
        # for i in range(len(selector)):
        new_labels.append(labels[i][selector].cpu().numpy().tolist())
    return new_labels

# 把 inputs(二维的 tensor，每一行是一个带 padding 的句子）（id的形式)
# 转化为不带 padding 的句子（ word 的形式 ）
def convert_ids_to_words(inputs):
    utts_list = []
    for i in range(inputs.shape[0]):
        decode_str = tokenizer.decode(inputs[i])
        word_list = decode_str.split(' ')
        new_word_list = []
        for word in word_list:
            if word != '[PAD]':
                new_word_list.append(word)
        # utt = ''.join(new_word_list)
        utts_list.append(new_word_list)
    return utts_list

###########
# 跑验证集 #
###########
def validation():
    model.eval()
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.batch_size,
                            collate_fn=collate_fn,
                            drop_last=True)
    all_predictions, all_labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(dev_loader):
            #模型计算
            #[b, lens] -> [b, lens, 8]
            inputs = inputs.to(device)
            labels = labels.to(device)
            outs, loss = model(inputs, labels)
            pred = model.decode(outs, convert_ids_to_words(inputs['input_ids']))
            remove_pad_label_ids = remove_pad(labels, inputs['attention_mask'])
            remove_pad_label = model.decode(remove_pad_label_ids, convert_ids_to_words(inputs['input_ids']))
            # print(pred, label) # 都是以 [ [‘inform-操作-导航’, 'inform-终点名称-贵州省炉盘水市水城县', 'inform-终点名称-高速路口'], 
            #                              [ ], ... ] 的形式出现的
            # exit()
            
            all_predictions.extend(pred)
            all_labels.extend(remove_pad_label)
            total_loss += loss.item()
            count += 1
        metrics = compute_metrics(all_predictions, all_labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count

#训练
def train(epochs):
    lr = 3e-5 if model.tuneing else 3e-3
    optimizer = AdamW(model.parameters(), lr=lr)
    # criterion = torch.nn.CrossEntropyLoss()

    best_result = {'dev_acc': 0., 'dev_f1': 0.}
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        count = 0
        for step, (inputs, labels) in enumerate(loader):
            #模型计算
            #[b, lens] -> [b, lens, 8]
            # 输入也要放到 GPU 上
            inputs = inputs.to(device)
            labels = labels.to(device)
            outs, loss = model(inputs, labels)
            

            #梯度下降
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            count += 1
        print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (epoch, time.time() - start_time, epoch_loss / count))
        torch.cuda.empty_cache()
        gc.collect()
        
        start_time = time.time()
        metrics, dev_loss = validation()
        dev_acc = metrics['accuracy']
        print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (epoch, time.time() - start_time, dev_acc, metrics['precision'], metrics['recall'], metrics['f1']))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc,  metrics['f1'], epoch
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, open('./output/model/bert_bilstm_crf_model.bin', 'wb'))
            print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (epoch, dev_loss, dev_acc, metrics['precision'], metrics['recall'], metrics['f1']))
    try:
        print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
    except:
        pass
    
model = Model(args).to(device)

model.fine_tuneing(False)
train(5)
model.fine_tuneing(True)
train(4)
