import sys, os, time, gc
from torch.optim import Adam
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.batch import from_example_list
from utils.example import Example
from utils.args import init_args
from utils.initialization import *
from utils.vocab import PAD
from BERT_CRF import MyExample, Model, SLUDataSet, convert_ids_to_words, tokenizer
import json
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
test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
MyExample.load_label_vocab(args.dataroot)
test_dataset = SLUDataSet(test_path)

print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: test -> %d." % (len(test_dataset)))

word_vocab = tokenizer.get_vocab()
args.vocab_size = len(word_vocab)
args.pad_idx = 0 # word_vocab['[PAD]'] = 0
args.num_tags = MyExample.label_vocab.num_tags
args.tag_pad_idx = MyExample.label_vocab.convert_tag_to_idx(PAD)
label_vocab = MyExample.label_vocab
label_vocab.tag2idx['[SPE-TOKEN]'] = 74
label_vocab.idx2tag[74] = '[SPE-TOKEN]'
args.num_tags += 1
#################
# 定义数据整理函数 #
#################
def test_collate_fn(data):
    utt_batch = [i['utt'] for i in data] 
    inputs = tokenizer.batch_encode_plus(utt_batch,
                                         padding=True,
                                         return_tensors='pt')

    return inputs, None
#数据加载器
test_loader = DataLoader(dataset=test_dataset,
                    batch_size=args.batch_size,
                    collate_fn=test_collate_fn)

checkpoint = './output/model/bert_crf_model.bin'
model = Model(args.num_tags, args.batch_size).to(device)
model.fine_tuneing(True)
# for name, parameters in model.named_parameters():
#     print(name, ':', parameters.size())
model.load_state_dict(torch.load(checkpoint)['model'])
model.eval()

predictions = []
with torch.no_grad():
    for step, (inputs, labels) in enumerate(test_loader):
        #模型计算
        #[b, lens] -> [b, lens, 8]
        inputs = inputs.to(device)
        outs, _ = model(inputs, labels)
        pred = model.decode(outs, convert_ids_to_words(inputs['input_ids']))
        predictions.extend(pred)
torch.cuda.empty_cache()
gc.collect()

new_json = json.load(open('data/test_unlabelled.json', 'r'))
for i, utt in enumerate(new_json):
    utt[0]['pred'] = predictions[i]
    
with open('output/test/BERT_CRF.json', 'w', encoding="utf8") as f:
    json.dump(new_json, f, indent=4, ensure_ascii=False)

print(predictions)