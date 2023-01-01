import sys, os, time, gc
from torch.optim import Adam
import torch
import torch.nn as nn
install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)
from model.slu_baseline_tagging_crf import SLUTagging
from utils.batch import from_example_list
from utils.example import Example
from utils.args import init_args
from utils.initialization import *
from utils.vocab import PAD

args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
test_path = os.path.join('./data', 'test_unlabelled.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
dataset = Example.load_dataset(test_path)
device = 'cuda:0'

args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)

model = SLUTagging(args).to(device)
model.load_state_dict(torch.load('./output/model/crf_model.bin')['model'])
model.eval()

Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)

predictions = []
with torch.no_grad():
    for i in range(0, len(dataset), 64):
        cur_dataset = dataset[i: i + 64]
        current_batch = from_example_list(args, cur_dataset, device, train=False)
        pred, _, _ = model.decode(Example.label_vocab, current_batch)
        predictions.extend(pred)
torch.cuda.empty_cache()
gc.collect()
print(predictions)