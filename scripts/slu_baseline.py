#coding=utf8
import sys, os, time, gc
from torch.optim import Adam

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from model.slu_baseline_tagging import SLUTagging

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
train_dataset = Example.load_dataset(train_path)
dev_dataset = Example.load_dataset(dev_path)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)


model = SLUTagging(args).to(device)
# print(model)
# for name, parameters in model.named_parameters():
#     print(name, ':', parameters.size())
Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)


def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer


def decode(choice):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(Example.label_vocab, current_batch)
            for j in range(len(current_batch)):
                # print(current_batch.utt[j])
                # print(pred[j])
                # print(label[j])
                # 其中一条输出：
                #   黄梅镇派出所
                #   ['inform-poi名称-所']
                #   ['inform-poi名称-黄梅镇派出所']
                if any([l.split('-')[-1] not in current_batch.utt[j] for l in pred[j]]):
                    print(current_batch.utt[j], pred[j], label[j])
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        metrics = Example.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count


if not args.testing:
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    print('Total training steps: %d' % (num_training_steps))
    optimizer = set_optimizer(model, args)
    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size
    print('Start training ......')
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_loss = 0
        np.random.shuffle(train_index)
        model.train()
        count = 0
        for j in range(0, nsamples, step_size):
            cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            # print(current_batch.utt)
            #    ['动车开始减到昌吉市区两个小时接下去可以现在不要起来观', '继续玩到体健身完成了', '我想去松桃县普觉镇', 
            #     '我要到枫韵蓝湾东门', '我要去领秀翡翠山', '安安你好取消导航', '江苏汽车技师学院', '导航到厦门鼓浪屿',
            #     '我要导航去巧什', '附近哪里有吃的', '我要去东升大厦', '导航去人民医院', '导航到永兴佳苑', '导航去厚街广场',
            #     '导航到顺欣花园', '临沂齐鲁证券', '到金玛到金源', '去月湖馨苑', '我要导航', 'null', 'null', '碧导航走',
            #     '下一页', '讲健身', '去柳溪', '第一个', '导航', '血战', '焦滩', '导航', '天津', '一']
            # print(current_batch.input_ids)
            #     tensor([[ 157,  158,   45,  335,    1,    4,  461,  579,   62,   80,  824,   12,
            #             132,  888,  643,   33,   41,  698,  699,   22,   23,  173,   40, 1572,
            #             370,  776],
            #             [ 681,  682, 1043,    4,  570,  532,  566,  573,  457,  172,    0,    0,
            #                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            #                 0,    0], ... ]
            # print(current_batch.lengths)
            #     [26, 10, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 6, 6, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1]
            # print(current_batch.labels)
            #     [['inform-终点名称-米泉'], ['inform-poi名称-健身房奥体健身奥体健身'], ['inform-终点名称-松桃县普觉镇'],
            #      ['inform-终点名称-枫韵蓝湾东门'], ['inform-终点名称-领秀翡翠山'], ['inform-操作-取消', 'inform-对象-导航'], ... ]
            # print(current_batch.tag_ids)
            # print(current_batch.tag_mask)
            output, loss = model(current_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1
        print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))
        torch.cuda.empty_cache()
        gc.collect()

        start_time = time.time()
        metrics, dev_loss = decode('dev')
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, open('./output/model/model.bin', 'wb'))
            print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
else:
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
