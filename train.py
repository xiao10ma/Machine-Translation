import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from net_utils import save_model, load_model
from torchmetrics.text.bleu import BLEUScore
from network import LSTMseq2seq, GRUseq2seq
import enzh_dataset
# from net_utils import save_model, load_model
import argparse
from tqdm import tqdm
import uuid
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 400
BATCH_SIZE = 200
LR = 0.000001
MODEL = 'GRU'  # 'LSTM'

def collate_fn(batch):
    en_sentences, zh_sentences = zip(*batch)
    en_lengths = [len(seq) for seq in en_sentences]
    zh_lengths = [len(seq) for seq in zh_sentences]

    en_padded = pad_sequence([seq.clone().detach() for seq in en_sentences], batch_first=True, padding_value=0)
    zh_padded = pad_sequence([seq.clone().detach() for seq in zh_sentences], batch_first=True, padding_value=0)

    return en_padded, zh_padded, en_lengths, zh_lengths

def prepare_output_and_logger(args):    
    if not args.record_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.record_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.record_path))
    os.makedirs(args.record_path, exist_ok = True)
    with open(os.path.join(args.record_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(argparse.Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.record_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def train(tb_writer, args):
    dataset = enzh_dataset.enzhDataset()
    train_dataset = enzh_dataset.enzhWrapper(dataset, 'train')
    test_dataset = enzh_dataset.enzhWrapper(dataset, 'test')

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, collate_fn=collate_fn, drop_last=True)

    # model = LSTMseq2seq(len(dataset.en_vocab), len(dataset.zh_vocab), args.embedded_size, args.hidden_size, BATCH_SIZE).to(device)
    if MODEL == 'GRU':
        model = GRUseq2seq(len(dataset.en_vocab), len(dataset.zh_vocab), args.embedded_size, args.hidden_size, BATCH_SIZE).to(device)
    elif MODEL == 'LSTM':
        model = LSTMseq2seq(len(dataset.en_vocab), len(dataset.zh_vocab), args.embedded_size, args.hidden_size, BATCH_SIZE).to(device)
    else:
        raise Exception

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    begin_epoch, global_step = 0, 0


    for epoch in tqdm(range(begin_epoch, EPOCH)):
        for iter, batch in enumerate(train_loader):
            iter_start.record()
            optimizer.zero_grad()
            pred = model('train', batch[0].to(device), batch[1].to(device))
            eos = torch.ones((BATCH_SIZE, 1), dtype=torch.long) * (len(dataset.zh_vocab) - 1)
            zh_sen = torch.concatenate((batch[1][:, 1:], eos), dim=1)
            loss = criterion(pred.cpu(), zh_sen.reshape(-1))
            loss.backward()
            optimizer.step()
            iter_end.record()
            torch.cuda.synchronize()

            if (iter % 10000 == 0):
                global_step += 1
                bleu = evaluate(model, test_loader, len(dataset.zh_vocab))
                training_report(tb_writer, global_step, loss, iter_start.elapsed_time(iter_end), bleu)
        if (epoch + 1) % args.save_ep == 0:
            save_model(model, optimizer, args.model_path, epoch, global_step)
        if (epoch + 1) % args.save_latest_ep == 0:
            save_model(model, optimizer, args.model_path, epoch, global_step, last=True)


def evaluate(network, test_loader, zh_vocab_size):
    with torch.no_grad():
        bleu_list = []
        bleu = BLEUScore(n_gram=4, smooth=True)
        for _, batch in enumerate(test_loader):
            pred = network('test', batch[0].to(device), batch[1].to(device), 0.0)   # [batch * seq, vocab]

            eos = torch.ones((BATCH_SIZE, 1), dtype=torch.long) * (zh_vocab_size - 1)
            zh_sen = torch.concatenate((batch[1][:, 1:], eos), dim=1)       # [batch, seq]

            pred_indices = torch.argmax(pred, dim=1).reshape(BATCH_SIZE, -1)
            pred_tokens = [[test_loader.dataset.dataset.zh_idx2token[idx.item()] for idx in sentence] for sentence in pred_indices]
            ref_tokens = [[test_loader.dataset.dataset.zh_idx2token[idx.item()] for idx in sentence] for sentence in zh_sen]
            pred_strs = [' '.join(sentence) for sentence in pred_tokens]
            ref_strs = [' '.join(sentence) for sentence in ref_tokens]
            ref_strs = [[sentence] for sentence in ref_strs]

            score = bleu(pred_strs, ref_strs)
            bleu_list.append(score)
        
        bleu_score = sum(bleu_list) / len(bleu_list)
    return bleu_score

def training_report(tb_writer, iter, loss, elapsed, bleu):
    if tb_writer:
        tb_writer.add_scalar('loss' + '  lr: {}  epoch: {}'.format(LR, EPOCH), loss.item(), iter)
        # tb_writer.add_scalar('iter_time' + '  lr: {}  epoch: {}'.format(LR, EPOCH), elapsed, iter)
        tb_writer.add_scalar('bleu' + '  lr: {}  epoch: {}'.format(LR, EPOCH), bleu, iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Machine Translation')
    parser.add_argument('-r', '--record_path', default='./output/{}_200d'.format(MODEL), type=str)
    parser.add_argument('--embedded_size', default=200, type=int)
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--save_ep', default=50, type=int)
    parser.add_argument('--save_latest_ep', default=10, type=int)
    parser.add_argument('-m', '--model_path', default='./trained_model/{}_200d'.format(MODEL), type=str)
    args = parser.parse_args()

    tb_writer = prepare_output_and_logger(args)

    train(tb_writer, args)