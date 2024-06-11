import torch
from torch import nn
import torch.nn.functional as F
import heapq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Beam:
    def __init__(self, beam_width):
        self.heap = list() #保存数据的位置
        self.beam_width = beam_width #保存数据的总数

    def add(self,probility,complete,seq,decoder_input,decoder_hidden):
        """
        添加数据，同时判断总的数据个数，多则删除
        :param probility: 概率乘积
        :param complete: 最后一个是否为EOS
        :param seq: list，所有token的列表
        :param decoder_input: 下一次进行解码的输入，通过前一次获得
        :param decoder_hidden: 下一次进行解码的hidden，通过前一次获得
        :return:
        """
        heapq.heappush(self.heap,[probility,complete,seq,decoder_input,decoder_hidden])
        #判断数据的个数，如果大，则弹出。保证数据总个数小于等于3
        if len(self.heap)>self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):#让该beam能够被迭代
        return iter(self.heap)

class GRUseq2seq(torch.nn.Module):
    def __init__(self, en_vocab_size, zh_vocab_size, embedded_size, hidden_size, batch_size):
        super(GRUseq2seq, self).__init__()
        self.zh_vocab_size = zh_vocab_size
        self.batch_size = batch_size
        self.embedded_size = embedded_size
        self.hidden_size = hidden_size
        self.device = device

        self.en_embedder = nn.Embedding(en_vocab_size, embedded_size)
        self.zh_embedder = nn.Embedding(zh_vocab_size, embedded_size)

        self.encoder = nn.GRU(input_size=embedded_size, hidden_size=hidden_size, num_layers=2, bidirectional=False, batch_first=True)
        self.decoder = nn.GRU(input_size=embedded_size, hidden_size=hidden_size, num_layers=2, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(embedded_size, zh_vocab_size)

        self.h0 = torch.randn((2, self.batch_size, hidden_size), device=device)

    def forward(self, mode, en_sen, zh_sen, teacher_forcing_ratio=0.5, k=2):
        emebedded_enseq = self.en_embedder(en_sen)
        en_output, hidden = self.encoder(emebedded_enseq, self.h0)

        # 处理中文句子，前面加入['sos']，并去除最后一个
        sos = torch.ones((self.batch_size, 1), dtype=torch.long, device=device) * (self.zh_vocab_size - 2)
        zh_sen = torch.concatenate((sos, zh_sen[:, :-1]), dim=1)
        emebedded_zhseq = self.zh_embedder(zh_sen)  # [batch_size, seq_len, embedded_size]
        seq_len = emebedded_zhseq.shape[1]

        outputs = []
        # if mode == 'test':
        #     max_seq_len = 60
        #     self.beam_search(en_output, hidden, k, max_seq_len)
        for t in range(seq_len):
            # 采用gt，(当teacher_forcing_ratio=0.0时，即位test mode)
            if t == 0 or torch.rand(1).item() < teacher_forcing_ratio:
                input_t = emebedded_zhseq[:, t, :].unsqueeze(1)
            else:
                input_t = output_t
            output_t, hidden = self.decoder(input_t, hidden)
            scores = torch.matmul(en_output, output_t.permute(0, 2, 1)) # [batch_size, seq, hidden_size] @ [batch_size, hidden_size, 1] = [batch_size, seq, 1]
            attention_weights = F.softmax(scores, dim=1)
            weighted_sum = torch.sum(en_output * attention_weights, dim=1).unsqueeze(1)
            output_t = torch.concatenate((output_t, weighted_sum), dim=-1)

            outputs.append(output_t)

        de_output = torch.cat(outputs, dim=1)
        output = self.fc(de_output.reshape(-1, self.embedded_size))

        return output

    def beam_search(self, encoder_outpus, encoder_hidden, k, max_seq_len):
        # 1. 构造第一次需要的输入数据，保存在堆中
        decoder_input = torch.ones((self.batch_size, 1), dtype=torch.long, device=device) * (self.zh_vocab_size - 2)
        decoder_hidden = encoder_hidden #需要输入的hidden

        prev_beam = Beam(k)
        prev_beam.add(1, False, [decoder_input], decoder_input, decoder_hidden)
        for _ in range(max_seq_len):
            cur_beam = Beam(k)
            # 2. 取出堆中的数据，进行forward_step的操作，获得当前时间步的output，hidden
            # 这里使用下划线进行区分
            for _probility,_complete,_seq,_decoder_input,_decoder_hidden in prev_beam:
                #判断前一次的_complete是否为True，如果是，则不需要forward
                #有可能为True，但是概率并不是最大
                if _complete == True:
                    cur_beam.add(_probility,_complete,_seq,_decoder_input,_decoder_hidden)
                else:
                    _decoder_input = self.zh_embedder(_decoder_input)
                    decoder_output_t, decoder_hidden = self.decoder(_decoder_input, _decoder_hidden)
                    scores = torch.matmul(encoder_outpus, decoder_output_t.permute(0, 2, 1))
                    attention_weights = F.softmax(scores, dim=1)
                    weighted_sum = torch.sum(encoder_outpus * attention_weights, dim=1).unsqueeze(1)
                    decoder_output_t = torch.concatenate((decoder_output_t, weighted_sum), dim=-1)
                    decoder_output_t = self.fc(decoder_output_t)
                    value, index = torch.topk(decoder_output_t, k)
                #3. 从output中选择topk（k=beam width）个输出，作为下一次的input    
                    for i in range(k):
                        decoder_input = index[:, :, i]
                        seq = _seq + [decoder_input]
                        probility = _probility * value[:, :, i]
                        complete = index[:, :, i] == self.zh_vocab_size - 1

                        cur_beam.add(probility, complete, seq, decoder_hidden, decoder_hidden)

            #5. 获取新的堆中的优先级最高（概率最大）的数据，判断数据是否是EOS结尾或者是否达到最大长度，如果是，停止迭代
            best_prob, best_complete, best_seq, _, _ = max(cur_beam)
            if best_complete == True:
                return self._prepar_seq(best_seq)
            else:
                #6. 则重新遍历新的堆中的数据
                prev_beam = cur_beam
        def _prepar_seq(self,seq):#对结果进行基础的处理，共后续转化为文字使用
            if seq[0].item() == self.zh_vocab_size - 2:
                seq=  seq[1:]
            if  seq[-1].item() == self.zh_vocab_size - 1:
                seq = seq[:-1]
            seq = [i.item() for i in seq]
            return seq


class LSTMseq2seq(torch.nn.Module):
    def __init__(self, en_vocab_size, zh_vocab_size, embedded_size, hidden_size, batch_size):
        super(LSTMseq2seq, self).__init__()
        self.zh_vocab_size = zh_vocab_size
        self.batch_size = batch_size
        self.embedded_size = embedded_size
        self.hidden_size = hidden_size

        self.en_embedder = nn.Embedding(en_vocab_size, embedded_size)
        self.zh_embedder = nn.Embedding(zh_vocab_size, embedded_size)

        self.encoder = nn.LSTM(input_size=embedded_size, hidden_size=hidden_size, num_layers=2, bidirectional=False, batch_first=True)
        self.decoder = nn.LSTM(input_size=embedded_size, hidden_size=hidden_size, num_layers=2, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(embedded_size, zh_vocab_size)

        self.h0 = torch.randn((2, self.batch_size, hidden_size), device=device)
        self.c0 = torch.randn((2, self.batch_size, hidden_size), device=device)

    def forward(self, mode, en_sen, zh_sen, teacher_forcing_ratio=0.5):
        emebedded_enseq = self.en_embedder(en_sen)
        en_output, (hidden, cell) = self.encoder(emebedded_enseq, (self.h0, self.c0))

        # 处理中文句子，前面加入['sos']，并去除最后一个
        sos = torch.ones((self.batch_size, 1), dtype=torch.long, device=device) * (self.zh_vocab_size - 2)
        zh_sen = torch.concatenate((sos, zh_sen[:, :-1]), dim=1)
        emebedded_zhseq = self.zh_embedder(zh_sen)  # [batch_size, seq_len, embedded_size]
        seq_len = emebedded_zhseq.shape[1]

        outputs = []
        for t in range(seq_len):
            # 采用gt，(当teacher_forcing_ratio=0.0时，即位test mode)
            if t == 0 or torch.rand(1).item() < teacher_forcing_ratio:
                input_t = emebedded_zhseq[:, t, :].unsqueeze(1)
            else:
                input_t = output_t
            output_t, (hidden, cell) = self.decoder(input_t, (hidden, cell))
            scores = torch.matmul(en_output, output_t.permute(0, 2, 1)) # [batch_size, seq, hidden_size] @ [batch_size, hidden_size, 1] = [batch_size, seq, 1]
            attention_weights = F.softmax(scores, dim=1)
            weighted_sum = torch.sum(en_output * attention_weights, dim=1).unsqueeze(1)
            output_t = torch.concatenate((output_t, weighted_sum), dim=-1)

            outputs.append(output_t)

        de_output = torch.cat(outputs, dim=1)
        output = self.fc(de_output.reshape(-1, self.embedded_size))

        return output