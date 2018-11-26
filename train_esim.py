from data_help_new import DatasetReader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
from models.Double_ESIM import Double_ESIM
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        dataset = DatasetReader(embed_dim=opt.embed_dim, char_embed_dim=opt.char_dim, max_seq_len=opt.max_seq_len, max_char_len=opt.max_char_len)
        self.train_data_loader = DataLoader(dataset=dataset.train_data, batch_size=opt.batch_size, shuffle=False)
        self.test_data_loader = DataLoader(dataset=dataset.test_data, batch_size=opt.batch_size, shuffle=False)
        self.val_data_loader = DataLoader(dataset=dataset.val_data, batch_size=len(dataset.val_data), shuffle=False)

        self.model = Double_ESIM(opt, dataset.embedding_matrix, dataset.char_embedding_matrix).to(self.opt.device)
        # print(dataset.embedding_matrix[1:10])

    def _train(self, criterion, optimizer):
        # writer = SummaryWriter(log_dir=self.opt.logdir)
        max_val_acc = 0
        max_val_epoch = 0
        global_step = 0
        for epoch in range(self.opt.num_epoch):
            print('>'*50)
            print('epoch:', epoch)
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                self.model.train()
                optimizer.zero_grad()
                # print(sample_batched['p'].size())
                if self.opt.use_char_emb:
                    inputs = [sample_batched['p'].to(self.opt.device), sample_batched['h'].to(self.opt.device),
                              sample_batched['p_char'].to(self.opt.device), sample_batched['h_char'].to(self.opt.device)]
                else:
                    inputs = [sample_batched['p'].to(self.opt.device), sample_batched['h'].to(self.opt.device)]
                outputs = self.model(inputs)
                # print(outputs.size())
                label = sample_batched['label'].to(self.opt.device)

                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                val_acc_,_,_,val_f1 = self._evaluate_acc()
                if float(val_acc_) >= 0.845:
                    self._test(epoch, i_batch)
                    print('----epoch: ' + str(epoch)+ '----batch: ' + str(i_batch)
                          + '---val_acc: ' + str(val_acc_) + '---val_f1: ' + str(val_f1))

                if global_step % self.opt.log_step == 0:
                    # pred = torch.argmax(outputs, dim=1).item()
                    # acc = accuracy_score(label, outputs)
                    # recall = recall_score(label, outputs)
                    # f1 = f1_score(label, outputs)
                    n_correct += (torch.argmax(outputs, -1) == label).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    val_acc, val_p, val_r, val_f = self._evaluate_acc()
                    if val_acc > max_val_acc:
                        max_val_acc = val_acc
                        max_val_epoch = epoch

                    print('loss: {:.4f}, train_acc:{:.4f}, val_acc:{:.4f}, val_p:{:.4f}, '
                          'val_r:{:.4f}, val_f:{:.4f}'.format(loss.item(), train_acc, val_acc, val_p, val_r, val_f))
            self._test(epoch)
        return max_val_acc, max_val_epoch

    def _evaluate_acc(self):
        self.model.eval()
        n_val_correct, n_val_total = 0, 0
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.val_data_loader):
                if self.opt.use_char_emb:
                    t_inputs = [t_sample_batched['p'].to(self.opt.device), t_sample_batched['h'].to(self.opt.device),
                              t_sample_batched['p_char'].to(self.opt.device), t_sample_batched['h_char'].to(self.opt.device)]
                else:
                    t_inputs = [t_sample_batched['p'].to(self.opt.device), t_sample_batched['h'].to(self.opt.device)]
                # t_inputs = [t_sample_batched['p'].to(self.opt.device), t_sample_batched['h'].to(self.opt.device)]
                t_label = t_sample_batched['label'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_val_correct += (torch.argmax(t_outputs, -1) == t_label).sum().item()
                n_val_total += len(t_outputs)

                val_p = precision_score(t_label, torch.argmax(t_outputs, -1))
                val_r = recall_score(t_label, torch.argmax(t_outputs, -1))
                val_f = f1_score(t_label, torch.argmax(t_outputs, -1))
        val_acc = n_val_correct / n_val_total
        return val_acc, val_p, val_r, val_f

    def _test(self, epoch, batch='last'):
        self.model.eval()
        output = []
        for t_batch, t_sample_batched in enumerate(self.test_data_loader):
            if self.opt.use_char_emb:
                t_inputs = [t_sample_batched['p'].to(self.opt.device), t_sample_batched['h'].to(self.opt.device),
                            t_sample_batched['p_char'].to(self.opt.device),
                            t_sample_batched['h_char'].to(self.opt.device)]
            else:
                t_inputs = [t_sample_batched['p'].to(self.opt.device), t_sample_batched['h'].to(self.opt.device)]
            # t_inputs = [t_sample_batched['p'].to(self.opt.device), t_sample_batched['h'].to(self.opt.device)]
            t_outputs = self.model(t_inputs)
            t_outputs = torch.argmax(t_outputs, -1)

            output.extend(t_outputs.cpu().numpy())

        # print (len(output))
        df_test = pd.read_csv('data/new_test.csv', header=None, sep=',', names=['qid1','qid2','wid_1','wid_2','cid_1','cid_2'])
        submission = pd.DataFrame({
            'qid1': df_test['qid1'],
            'qid2': df_test['qid2']
        })
        submission['label'] = pd.Series(output)
        submission.to_csv('results/Double_ESIM_epoch'+str(epoch)+'batch'+str(batch)+'.csv', index=False)


    def run(self):
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate)

        max_val_acc, max_val_epoch = self._train(criterion, optimizer)
        print("max_val_acc: {0}".format(max_val_acc))
        print('max_val_epoch: {0}'.format(max_val_epoch))
        return max_val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--learning_rate', default=0.0003, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--max_seq_len', default=22, type=int)
    parser.add_argument('--class_size', default=2, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_char_emb', default=True, type=bool)
    parser.add_argument('--char_dim', default=300, type=int)
    parser.add_argument('--char_hidden_size', default=100, type=int)
    parser.add_argument('--max_char_len', default=22, type=int)
    opt = parser.parse_args()

    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adam':torch.optim.Adam,
        'sgd': torch.optim.SGD
    }
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\
                    if opt.device is None else torch.device(opt.device)

    ins = Instructor(opt)
    ins.run()









