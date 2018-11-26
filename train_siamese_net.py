from data_helper import DatasetReader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
from models.siamese_network import Siamese_Net

class Instructor:
    def __init__(self, opt):
        self.opt = opt

        dataset = DatasetReader(embed_dim=opt.embed_dim, max_seq_len=opt.max_seq_len)
        self.train_data_loader = DataLoader(dataset=dataset.train_data, batch_size=opt.batch_size, shuffle=False)
        self.test_data_loader = DataLoader(dataset=dataset.test_data, batch_size=len(dataset.test_data), shuffle=False)
        self.val_data_loader = DataLoader(dataset=dataset.val_data, batch_size=opt.batch_size, shuffle=False)

        self.model = Siamese_Net(opt, dataset.embedding_matrix).to(self.opt.device)
        self._init_and_print_parameters()

    def _init_and_print_parameters(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    n_nontrainable_params += n_params

        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))

    def _train(self, criterion, optimizer):
        # writer = SummaryWriter(log_dir=self.opt.logdir)
        max_val_acc = 0
        max_val_epoch = 0
        global_step = 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 50)
            print('epoch:', epoch)
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                self.model.train()
                optimizer.zero_grad()
                # print(sample_batched['p'].size())
                inputs = [sample_batched['p'].to(self.opt.device), sample_batched['h'].to(self.opt.device)]
                # print(sample_batched['p'].size())
                # print(sample_batched['h'])
                outputs = self.model(inputs)
                # print(outputs.size())
                label = sample_batched['label'].to(self.opt.device)

                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    # pred = torch.argmax(outputs, dim=1).item()
                    # acc = accuracy_score(label, outputs)
                    # recall = recall_score(label, outputs)
                    # f1 = f1_score(label, outputs)
                    n_correct += (torch.argmax(outputs, -1) == label).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    val_acc = self._evaluate_acc()
                    if val_acc > max_val_acc:
                        max_val_acc = val_acc
                        max_val_epoch = epoch

                    print('loss: {:.4f}, train_acc:{:.4f}, val_acc:{:.4f}'.format(loss.item(), train_acc, val_acc))
        return max_val_acc, max_val_epoch

    def _evaluate_acc(self):
        self.model.eval()
        n_val_correct, n_val_total = 0, 0
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.val_data_loader):
                t_inputs = [t_sample_batched['p'].to(self.opt.device), t_sample_batched['h'].to(self.opt.device)]
                t_label = t_sample_batched['label'].to(self.opt.device)
                t_outputs = self.model(t_inputs)
                # print(t_outputs.size())

                n_val_correct += (torch.argmax(t_outputs, -1) == t_label).sum().item()
                n_val_total += len(t_outputs)
        val_acc = n_val_correct / n_val_total
        return val_acc

    def _test(self):
        self.model.eval()
        for t_batch, t_sample_batched in enumerate(self.test_data_loader):
            t_inputs = [t_sample_batched['p'].to(self.opt.device), t_sample_batched['h'].to(self.opt.device)]
            t_outputs = self.model(t_inputs)

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
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--num_epoch', default=40, type=int)
    parser.add_argument('--batch_size', default=300, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--max_seq_len', default=10, type=int)
    parser.add_argument('--num_perspective', default=10, type=int)
    parser.add_argument('--class_size', default=2, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_char_emb', default=False, type=bool)
    parser.add_argument('--char_hidden_size', default=50, type=int)
    opt = parser.parse_args()

    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD
    }
    initializers = {
        'xavier_uniform_', torch.nn.init.xavier_uniform_,
        'xavier_normal_', torch.nn.init.xavier_normal_,
        'orthogonal_', torch.nn.init.orthogonal_,
    }
    opt.initializer = torch.nn.init.xavier_uniform_
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    ins = Instructor(opt)
    ins.run()

