import os
import numpy as np
import torch
from torch import nn


class RecurrentModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size):
        super(RecurrentModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size

        self.recurrent = nn.RNN(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.n_layers,
                                nonlinearity='tanh',
                                bias=True)
        self.output = nn.Linear(in_features=self.hidden_size,
                                out_features=self.output_size,
                                bias=True)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        # x: batch_size, length, n_features
        hidden = None
        rnn_output, hidden = self.recurrent(x, hidden)
        # rnn_output: batch_size, length, n_hidden
        # hidden: 5, length, n_hidden
        rnn_output_flat = rnn_output.view(-1, self.hidden_size)
        # rnn_output_flat: batch_size*length, n_hidden
        lin_output = self.output(rnn_output_flat)
        # lin_output: batch_size*length, n_out
        output_flat = self.softmax(lin_output)
        # output_flat: batch_size*length, n_out
        output = output_flat.view(rnn_output.size(0), rnn_output.size(1), output_flat.size(1))
        # output: batch_size, length, n_out

        return output


class ModelTrainer():
    def __init__(self, model, val_loader, train_loader, n_epochs):
        self.model = model
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.n_epochs = n_epochs
        self.epoch = 0

        self.gpu = torch.cuda.is_available()
        self.val_losses = []
        self.train_losses = []

        if self.gpu:
            self.model = self.model.cuda()
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.004)
        self.criterion = nn.CTCLoss()
  
    def train(self):

        for _ in range(self.epoch, self.n_epochs):

            self.model.train()

            epoch_loss = 0

            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                print('\rBatch {:03}/{:03}'.format(batch_idx+1, len(self.train_loader)), end='')
                epoch_loss += self.train_batch(inputs, targets)
            
            epoch_loss = epoch_loss / len(self.train_loader)

            self.train_losses.append(epoch_loss)

            self.epoch += 1

            print('\r[TRAIN] Epoch {:02}/{:02} Loss {:7.4f}'.format(
                self.epoch, self.n_epochs, epoch_loss
            ), end='\t')
        
    def train_batch(self, inputs, targets):
        if self.gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()
        
        outputs = self.model(inputs)
        targets = targets.type(torch.LongTensor)

        input_lens = np.sum(inputs.detach().numpy()[:,:,0] != -1, axis=1)
        targets_lens = np.sum(targets.detach().numpy() != -1, axis=1)

        loss = self.criterion(outputs.permute(1,0,2), targets, tuple(input_lens), tuple(targets_lens))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
    
    def save(self):
        torch.save(
            {'state_dict': self.model.state_dict()},
            os.path.join('experiments', 'model_{}.pkl'.format(self.epoch))
        )

        with open(os.path.join('experiments', 'train_losses.txt'), 'w') as fw:
            for i in range(len(self.train_losses)):
                fw.write('{:02} {:10.6}\n'.format(i, self.train_losses[i]))
