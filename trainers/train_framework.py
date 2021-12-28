'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import torch


class TrainFramework:
    def __init__(self, model, loss, opts):
        self.model = model
        self.loss = loss
        self.lr = opts.lr
        self.beta1 = opts.beta1
        self.beta2 = opts.beta2
        if opts.optimizer == "adam":
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), eps=1e-08, weight_decay=0, amsgrad=False)
        elif opts.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr)
        else: 
            print("Option {} not supported. Available options: adam, sgd".format(opts.optimizer))
            raise NotImplementedError

        if torch.cuda.is_available():
            self.model = model.cuda()
            self.loss = loss.cuda()

    def optimize(self, X, y):
        y_pred = self.model.forward(X)
        loss = self.loss(y, y_pred)
        return loss, y_pred

    def backwardpass(self, loss):
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)