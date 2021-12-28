'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
from options.base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
            
        #Options for training
        parser.add_argument('--n_epochs', type=int, default=20, help='# of training epochs')
        parser.add_argument('--optimizer', type=str, default='adam', help='optimizer for training model [adam | sgd]')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')

        parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--loss', type=str, default='bce', help='loss for training model [bce | wbce | dice | dice_bce]')
        parser.add_argument('--scheduler_patience', type=int, default=5, help='lr scheduler patience')
        

        self.isTrain = True
        return parser