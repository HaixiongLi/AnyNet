import torch
from tensorboardX import SummaryWriter
import models.anynet
import argparse

parser = argparse.ArgumentParser(description='AnyNet with Flyingthings3d')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datatype', default='2015',help='datapath')
parser.add_argument('--datapath', default='/home/lab3/Datasets/kitti2015/training/',help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=12,
                    help='batch size for training (default: 12)')
parser.add_argument('--test_bsize', type=int, default=8,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/pretrained_anynet',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels of the 3d network')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers of the 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--split_file', type=str, default=None)

args = parser.parse_args()



torch.manual_seed(1.0)
left = torch.randn(1, 3, 256, 512)
right = torch.randn(1, 3, 256, 512)
model = models.anynet.AnyNet(args)

with SummaryWriter(comment='Net')as w:
    w.add_graph(model, (left, right,))