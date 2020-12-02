import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
#from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 1024)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=1.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--latent_weight', type=float, default=1.0)
parser.add_argument('--recons_weight', type=float, default=5.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
args = parser.parse_args()

device = torch.device('cuda')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
#writer = SummaryWriter(log_dir=str(log_dir))

decoder = net.decoder
vgg = net.vgg
fc_encoder = net.fc_encoder
fc_decoder = net.fc_decoder

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder, fc_encoder, fc_decoder)
network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)
optimizer_fc_encoder = torch.optim.Adam(network.fc_encoder.parameters(), lr=args.lr)
optimizer_fc_decoder = torch.optim.Adam(network.fc_decoder.parameters(), lr=args.lr)


for i in tqdm(range(args.max_iter)):
    
    for param in network.fc_encoder.parameters():
        param.requires_grad = True
    for param in network.fc_decoder.parameters():
        param.requires_grad = True
        
    adjust_learning_rate(optimizer, iteration_count=i)
    adjust_learning_rate(optimizer_fc_encoder, iteration_count=i)
    adjust_learning_rate(optimizer_fc_decoder, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    loss_c, loss_s, loss_l, loss_r = network(content_images, style_images)
    loss_l = args.latent_weight * loss_l
    loss_r = args.recons_weight * loss_r
    loss_fc = loss_l + loss_r
    optimizer_fc_encoder.zero_grad()
    optimizer_fc_decoder.zero_grad()
    loss_fc.backward(retain_graph = True)
    optimizer_fc_encoder.step()
    optimizer_fc_decoder.step()
    
    for param in network.fc_encoder.parameters():
        param.requires_grad = False
    for param in network.fc_decoder.parameters():
        param.requires_grad = False
    
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    optimizer.zero_grad()
    loss_de = loss_c + loss_s
    loss_de.backward()
    optimizer.step()
    

    print(loss_c.item(), loss_s.item(), loss_l.item(), loss_r.item())

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = net.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'decoder_iter_{:d}.pth'.format(i + 1))
        
        state_dict = net.fc_encoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'fc_encoder_iter_{:d}.pth'.format(i + 1))
        
        state_dict = net.fc_decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'fc_decoder_iter_{:d}.pth'.format(i + 1))
