import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import os.path as osp
from torchvision.utils import save_image, make_grid

#from model.ASM_G import Res_Deeplab
from model.ASM_G import Res_Deeplab
#from model.ASM_D import FCDiscriminator
from model.RAIN import encoder, decoder, fc_encoder, fc_decoder, device

from utils.loss import CrossEntropy2d
from utils.loss import WeightedBCEWithLogitsLoss

from dataset.gta5_dataset import GTA5DataSet
from dataset.synthia_dataset import SYNTHIADataSet
from dataset.cityscapes_dataset import cityscapesDataSet
#import matplotlib.pyplot as plt
#from PIL import Image
import imageio



IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'ResNet'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 2

IGNORE_LABEL = 255

MOMENTUM = 0.9
NUM_CLASSES = 19
RESTORE_FROM = './pretrained/DeepLab_resnet_pretrained_init-f81d91e8.pth'
#RESTORE_FROM = './snapshots/GTA2Cityscapes_CLAN/GTA5_20000.pth' #For retrain
#RESTORE_FROM_D = './snapshots/GTA2Cityscapes_CLAN/GTA5_20000_D.pth' #For retrain

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = './snapshots/'

#Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
NUM_STEPS = 100000
NUM_STEPS_STOP = 100000  # Use damping instead of early stopping
WARMUP_STEPS = int(NUM_STEPS_STOP/20)
POWER = 0.9
RANDOM_SEED = 1234

SOURCE = 'GTA5'
TARGET = 'cityscapes'
SET = 'train'

if SOURCE == 'GTA5':
    INPUT_SIZE_SOURCE = '1024,512' # 24GB '960,480'
    DATA_DIRECTORY = '/data02/yawei/Data/GTA5/'
    DATA_LIST_PATH = './dataset/gta5_list/train.txt'
    Lambda_weight = 0.01
    Lambda_adv = 0.001
    Lambda_local = 40
    Epsilon = 0.4
elif SOURCE == 'SYNTHIA':
    INPUT_SIZE_SOURCE = '960,480'
    DATA_DIRECTORY = '/data02/yawei/Data/SYNTHIA'
    DATA_LIST_PATH = './dataset/synthia_list/train.txt'
    Lambda_weight = 0.01
    Lambda_adv = 0.001
    Lambda_local = 10
    Epsilon = 0.4

INPUT_SIZE_TARGET = '1024,512'
DATA_DIRECTORY_TARGET = '/data02/yawei/Data/Cityscapes/'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : ResNet")
    parser.add_argument("--source", type=str, default=SOURCE,
                        help="available options : GTA5, SYNTHIA")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size-source", type=str, default=INPUT_SIZE_SOURCE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument('--vgg_encoder', type=str, default='pretrained/vgg_normalised.pth')
# =============================================================================
#     parser.add_argument('--vgg_decoder', type=str, default='pretrained/vgg_decoder.pth')
#     parser.add_argument('--style_encoder', type=str, default='pretrained/style_encoder.pth')
#     parser.add_argument('--style_decoder', type=str, default='pretrained/style_decoder.pth')
# =============================================================================
    parser.add_argument('--vgg_decoder', type=str, default='pretrained/decoder_iter_160000.pth')
    parser.add_argument('--style_encoder', type=str, default='pretrained/fc_encoder_iter_160000.pth')
    parser.add_argument('--style_decoder', type=str, default='pretrained/fc_decoder_iter_160000.pth')
    parser.add_argument('--fp16', action='store_true',
                        help='use float16 instead of float32, which will save about 50% memory' )
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d(NUM_CLASSES).cuda(gpu)
    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)


def adjust_learning_rate(optimizer, i_iter):
    if i_iter < WARMUP_STEPS:
        lr = lr_warmup(args.learning_rate, i_iter, WARMUP_STEPS)
    else:
        lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    if i_iter < WARMUP_STEPS:
        lr = lr_warmup(args.learning_rate_D, i_iter, WARMUP_STEPS)
    else:
        lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_feat_mean_std(input, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = input.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = input.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C)
    feat_mean = input.view(N, C, -1).mean(dim=2).view(N, C)
    return torch.cat([feat_mean, feat_std], dim = 1)


def adaptive_instance_normalization_with_noise(content_feat, style_feat):
    #assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    N, C = size[:2]
    style_mean = style_feat[:, :512].view(N, C, 1, 1)
    style_std = style_feat[:, 512:].view(N, C, 1, 1)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def style_transfer(encoder, decoder, fc_encoder, fc_decoder, content, style, sampling = None):
    with torch.no_grad():
        content_feat = encoder(content)
        style_feat = encoder(style)
    style_feat_mean_std = calc_feat_mean_std(style_feat)
    intermediate = fc_encoder(style_feat_mean_std)
    intermediate_mean = intermediate[:, :512]
    intermediate_std = intermediate[:, 512:]
    noise = torch.randn_like(intermediate_mean)
    if sampling is None:
        sampling = intermediate_mean + noise * intermediate_std #N, 512
    sampling.require_grad = True
    style_feat_mean_std_recons = fc_decoder(sampling) #N, 1024
    feat = adaptive_instance_normalization_with_noise(content_feat, style_feat_mean_std_recons)

    return decoder(feat), sampling


# =============================================================================
# def my_trans(A, B):
#     ## A: 0-1, RGB
#     ## B: -128-127, BGR
# =============================================================================


def main():
    """Create the model and start the training."""


    h, w = map(int, args.input_size_source.split(','))
    input_size_source = (h, w)

    h, w = map(int, args.input_size_target.split(','))
    input_size_target = (h, w)

    cudnn.enabled = True

    # Create Network
    segmentor = Res_Deeplab(num_classes=args.num_classes)
    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    new_params = segmentor.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not args.num_classes == 19 or not i_parts[1] == 'layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]

    if args.restore_from[:4] == './pr':
        segmentor.load_state_dict(new_params)
    else:
        segmentor.load_state_dict(saved_state_dict)

    segmentor.train()
    segmentor.cuda(args.gpu)

    cudnn.benchmark = True

    # Init D
    #model_D = FCDiscriminator(num_classes=args.num_classes)
    #for retrain
    #saved_state_dict_D = torch.load(RESTORE_FROM_D)
    #model_D.load_state_dict(saved_state_dict_D)

    #model_D.train()
    #model_D.cuda(args.gpu)

    vgg_encoder = encoder
    vgg_decoder = decoder

    style_encoder = fc_encoder
    style_decoder = fc_decoder

    vgg_encoder.eval()
    style_encoder.eval()
    vgg_decoder.eval()
    style_decoder.eval()

    vgg_encoder.load_state_dict(torch.load(args.vgg_encoder))
    vgg_encoder = nn.Sequential(*list(vgg_encoder.children())[:31])
    vgg_decoder.load_state_dict(torch.load(args.vgg_decoder))

    style_encoder.load_state_dict(torch.load(args.style_encoder))
    style_decoder.load_state_dict(torch.load(args.style_decoder))

    vgg_encoder.to(device)
    vgg_decoder.to(device)
    style_encoder.to(device)
    style_decoder.to(device)

    for param in vgg_encoder.parameters():
        param.requires_grad = False
# =============================================================================
#     for param in style_encoder.parameters():
#         param.requires_grad = False
# =============================================================================

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    if args.source == 'GTA5':
        trainloader = data.DataLoader(
            GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                        crop_size=input_size_source,
                        scale=False, mirror=False, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = data.DataLoader(
            SYNTHIADataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                        crop_size=input_size_source,
                        scale=True, mirror=False, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=False, mean=IMG_MEAN,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)


    targetloader_iter = enumerate(targetloader)

    optimizer = optim.SGD(segmentor.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()


    interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear', align_corners=True)
    #interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    # Labels for Adversarial Training
    #source_label = 0
    #target_label = 1

    loss_norm = nn.MSELoss()


    _, batch_t = next(targetloader_iter)
    images_t, images_t_rgb, _, _ = batch_t
    images_t = Variable(images_t).cuda(args.gpu)
    images_t_rgb = Variable(images_t_rgb).cuda(args.gpu)
    images_t.requires_grad = False
    images_t_rgb.requires_grad = False

    for i_iter in range(args.num_steps):

        sampling = None

        adjust_learning_rate(optimizer, i_iter)

        #damping = (1 - i_iter/NUM_STEPS)

        # Train with Source
        _, batch_s = next(trainloader_iter)

        images_s, labels_s, images_s_rgb, _, _ = batch_s

        images_s = Variable(images_s).cuda(args.gpu)

        images_s_rgb = Variable(images_s_rgb).cuda(args.gpu)

        images_s.requires_grad = False

        images_s_rgb.requires_grad = False


        for i in range(2):

            optimizer.zero_grad()
            images_s_style, sampling = style_transfer(vgg_encoder, vgg_decoder,
                                                          style_encoder, style_decoder, images_s_rgb, images_t_rgb, sampling)
            indices = torch.tensor([2,1,0]).cuda()
            images_s_style = torch.index_select(images_s_style, dim = 1, index = indices) * 255.0 - torch.FloatTensor(IMG_MEAN).cuda().view(1,3,1,1)

# =============================================================================
#             #show styled images
            if i_iter % 200 == 0:
                img = make_grid(images_s_style).data.cpu().numpy()
                img = np.int_(np.transpose(img, (1, 2, 0)) + IMG_MEAN)
                img = img[:, :, ::-1]
                imageio.imwrite('style_track/{:d}_{:d}_style.jpg'.format(i_iter, i+1), img)
#
#            if i_iter % 100 == 0 and i == 0:
#                 #show origin images
#                 img = make_grid(images_s).data.cpu().numpy()
#                 img = np.int_(np.transpose(img, (1, 2, 0)) + IMG_MEAN)
#                 img = img[:, :, ::-1]
#                 imageio.imwrite('style_track/{:d}_{:d}_origin.jpg'.format(i_iter, i), img)
# =============================================================================

            images_s_style = interp_source(images_s_style)
            pred, pred_norm = segmentor(torch.cat([images_s_style, images_s], dim = 0))
            pred = interp_source(pred)

            #Segmentation Loss
            loss_1 = loss_calc(pred, torch.cat([labels_s, labels_s], dim = 0), args.gpu)
            loss_2 = loss_norm(pred_norm[39616:], torch.zeros(pred_norm[39616:].size()).cuda())

            sampling.retain_grad()
            loss = loss_1 + 2e-4 * loss_2

            loss.backward(retain_graph=True)

            sampling = sampling + (20.0/loss.item()) * sampling.grad.data
            optimizer.step()
            print('exp = {}'.format(args.snapshot_dir))
            print(
            'iter = {0:6d}/{1:6d}, loss_seg = {2:.4f} loss_adv = {3:.4f}'.format(
                i_iter, args.num_steps, loss_1, loss_2))

        f_loss = open(osp.join(args.snapshot_dir,'loss.txt'), 'a')
        f_loss.write('{0:.4f} {1:.4f}\n'.format(
            loss_1, loss_2))
        f_loss.close()

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(segmentor.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '.pth'))
            #torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(segmentor.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            #torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D.pth'))

if __name__ == '__main__':
    main()
