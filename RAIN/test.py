import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization, adaptive_instance_normalization_with_noise, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def calc_feat_mean_std(input, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = input.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = input.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C)
    feat_mean = input.view(N, C, -1).mean(dim=2).view(N, C)
    return torch.cat([feat_mean, feat_std], dim = 1)


def style_transfer(vgg, decoder, fc_encoder, fc_decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_feat = vgg(content)
    style_feat = vgg(style)
    style_feat_mean_std = calc_feat_mean_std(style_feat)
    intermediate = fc_encoder(style_feat_mean_std)
    intermediate_mean = intermediate[:, :512]
    intermediate_std = intermediate[:, 512:]
    noise = torch.randn_like(intermediate_mean)
    sampling = intermediate_mean + noise * intermediate_std #N, 512
    style_feat_mean_std_recons = fc_decoder(sampling) #N, 1024
    print(style_feat_mean_std)
    print(style_feat_mean_std_recons)
    #assert(0)
    if interpolation_weights:
        _, C, H, W = content_feat.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization_with_noise(content_feat, style_feat_mean_std_recons)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_feat = content_feat[0:1]
    else:
        feat = adaptive_instance_normalization_with_noise(content_feat, style_feat_mean_std_recons)
    
    feat = feat * alpha + content_feat * (1 - alpha)
    return decoder(feat)


def style_transfer_ori(vgg, decoder, fc_encoder, fc_decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_feat = vgg(content)
    style_feat = vgg(style)
    style_feat_mean_std = calc_feat_mean_std(style_feat)
    print(style_feat_mean_std)
    intermediate = fc_encoder(style_feat_mean_std)
    intermediate_mean = intermediate[:, :512]
    intermediate_std = intermediate[:, 512:]
    noise = torch.randn_like(intermediate_mean)
    sampling = intermediate_mean + noise * intermediate_std #N, 512
    style_feat_mean_std_recons = fc_decoder(sampling) #N, 1024
    if interpolation_weights:
        _, C, H, W = content_feat.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_feat, style_feat)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_feat = content_feat[0:1]
    else:
        feat = adaptive_instance_normalization(content_feat, style_feat)
    
    feat = feat * alpha + content_feat * (1 - alpha)
    return decoder(feat)


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='experiments/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='experiments/decoder_iter_160000.pth')
parser.add_argument('--fc_encoder', type=str, default='experiments/fc_encoder_iter_160000.pth')
parser.add_argument('--fc_decoder', type=str, default='experiments/fc_decoder_iter_160000.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=0,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=0,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(args.style)]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

decoder = net.decoder
vgg = net.vgg
fc_encoder = net.fc_encoder
fc_decoder = net.fc_decoder

decoder.eval()
vgg.eval()
fc_encoder.eval()
fc_decoder.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
fc_encoder.load_state_dict(torch.load(args.fc_encoder))
fc_decoder.load_state_dict(torch.load(args.fc_decoder))

vgg.to(device)
decoder.to(device)
fc_encoder.to(device)
fc_decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

for content_path in content_paths:
    if do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
        content = content_tf(Image.open(str(content_path))) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, fc_encoder, fc_decoder, content, style,
                                    args.alpha, interpolation_weights)
        output = output.cpu()
        output_name = output_dir / '{:s}_interpolation{:s}'.format(
            content_path.stem, args.save_ext)
        save_image(output, str(output_name))

    else:  # process one content and one style
        for style_path in style_paths:
            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))
            if args.preserve_color:
                style = coral(style, content)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(vgg, decoder, fc_encoder, fc_decoder, content, style,
                                            args.alpha)
            output = output.cpu()
            print(output.shape)

            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
            save_image(output, str(output_name))
