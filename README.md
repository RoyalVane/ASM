# ASM
PyTorch implementation for our NeurIPS 2020 paper: [Adversarial Style Mining for One-Shot Unsupervised Domain Adaptation](https://proceedings.neurips.cc/paper/2020/hash/ed265bc903a5a097f61d3ec064d96d2e-Abstract.html).

 
## Usage

### Prerequisites
- Python 3.6
- GPU Memory >= 32G
- Pytorch >= 1.5.0

### Download ImageNet-pretained DeepLab:
[DeepLab_resnet_pretrained_init-f81d91e8.pth]( https://drive.google.com/open?id=13kjtX481LdtgJcpqD3oROabZyhGLSBm2) and put it under `pretrained/`.

### Download Pretained RAIN
Download [vgg_normalized.pth](https://drive.google.com/file/d/1EwhOhRSRxDfGebTyMAFMLxZgd-fztc9o/view?usp=sharing)/[decoder_iter_160000.pth](https://drive.google.com/file/d/1p56j29T2B-q2LAEpiwwW3ahlkEep0yNq/view?usp=sharing)/[fc_encoder_iter_160000.pth](https://drive.google.com/file/d/1MzeG28skoWdcjjc0DbmYj4iPbnZbbfDm/view?usp=sharing)/[fc_decoder_iter_160000.pth](https://drive.google.com/file/d/1xrkpSPljeGJvhBYQbL8yLiOIRmWGxBc0/view?usp=sharing) and put them under `pretrained/`.

### Download DataSets
Download [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/)
Download [Cityscapes]( https://www.cityscapes-dataset.com/)


### Modify data path to your own
https://github.com/RoyalVane/ASM/blob/1dac4bfc702da7aaff342683cad628b73807ab2e/ASM/ASM_train.py#L67
https://github.com/RoyalVane/ASM/blob/1dac4bfc702da7aaff342683cad628b73807ab2e/ASM/ASM_train.py#L83
https://github.com/RoyalVane/ASM/blob/1dac4bfc702da7aaff342683cad628b73807ab2e/ASM/ASM_evaluate.py#L14

### Train
```
CUDA_VISIBLE_DEVICES=<gpu_id> python ASM_train.py --snapshot-dir ./snapshots/GTA2Cityscapes
```

### Test
```
CUDA_VISIBLE_DEVICES==<gpu_id> python ASM_evaluate.py
```

### Compute IOU
```
python ASM_IOU.py
```

### Our trained Model
We also provide our trained models for direct evaluation. These models are trained using 32G V100.

The first model is consist with our reported IoU result in the paper.
[IoU = 44.53:](https://drive.google.com/file/d/1SA8jxfdLt15AzE-nOM24dccf47fMjil6/view?usp=sharing) 

The second model is trained recently, whose performance is slightly higher than the paper.
[IoU = 44.78:](https://drive.google.com/file/d/1C1fbhMfZW6aIah5L58yw9mQ7EpuriNt5/view?usp=sharing)


## Related Works
[CLAN](https://github.com/RoyalVane/CLAN): One-shot UDA is a realistic but more challenging setting than UDA, which we tried to solve in our CVPR2019 oral paper "Taking A Closer Look at Domain Shift: Category-level Adversaries for Semantics Consistent Domain Adaptation".

[Copy and Paste GAN](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Copy_and_Paste_GAN_Face_Hallucination_From_Shaded_Thumbnails_CVPR_2020_paper.pdf): RAIN is also employed as a strong data augmentation module in our CVPR2020 oral paper "Copy and Paste GAN: Face Hallucination from Shaded Thumbnails".

