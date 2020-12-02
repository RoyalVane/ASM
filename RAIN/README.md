# pytorch-RAIN

This is an official pytorch implementation of the paper, Adversarial Style Mining for One-Shot Unsupervised Domain Adaptation. [Yawei Luo+, NeurIPS2020].


## Usage

### Download models
Download [vgg_normalized.pth](https://drive.google.com/file/d/1EwhOhRSRxDfGebTyMAFMLxZgd-fztc9o/view?usp=sharing)/[decoder_iter_160000.pth](https://drive.google.com/file/d/1p56j29T2B-q2LAEpiwwW3ahlkEep0yNq/view?usp=sharing)/[fc_encoder_iter_160000.pth](https://drive.google.com/file/d/1MzeG28skoWdcjjc0DbmYj4iPbnZbbfDm/view?usp=sharing)/[fc_decoder_iter_160000.pth](https://drive.google.com/file/d/1xrkpSPljeGJvhBYQbL8yLiOIRmWGxBc0/view?usp=sharing) and put them under `models/`.

### Download DataSets
Download [WikiART](https://storage.googleapis.com/kagglesdsdata/competitions/5127/868727/train.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1607156836&Signature=WyHDcxlognSXjpF0nYeRMMUGzdn%2B0n6knOT3I%2BygltsMFfSoT2AoWBP4Swaaacs6j9J7vpG3SiYTskH90POT6olz1FNsBO3pd0YKPyRTNb6%2BpoTXZTe5p2GHXP3l7FF6CryIu8yzGUErDEHnky0ZHQ23riSVJG%2F95JikmY9xdXhbKZqD7I%2Bktoup8T1r%2BhgGtayZAXRaerhTwhekYPMQKvIUQvdWedvEyAqo5%2BouUiM8KTKfuTcsnZ1a4H7Q0IYIJMMk5nbuoy5z80gqUGhBf8kI17RSTEpOY7q4Gz5lU66rCtjElvjGSHsgWdfGcqSxQ2zD2%2B%2FSHMCzcFHViC1oBQ%3D%3D&response-content-disposition=attachment%3B+filename%3Dtrain.zip) and put them under `datasets/`.

Download [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) and put them under `datasets/`.


### Test
Use `--content` and `--style` to provide the respective path to the content and style image.
```
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content input/content/GTA5.png --style input/style/Cityscapes.png
```

This would generate one randomly stylized image near the anchor style.

### Train
Use `--content_dir` and `--style_dir` to provide the respective directory to the content and style images.
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py --content_dir datasets/GTA5/images --style_dir datasets/wikiarts
```



