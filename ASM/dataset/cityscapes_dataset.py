import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image
import random
from torchvision import transforms


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize([0,0]))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321,321), mean=(128, 128, 128), scale=False, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_w, self.crop_h = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.tf = test_transform(0, False)
        #self.mean_bgr = np.array([72.30608881, 82.09696889, 71.60167789])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            #print(img_file)
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __scale__(self):
# =============================================================================
#         cropsize = self.crop_size
#         if self.scale:
#             r = random.random()
#             if r > 0.7:
#                 cropsize = (int(self.crop_size[0] * 1.05), int(self.crop_size[1] * 1.05))
#             elif r < 0.3:
#                 cropsize = (int(self.crop_size[0] * 0.7), int(self.crop_size[1] * 0.7))
# =============================================================================
        return self.crop_size


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        #cropsize = self.__scale__()

        try:
            image = Image.open(datafiles["img"]).convert('RGB')
            if self.set == 'train':
                image = image.resize((1024,512), Image.BICUBIC)
                img_w, img_h = image.size
                assert(img_h >= self.crop_h)
                assert(img_w >= self.crop_w)
                h_off = random.randint(0, img_h - self.crop_h)
                w_off = random.randint(0, img_w - self.crop_w)
                image = image.crop((w_off, h_off, w_off + self.crop_w, h_off + self.crop_h))
            else:
                image = image.resize((self.crop_w,self.crop_h), Image.BICUBIC)

            image_rgb = self.tf(image)
            name = datafiles["name"]
            # resize
            #image = image.resize(cropsize, Image.BICUBIC)
            image = np.asarray(image, np.float32)
            size = image.shape
            image = image[:, :, ::-1]  # change to BGR
            #image_rgb = image[:, :, ::-1]
            image -= self.mean
            image = image.transpose((2, 0, 1))
            #image_rgb = image_rgb.transpose((2, 0, 1))


            if self.is_mirror and random.random() < 0.5:
                idx = [i for i in range(size[1] - 1, -1, -1)]
                image = np.take(image, idx, axis = 2)

        except Exception as e:
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)

        return image.copy(), image_rgb, np.array(size), name


if __name__ == '__main__':
    dst = GTA5DataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=200)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
