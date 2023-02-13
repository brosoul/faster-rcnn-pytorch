import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class FRCNNDataset(Dataset):
    def __init__(self, annotation_lines, borders_lines, input_shape = [600, 600], train = True):
        self.annotation_lines   = annotation_lines
        self.anchors_lines = borders_lines
        self.length             = len(annotation_lines)
        
        assert len(annotation_lines) == len(borders_lines)
        self.input_shape        = input_shape
        self.train              = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # TODO 新增返回网页结构框
        index       = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        image, y, z    = self.get_random_data(self.annotation_lines[index], self.anchors_lines[index],self.input_shape[0:2], random = self.train)
        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box_data    = np.zeros((len(y), 5))
        anchors_data = np.zeros((len(z), 4))
        if len(y) > 0:
            box_data[:len(y)] = y

        if len(z) > 0:
            anchors_data[:len(z)] = z
        
        box         = box_data[:, :4]
        label       = box_data[:, -1]
        anchors = anchors_data[:, :4]
        return image, box, label, anchors

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, anchors_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        
        line_for_anchors = anchors_line.split()
        # 获得 anchors 信息
        anchors = np.array([np.array(list(map(int,box.split(',')))) for box in line_for_anchors[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            
            if len(anchors) > 0:
                np.random.shuffle(anchors)
                anchors[:, [0,2]] = anchors[:, [0,2]]*nw/iw + dx
                anchors[:, [1,3]] = anchors[:, [1,3]]*nh/ih + dy
                anchors[:, 0:2][anchors[:, 0:2]<0] = 0
                anchors[:, 2][anchors[:, 2]>w] = w
                anchors[:, 3][anchors[:, 3]>h] = h
                anchors_w = anchors[:, 2] - anchors[:, 0]
                anchors_h = anchors[:, 3] - anchors[:, 1]
                anchors = anchors[np.logical_and(anchors_w>1, anchors_h>1)] # discard invalid box

            return image_data, box, anchors
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 

        # 对 anchors 信息也进行调整
        if len(anchors)>0:
            np.random.shuffle(anchors)
            anchors[:, [0,2]] = anchors[:, [0,2]]*nw/iw + dx
            anchors[:, [1,3]] = anchors[:, [1,3]]*nh/ih + dy
            if flip: anchors[:, [0,2]] = w - anchors[:, [2,0]]
            anchors[:, 0:2][anchors[:, 0:2]<0] = 0
            anchors[:, 2][anchors[:, 2]>w] = w
            anchors[:, 3][anchors[:, 3]>h] = h
            anchors_w = anchors[:, 2] - anchors[:, 0]
            anchors_h = anchors[:, 3] - anchors[:, 1]
            anchors = anchors[np.logical_and(anchors_w>1, anchors_h>1)] 

        
        return image_data, box, anchors

# DataLoader中collate_fn使用
def frcnn_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    anchors = []
    for img, box, label, anchor in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
        anchors.append(anchor)
    images = torch.from_numpy(np.array(images))
    return images, bboxes, labels, anchors

