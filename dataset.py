from PIL import Image
from PIL import ImageEnhance
import glob
import random
import os
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as tf
import torchvision.transforms.functional as F
def random_roate(image, mask):
    # 拿到角度的随机数。angle是一个-180到180之间的一个数
    angle = tf.RandomRotation.get_params([-180, 180])
    # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
    image = image.rotate(angle)
    mask = mask.rotate(angle)
    return image, mask
def enhance_feature(image):
    if random.random() > 0.5:
        enh_image = ImageEnhance.Brightness(image)
        brightness = 1.5
        image = enh_image.enhance(brightness)
    if random.random() > 0.5: 
        enh_col = ImageEnhance.Color(image)
        color = 1.5
        image = enh_col.enhance(color)
    if random.random() > 0.5:
        enh_con = ImageEnhance.Contrast(image)
        contrast = 1.5
        image = enh_con.enhance(contrast)
    return image

def rand_crop(feature,label,height=288,width=288):
    """
    Random crop feature (PIL image) and label (PIL image).
    """
    i, j, h, w = tf.RandomCrop.get_params(img=feature,output_size=(height,width))

    feature = F.crop(feature, i, j, h, w)
    label = F.crop(label, i, j, h, w)
    return feature, label


class MyDataset(Dataset):
    def __init__(self, root, is_training=False):
        self.is_training = is_training
        self.root = root
        self.files_A = sorted(glob.glob(os.path.join(root, 'train_A') + '/*.png')) #cloud
        self.files_D = sorted(glob.glob(os.path.join(root, 'train_B') + '/*.png')) #二值图
        self.trans = tf.Compose([
                tf.ToTensor(),
                tf.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])

    def __getitem__(self, index):
        # img = cv2.imread(self.files_A[index % len(self.files_A)])
        # mask = cv2.imread(self.files_D[index % len(self.files_D)])
        # shadow = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).convert('RGB')
        # mask = Image.fromarray(cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)).convert('L') 
        shadow = Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')
        mask = Image.open(self.files_D[index % len(self.files_D)]).convert('L')
        if self.is_training:
            shadow, mask = rand_crop(shadow,mask)
            shadow, mask = random_roate(shadow, mask)
            shadow = enhance_feature(shadow)
        else:    #(显存不够测试时)
            shadow = tf.Resize((288,288))(shadow)
            mask = tf.Resize((288,288))(mask)
        if len(mask.size) > 2:
            mask = mask[:, :, 0]

        shadow_img = self.trans(shadow)
        mask_img = tf.ToTensor()(mask)

        return shadow_img, mask_img

    def __len__(self):
        return len(self.files_A)

