# -----------------------------------------------------------
# "Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval"
# Yuan, Zhiqiang and Zhang, Wenkai and Fu, Kun and Li, Xuan and Deng, Chubo and Wang, Hongqi and Sun, Xian
# IEEE Transactions on Geoscience and Remote Sensing 2021
# Writen by YuanZhiqiang, 2021.  Our code is depended on MTFN
# ------------------------------------------------------------

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import yaml
import argparse
from PIL import Image


def read_detail_file(file_path):
    data_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.rstrip('\n').split(' ')
            if len(parts) >= 2:
                image_name = parts[0]
                description = ' '.join(parts[1:])  
                data_dict[image_name] = description
    return data_dict


class ImageTextDetailWordDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_split, test_data_index, opt, transformer='clip'):
        self.loc = opt['dataset']['data_path']
        self.img_path_train = opt['dataset']['image_path_train']
        self.img_path_test = opt['dataset']['image_path_test']

        self.detail_path_train = opt['dataset']['detail_path_train']
        self.detail_path_test = opt['dataset']['detail_path_test']
        self.word_path_train = opt['dataset']['word_path_train']
        self.word_path_test = opt['dataset']['word_path_test']
        # Captions
        self.captions = []
        self.detail = []
        self.word = []
        self.maxlength = 0

        detail_dict_train = read_detail_file(self.detail_path_train)
        detail_dict_test = read_detail_file(self.detail_path_test)
        word_dict_train = read_detail_file(self.word_path_train)
        word_dict_test = read_detail_file(self.word_path_test)
        if data_split == 'train':
            self.images = []
            with open(self.loc + '/csv_file_train_Ret-3_train.csv', 'rb') as f:
                next(f)
                for line in f:
                    line = line.decode("utf-8")
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) >= 2:
                        image_name = parts[-1]
                        description = parts[0]
                        self.captions.append(description)
                        image_path = os.path.join(self.img_path_train, image_name)
                        self.images.append(image_path)
                        self.detail.append(detail_dict_train[image_name])
                        self.word.append(word_dict_train[image_name])
        else:
            self.images = []
            if test_data_index == 0:
                test_csv = '/csv_file_test_rsitmd_test.csv'
            elif test_data_index == 1:
                test_csv = '/csv_file_test_rsicd_test.csv'
            else:
                test_csv = '/csv_file_test_ucm_test.csv'
            with open(self.loc + test_csv, 'rb') as f:
                next(f)
                for line in f:
                    line = line.decode("utf-8")
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) >= 2:
                        image_name = parts[-1]
                        description = parts[0]
                        self.captions.append(description)
                        image_path = os.path.join(self.img_path_test, image_name)
                        self.images.append(image_path)
                        self.detail.append(detail_dict_test[image_name])
                        self.word.append(word_dict_test[image_name])

        self.length = len(self.captions)
        print('dataset len: ', self.length)

        if len(self.images) != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == "train":
            if transformer == 'clip':
                self.transform = transforms.Compose([
                    transforms.Resize((248, 248)),
                    transforms.RandomRotation((45)),
                    transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711))])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomRotation((0, 90)),
                    transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            if transformer == 'clip':
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711))])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomRotation((0, 90)),
                    transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div

        caption = self.captions[index]
        imagename = self.images[index]
        image = Image.open(self.images[index]).convert('RGB')
        image = self.transform(image)  

        detail = self.detail[index]
        word = self.word[index]

        return image, caption, index, index, detail, word, imagename

    def __len__(self):
        return self.length

def custom_collate_fn(batch):

    images, captions, indices, img_ids, pths = zip(*batch)

    images = torch.stack(images, dim=0)
    captions = captions  
    indices = torch.tensor(indices)
    img_ids = torch.tensor(img_ids)

    pths = torch.stack(list(pths), dim=0)

    return images, captions, indices, img_ids, pths


def get_ImgTextdetailWord_loader(data_split, batch_size=100,
                             shuffle=True, num_workers=0, test_data_index=0, opt={}):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = ImageTextDetailWordDataset(data_split=data_split,test_data_index=test_data_index, opt=opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              # collate_fn=custom_collate_fn,
                                              # multiprocessing_context='spawn',
                                              num_workers=num_workers)
    return data_loader



def get_ITDWloaders(opt,test_data_index):
    train_loader = get_ImgTextdetailWord_loader('train',
                                            opt['dataset']['batch_size'], True, opt['dataset']['workers'], opt=opt)
    val_loader = [get_ImgTextdetailWord_loader('test',
                                          opt['dataset']['batch_size_val'], False, opt['dataset']['workers'],test_data_index = i, opt=opt)for i in test_data_index]
    return train_loader, val_loader


def get_test_ITDWloader(opt):
    test_loader = get_ImgTextdetailWord_loader('test',
                                           opt['dataset']['batch_size_val'], False, opt['dataset']['workers'], opt=opt)

    return test_loader

def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/RSITMD_AMFMN.yaml', type=str,
                        help='path to a yaml options file')
    # parser.add_argument('--text_sim_path', default='data/ucm_precomp/train_caps.npy', type=str,help='path to t2t sim matrix')
    opt = parser.parse_args()

    # load model options
    with open(opt.path_opt, 'r') as handle:
        options = yaml.load(handle, Loader=yaml.FullLoader)

    return options

