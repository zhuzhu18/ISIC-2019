import torch.utils.data as data
import pandas as pd
import os
import numpy as np
from PIL import Image


class ISIC(data.Dataset):
    img_size = 224
    def __init__(self, img_dir, truth_csv, train=True, transform=None):
        self.img_dir = img_dir
        self.train = train
        self.transform = transform
        self.truth_csv = truth_csv

        if not os.path.isfile('labels.csv'):
            self._create_labels(img_dir, truth_csv)
        self.annot = pd.read_csv('labels.csv', header=0)

    def _create_labels(self, img_dir, csv_path):
        d = pd.read_csv(csv_path, header=0, index_col='image')
        m = pd.Series()
        for f in os.listdir(img_dir):
            k = f.rstrip('.jpg')
            if k in [l.rstrip('_downsampled') for l in d.index]:
                n = pd.Series(np.nonzero(d.loc[k].values)[0], index=[k])
                m = m.append(n)
        m.name = 'label'
        m.to_csv('labels.csv', index_label='img', header=True)

    def __getitem__(self, index):
        img, label = self.annot.iloc[index].values
        img = os.path.join(self.img_dir, img+'.jpg')

        if self.transform is not None:
            img = self.transform(Image.open(img))

        return img, label

    def __len__(self):
        return len(self.annot)

