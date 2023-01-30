import json
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import string
import numpy as np
from PIL import Image
from my_build_vocab import Vocabulary, tokenize
from my_coco import COCO

# mostly understood by wjy -----wjy
class AmazonDataset(data.Dataset):
    """Amazon Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, data_dir, vocab, train = True, transform = None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            data_dir: amazon combines_data file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """

        self.root = root
        self.data = json.load(open(data_dir, 'r'))
        # self.ids = list( self.coco.anns.keys() )
        self.vocab = vocab
        self.train = train
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair ( image, title, id )."""
        data = self.data
        vocab = self.vocab
        #列表中取出一个信息字典

        one_data = data[index] 
        title = one_data['title']
        id = one_data['asin']
        
        if self.train == True:
            path_true = self.root + '/train/' + id + '.jpg'
        else:
            path_true = self.root + '/val/' + id + '.jpg'

        #print(self.root) 
        #print(path)
        image = Image.open( path_true ).convert('RGB')
        if self.transform is not None:
            image = self.transform( image )

        # Convert caption (string) to word ids.
        #tokens = str( title ).lower().translate( string.punctuation ).strip().split()
        tokens = tokenize(str(title))
        title = []
        title.append(vocab('<start>'))
        title.extend([vocab(token) for token in tokens])
        title.append(vocab('<end>'))
        # feature的部分可以用<sep>来分割句子
        target = torch.Tensor(title)
        return image, target, id

    def __len__(self):
        return len( self.data )


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
        img_ids: image ids in COCO dataset, for evaluation purpose
        filenames: image filenames in COCO dataset, for evaluation purpose
    """

    # Sort a data list by caption length (descending order).
    data.sort( key=lambda x: len( x[1] ), reverse=True )
    images, titles, img_ids = zip( *data ) # unzip

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    img_ids = list( img_ids )

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(title) for title in titles]
    targets = torch.zeros(len(titles), max(lengths)).long()
    for i, title in enumerate(titles):
        end = lengths[i]
        targets[i, :end] = title[:end]        
    return images, targets, lengths, img_ids


def get_loader(root, data_dir, vocab, train , transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom amazon dataset."""
    # Amazon caption dataset
    amazon = AmazonDataset(root=root,
                       data_dir=data_dir,
                       vocab=vocab,
                       train = train,
                       transform=transform)
    
    # Data loader for Amazon dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=amazon, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
