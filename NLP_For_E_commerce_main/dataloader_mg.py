import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import string
import numpy as np
from PIL import Image
from build_vocab import Vocabulary
from coco import COCO

content = 'caption'

# mostly understood by wjy -----wjy
class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """

        self.root = root
        self.coco = COCO( json )
        self.ids = list( self.coco.anns.keys() )
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair ( image, description, caption, image_id )."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        description = coco.anns[ann_id]['description']
        caption = coco.anns[ann_id][content]
        img_id = coco.anns[ann_id]['image_id']
        #print(img_id)
        #print(coco.loadImgs(img_id))
        filename = coco.loadImgs(img_id)[0]['file_name']

        #if 'train' in filename.lower():
        #    path = '/train2014/' + filename
        #else:
        #    path = '/val2014/' + filename
        path = '/LFY_2014/' + filename
        #print(self.root)
        #print(path)
        path_true = self.root + path
        #if os.path.exists(csv_file_path)
        image = Image.open( path_true ).convert('RGB')
        if self.transform is not None:
            image = self.transform( image )

        # Convert caption (string) to word ids.
        tokens = str( caption ).lower().translate( string.punctuation ).strip().split()
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        caption = torch.Tensor(caption)

        tokens_2 = str( description ).lower().translate( string.punctuation ).strip().split()
        description = []
        description.append(vocab('<start>'))
        description.extend([vocab(token) for token in tokens_2])
        description.append(vocab('<end>'))
        description = torch.Tensor(description)
        return image, description, caption, img_id, filename

    def __len__(self):
        return len( self.ids )


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
    images, descriptions, captions, img_ids, filenames = zip( *data ) # unzip

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    img_ids = list( img_ids )
    filenames = list( filenames )

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths_des = [len(des) for des in descriptions]
    targets_des = torch.zeros(len(descriptions), max(lengths_des)).long()
    for i, des in enumerate(descriptions):
        end = lengths_des[i]
        targets_des[i, :end] = des[:end]  

    lengths_cap = [len(cap) for cap in captions]
    #lengths_cap = [20 for cap in captions]
    targets_cap = torch.zeros(len(captions), max(lengths_cap)).long()
    for i, cap in enumerate(captions):
        end = lengths_cap[i]
        targets_cap[i, :end] = cap[:end]     
    # print(max(lengths))
    return images, targets_des, targets_cap, lengths_des, lengths_cap, img_ids, filenames


def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
