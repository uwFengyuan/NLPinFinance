import math
import json
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from utils_mg import coco_eval, to_var
from dataloader_mg import get_loader 
from adaptive_mg import Encoder2Decoder
from build_vocab import Vocabulary
from torch.autograd import Variable 
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence

#mostly understood by wjy -----wjy
def main(args):
    
    # To reproduce training results
    torch.manual_seed( args.seed )
    if torch.cuda.is_available():
        torch.cuda.manual_seed( args.seed )
        
    # Create model directory
    if not os.path.exists( args.model_path ):
        os.makedirs(args.model_path)
    
    # Image Preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([ 
        transforms.RandomCrop( args.crop_size ), # 224
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(( 0.485, 0.456, 0.406 ), 
                             ( 0.229, 0.224, 0.225 ))])
    
    # Load vocabulary wrapper.
    # print('-'*20 + 'Load Vocabulary Wrapper' + '-'*20)
    with open( args.vocab_path, 'rb') as f: # vocab.pkl
        vocab = pickle.load( f )
    
    # Build training data loader
    data_loader = get_loader( args.image_dir, args.caption_path, vocab, # resized, karpathy_split_train.json 建立一个json文件train
                              transform, args.batch_size,
                              shuffle=True, num_workers=args.num_workers ) 

    # Load pretrained model or build from scratch
    adaptive = Encoder2Decoder( args.embed_size, len(vocab), args.hidden_size ) # Adaptive: (256, len(vocab), 512)
    
        
    adaptive.load_state_dict( torch.load( '/home/liufengyuan/NLPinFinance/NLP_For_E_commerce_main/data/models/adaptive-Sports_and_Outdoors(Summarize)1.pkl' ) )
    # Get starting epoch #, note that model is named as '...your path to model/algoname-epoch#.pkl'
    # A little messy here.

        
    # Evaluation on validation set 
    cider_scores = []
    best_cider = 0.0
    best_epoch = 0       
    cider = coco_eval( adaptive, args, 1 )
    cider_scores.append( cider )        
    
    if cider > best_cider:
        best_cider = cider
        best_epoch = 1
    
    if len( cider_scores ) > 5:
        
        last_6 = cider_scores[-6:]
        last_6_max = max( last_6 )
        
        # Test if there is improvement, if not do early stopping
        if last_6_max != best_cider:
            
            print ('No improvement with CIDEr in the last 6 epochs...Early stopping triggered.')
            print ('Model of best epoch #: %d with CIDEr score %.2f'%( best_epoch, best_cider ))
            
            
            
if __name__ == '__main__':
    
    data_source = 'Five_Categories_Data/Sports_and_Outdoors/tokenized'
    image_source = 'LFYdata'
    parser = argparse.ArgumentParser()
    parser.add_argument( '-f', default='self', help='To make it runnable in jupyter' )
    parser.add_argument( '--model_path', type=str, default='/home/liufengyuan/NLPinFinance/NLP_For_E_commerce_main/data/models',
                         help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='/data/liufengyuan/NLPinFinance/' + data_source + '/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='/data/liufengyuan/NLPinFinance/' + image_source + '/resized' ,
                        help='directory for resized training images')
    parser.add_argument('--caption_path', type=str,
                        default='/data/liufengyuan/NLPinFinance/' + data_source + '/annotations/karpathy_split_train.json',
                        help='path for train annotation json file')
    parser.add_argument('--caption_val_path', type=str,
                        default='/data/liufengyuan/NLPinFinance/' + data_source + '/annotations/karpathy_split_val.json',
                        help='path for validation annotation json file')
    parser.add_argument('--log_step', type=int, default=50,
                        help='step size for printing log info')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed for model reproduction')
    
    # ---------------------------Hyper Parameter Setup------------------------------------
    
    # CNN fine-tuning
    parser.add_argument('--fine_tune_start_layer', type=int, default=5,
                        help='CNN fine-tuning layers from: [0-7]')
    parser.add_argument('--cnn_epoch', type=int, default=20,
                        help='start fine-tuning CNN after')
    
    # Optimizer Adam parameter
    parser.add_argument( '--alpha', type=float, default=0.8,
                         help='alpha in Adam' )
    parser.add_argument( '--beta', type=float, default=0.999,
                         help='beta in Adam' )
    parser.add_argument( '--learning_rate', type=float, default=4e-4,
                         help='learning rate for the whole model' )
    parser.add_argument( '--learning_rate_cnn', type=float, default=1e-4,
                         help='learning rate for fine-tuning CNN' )
    
    # LSTM hyper parameters
    parser.add_argument( '--embed_size', type=int, default=256,
                         help='dimension of word embedding vectors, also dimension of v_g' )
    parser.add_argument( '--hidden_size', type=int, default=512,
                         help='dimension of lstm hidden states' )
    
    # Training details
    parser.add_argument( '--pretrained', type=str, default='' )
    parser.add_argument( '--num_epochs', type=int, default=5 )
    parser.add_argument( '--batch_size', type=int, default=30) # on cluster setup, 60 each x 4 for Huckle server
    
    # For eval_size > 30, it will cause cuda OOM error on Huckleberry
    parser.add_argument( '--eval_size', type=int, default=2 ) # on cluster setup, 30 each x 4
    parser.add_argument( '--num_workers', type=int, default=4 )
    parser.add_argument( '--clip', type=float, default=0.1 )
    parser.add_argument( '--lr_decay', type=int, default=20, help='epoch at which to start lr decay' )
    parser.add_argument( '--learning_rate_decay_every', type=int, default=50,
                         help='decay learning rate at every this number')
    
    args = parser.parse_args()
    
    print ('------------------------Model and Training Details--------------------------')
    print(args)
    
    # Start training
    main( args )
