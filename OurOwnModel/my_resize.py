import argparse
import os
from PIL import Image
import multiprocessing as mp

def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.Resampling.LANCZOS)

def transfer_image(image_dir, image, size, output_dir):
    with open(os.path.join(image_dir, image), 'r+b') as f:
        with Image.open(f) as img:
            img = resize_image(img, size)
            img.save(os.path.join(output_dir, image), img.format)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)

    pool = mp.Pool(48)
    for i, image in enumerate(images):
        pool.apply_async(transfer_image, args = (image_dir, image, size, output_dir))
        if i % 1000 == 0:
            print ("[%d/%d] Resized the images and saved into '%s'."
                   %(i, num_images, output_dir))
    pool.close()
    pool.join()

def main(args):
    splits = [ 'train', 'val' , 'test']
    #years = ['2014']
    
    if not os.path.exists( args.output_dir ):
        os.makedirs( args.output_dir )
        
    for split in splits:
        #for year in years:
            
            # build path for input and output dataset
        dataset = split #+ year
        image_dir = os.path.join( args.image_dir, dataset )
        output_dir = os.path.join( args.output_dir, dataset )
        
        image_size = [ args.image_size, args.image_size ]
        resize_images( image_dir, output_dir, image_size )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='/data/liufengyuan/NLPinFinance/trainvalImage',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='/data/liufengyuan/NLPinFinance/Resized_Image',
                        help='directory for saving resized images')
    
    parser.add_argument('--image_size', type=int, default=256, # for cropping purpose
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)
