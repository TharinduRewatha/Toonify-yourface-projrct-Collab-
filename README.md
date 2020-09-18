Train Your Own StyleGAN2 Using Colab

Various Improvements to make StyleGAN2 more suitible to be trained on Google Colab

Supports Non-Square images, for example, 768x512, which basically as 6x4 (x2^7), or 640x384 as 5x3 (x2^7), etc.
Supports vertical mirror augmentation
Supports train from latest pkl automatically
Optimized dataset creation and access for non-progressive training and for colab training, which includes: create only the maximum size tfrecord; use raw JPEG instead of decoded numpy array, which reduce both tfrecord creation time and dataset size dramatically. (* Only tested for config-e and config-f, as no-progressive for these configurations)
Detailed instruction for training your stylegan2

Create training image set. Instead of image size of 2^n * 2^n, now you can process your image size as of (min_h x 2^n) X (min_w * 2^n) natually. For example, 640x384, min_h = 5, min_w =3, n=7. Please make sure all your raw images are preprocessed to the exact same size. To reduce the training set size, JPEG format is preferred.
Create tfrecord, clone this repo, then
python dataset_tool.py create_from_images_raw dataset_dir raw_image_dir
To train, for example, 640x384 training set
python run_training.py --num-gpus=your_gpu_num --data-dir=your_data_dir --config=config-e(or config_f) --dataset=your_data_set --mirror-augment=true --metric=none --total-kimg=12000 --min-h=5 --min-w=3 --res-log2=7 --result-dir=your_result_dir
Tips for Colab training

Clone this repo
%tensorflow_version 1.x
import tensorflow as tf

# Download the code
!git clone https://github.com/TharinduRewatha/Toonify-yourface-projrct-Collab-

You may also try this to boost your instance memory before training.

https://github.com/googlecolab/colabtools/issues/253

For image size 1280x768 (hxw), you may choose (min_h, min_w, res_log2) as (10, 6, 7) or (5, 3, 8) , the latter setup is preferred due to deeper and smaller network, change res_log2 argument for dataset creation and training accordingly.
!python dataset_tool.py create_from_images_raw --res_log2=8 ./dataset/dataset_name untared_raw_image_dir
!python run_training.py --num-gpus=1 --data-dir=./dataset --config=config-f --dataset=your_dataset_name --mirror-augment=true --metric=none --total-kimg=20000 --min-h=5 --min-w=3 --res-log2=8 --result-dir="/content/drive/My Drive/stylegan2/results"
You may change relevant arguments in run_traing.py for fakeimage/checkpoint interval, D/G learning rate, and minibatch_gpu_base to suit your needs or workaround gpu memory issues.

Stylegan2 removed the capability of setting learning rate of G/D seperately. Added it back but did not test much, you may use --dlr and --glr to overwrite the D/G's base learning rate

Added google attention option to certain D/G layer. You may try it with specifying --use-attention=true

if the tfrecord is created using create_from_images (instead of create_from_images_raw), please specify --use-raw=false

Also, exposed resume_with_new_nets to command line. The example usage would be, network trained without attention, but now you want to try the network with attention module, you can specify --resume_with_new_nets=true to copy weights from checkpoints.

Credits

https://github.com/NVlabs/stylegan2
https://github.com/akanimax/msg-stylegan-tf
https://github.com/TharinduRewatha
https://github.com/justinpinkney/stylegan2
https://colab.research.google.com/drive/1ShgW6wohEFQtqs_znMna3dzrcVoABKIH
