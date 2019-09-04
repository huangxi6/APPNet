# APPNet
Attention-based Progressive Partition Network for Human Parsing

This is an efficient implementation of APPNet.
### Download

Plesae download [LIP](http://sysu-hcp.net/lip/overview.php) dataset.

The trained models can be found at [google drive](https://drive.google.com/drive/folders/1jgyQi7rF62dOXyMg2p66ZpsHUqZInX-j).


### Environments

+ Python 3.5   

+ PyTorch 0.4.1  

+ cffi

+ matplotlib

+ numpy        

+ opencv-python

+ scipy

+ tqdm

+ You need to use InPlace-ABN with CUDA implementation, which must be compiled with the following commands:

```bash
cd libs
sh build.sh
python build.py
``` 
+ The model is trained on NVIDIA TITAN 1080 Ti GPU.


### Training

+ Please set the dataset dir in file 'run.sh'. The contents of each dataset include: 

  ─ train_images   

  ─ train_segmentations  

  ─ val_images  

  ─ val_segmentations    

  ─ train_id.txt  

  ─ val_id.txt  

+ Please put the pretrained resnet101-imagenet.pth in './dataset/'.

+ Run the `sh run.sh`. 

### Evaluation

If you want to evaluate the trained models on LIP, you can run the `sh run_evaluate.sh`.

