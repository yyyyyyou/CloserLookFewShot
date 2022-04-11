
# Few-shot Classification with SoftTriple Loss

Authors:
Yuzhe You, Zhen Gong

In this study, we investigate the feasibility of combining the SoftTriple loss (Qian et al., 2019) into the general structure of Baseline++ model proprosed in (Chen et al., 2019). We denote our models as BaselineST and BaselineST+.

The BaselineST model trains a feature extractor and a Baseline++ classifier with
a hybrid loss function that combines the standard cross-entropy loss and the SoftTriple loss, while the BaselineST+ model utilizes only the SoftTriple loss for both the training and fine-tuning stage.

## Citation of used resources:
```
@inproceedings{
chen2019closerfewshot,
title={A Closer Look at Few-shot Classification},
author={Chen, Wei-Yu and Liu, Yen-Cheng and Kira, Zsolt and Wang, Yu-Chiang and  Huang, Jia-Bin},
booktitle={International Conference on Learning Representations},
year={2019}
}

@inproceedings{qian2019striple,
  author    = {Qi Qian and
               Lei Shang and
               Baigui Sun and
               Juhua Hu and
               Hao Li and
               Rong Jin},
  title     = {SoftTriple Loss: Deep Metric Learning Without Triplet Sampling},
  booktitle = {{IEEE} International Conference on Computer Vision, {ICCV} 2019},
  year      = {2019}
}
```

# A Closer Look at Few-shot Classification

This repo contains the reference source code for the paper [A Closer Look at Few-shot Classification](https://arxiv.org/abs/1904.04232) in International Conference on Learning Representations (ICLR 2019). In this project, Chen et al provide a integrated testbed for a detailed empirical study for few-shot classification.


## Citation
If you find Chen et al's code useful, please consider citing their work using the bibtex:
```
@inproceedings{
chen2019closerfewshot,
title={A Closer Look at Few-shot Classification},
author={Chen, Wei-Yu and Liu, Yen-Cheng and Kira, Zsolt and Wang, Yu-Chiang and  Huang, Jia-Bin},
booktitle={International Conference on Learning Representations},
year={2019}
}
```

## Environment
 - Python3
 - [Pytorch](http://pytorch.org/) before 0.4 (for newer vesion, please see issue #3 )
 - json

## Getting started
### CUB
* Change directory to `./filelists/CUB`
* run `source ./download_CUB.sh`

### mini-ImageNet
* Change directory to `./filelists/miniImagenet`
* run `source ./download_miniImagenet.sh`

(WARNING: This would download the 155G ImageNet dataset. You can comment out correponded line 5-6 in `download_miniImagenet.sh` if you already have one.)

### mini-ImageNet->CUB (cross)
* Finish preparation for CUB and mini-ImageNet and you are done!

### Omniglot
* Change directory to `./filelists/omniglot`
* run `source ./download_omniglot.sh`

### Omniglot->EMNIST (cross_char)
* Finish preparation for omniglot first
* Change directory to `./filelists/emnist`
* run `source ./download_emnist.sh`  

### Self-defined setting
* Require three data split json file: 'base.json', 'val.json', 'novel.json' for each dataset  
* The format should follow   
{"label_names": ["class0","class1",...], "image_names": ["filepath1","filepath2",...],"image_labels":[l1,l2,l3,...]}  
See test.json for reference
* Put these file in the same folder and change data_dir['DATASETNAME'] in configs.py to the folder path  

## Train
Run
```python ./train.py --dataset [DATASETNAME] --model [BACKBONENAME] --method [METHODNAME] [--OPTIONARG]```

For example, run `python ./train.py --dataset miniImagenet --model Conv4 --method baseline --train_aug`  
Commands below follow this example, and please refer to io_utils.py for additional options.

## Save features
Save the extracted feature before the classifaction layer to increase test speed. This is not applicable to MAML, but are required for other methods.
Run
```python ./save_features.py --dataset miniImagenet --model Conv4 --method baseline --train_aug```

## Test
Run
```python ./test.py --dataset miniImagenet --model Conv4 --method baseline --train_aug```

## Results
* The test results will be recorded in `./record/results.txt`
* For all the pre-computed results, please see `./record/few_shot_exp_figures.xlsx`. This will be helpful for including your own results for a fair comparison.

## References
Chen et al's testbed builds upon several existing publicly available code. Specifically, they have modified and integrated the following code into this project:

* Framework, Backbone, Method: Matching Network
https://github.com/facebookresearch/low-shot-shrink-hallucinate
* Omniglot dataset, Method: Prototypical Network
https://github.com/jakesnell/prototypical-networks
* Method: Relational Network
https://github.com/floodsung/LearningToCompare_FSL
* Method: MAML
https://github.com/cbfinn/maml  
https://github.com/dragen1860/MAML-Pytorch  
https://github.com/katerakelly/pytorch-maml

## FAQ
For this FAQ, please refer to issues from Chen et al's original github repo (https://github.com/wyharveychen/CloserLookFewShot).

* Q1 Why some of my reproduced results for CUB dataset are around 4~5% different with you reported result? (#31, #34, #42)
* A1 My reported results on the paper may run in different epochs or episodes, please see each issue for details.

* Q2 Why some of my reproduced results for mini-ImageNet dataset are around 1~2% different with your reported results? (#17, #40, #41 #43)
* A2 Due to random initialization, each training process could lead to different accuracy. Also, each test time could lead to different accuracy.

* Q3 How do you decided the mean and the standard variation for dataset normalization? (#18, #39)
* A3 I use the mean and standard variation from ImageNet, but you can use the ones calculated from your own dataset.

* Q4 Do you have the mini-ImageNet dataset available without downloading the whole ImageNet? (#45 #29)
* A4 You can use the dataset here https://github.com/oscarknagg/few-shot, but you will need to modify filelists/miniImagenet/write_miniImagenet_filelist.py.
