# infoVerse: Dataset Characterization with Multi-dimensional Meta-information

This repository contains code for the paper
**"infoVerse: Dataset Characterization with Multi-dimensional Meta-information"** 
by [Jaehyung Kim](https://sites.google.com/view/jaehyungkim), [Yekyung Kim](https://www.linkedin.com/in/yekyung-kim-b9413a91/), [Karin de Langis](https://karinjd.github.io/), [Jinwoo Shin](http://alinlab.kaist.ac.kr/shin.html) and [Dongyeop Kang](https://dykang.github.io/). 

## Dependencies

* `python3`
* `pytorch >= 1.6.0`
* `transformers >= 4.0.0`
* `datasets`
* `torchvision`
* `tqdm`
* `scikit-learn`

## Construction of InfoVerse
1. Train the classifiers used for for gathering meta-informations 
```
python train.py --train_type 0000_base --save_ckpt --epochs 7 --dataset sst2 --seed 1234 --backbone roberta_large
```
2. Construction of infoVerse 
```
python construct_infoverse.py --train_type 0000_base --epochs 7 --dataset sst2 --seed 1234 --backbone roberta_large
```
## Real-world Application #1: Data Pruning

Please check out `run.sh` for all the scripts to reproduce the results.

## Real-world Application #2: Active Learning

Please check out `run.sh` for all the scripts to reproduce the results.

## Real-world Application #3: Data Annotation

Please check out `run.sh` for all the scripts to reproduce the results.

