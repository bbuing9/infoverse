# infoVerse: Dataset Characterization with Multi-dimensional Meta-information

This repository contains code for the paper
**"infoVerse: Dataset Characterization with Multi-dimensional Meta-information"** 
by [Jaehyung Kim](https://sites.google.com/view/jaehyungkim), [Yekyung Kim](https://www.linkedin.com/in/yekyung-kim-b9413a91/), [Karin de Langis](https://karinjd.github.io/), [Jinwoo Shin](http://alinlab.kaist.ac.kr/shin.html) and [Dongyeop Kang](https://dykang.github.io/). 

<p align="center" >
    <img src=assets/acl23_main_figure.jpg width="20%">
</p>


## ToDo Lists

* [ ] Requirements
* [ ] Constructed infoVerse and visualization
* [x] Active learning code re-verification
* [x] Annotation notebook file

## Dependencies

* `python3`
* `pytorch >= 1.6.0`
* `transformers >= 4.0.0`
* `datasets`
* `torchvision`
* `tqdm`
* `scikit-learn`

## Construction of infoVerse
Please check out `run.sh`.

1. Train the classifiers used for gathering meta-informations 
```
python train.py --train_type 0000_base --save_ckpt --epochs 10 --dataset sst2 --seed 1234 --backbone roberta_large
```
2. Construction of infoVerse 
```
python construct_infoverse.py --train_type 0000_base --seed_list "1234 2345 3456" --epochs 10 --dataset sst2 --seed 1234 --backbone roberta_large
```
## Real-world Application #1: Data Pruning

**Remark**. First, one needs to construct infoVerse following the above procedures. 

After that, one can conduct data pruning by controlling `data_ratio` (0.0 to 1.0). Please check out `./data_annotation/run_pruning.sh`. 
```
python ./data_pruning/train_pruning.py --train_type xxxx_infoverse_dpp --save_ckpt --data_ratio 0.xx --batch_size 16 --epochs 10 --dataset sst2 --seed 1234 --backbone roberta_large
```

## Real-world Application #2: Active Learning

Please see the repository `./active_learning`.

## Real-world Application #3: Data Annotation

Please see the repository `./data_annotation`.
