# infoVerse: Dataset Characterization with Multi-dimensional Meta-information

This repository contains code for the paper
**"infoVerse: Dataset Characterization with Multi-dimensional Meta-information"** 
by [Jaehyung Kim](https://sites.google.com/view/jaehyungkim), [Yekyung Kim](https://www.linkedin.com/in/yekyung-kim-b9413a91/), [Karin de Langis](https://karinjd.github.io/), [Jinwoo Shin](http://alinlab.kaist.ac.kr/shin.html) and [Dongyeop Kang](https://dykang.github.io/). 

<p align="center" >
    <img src=assets/acl23_main_figure.jpg width="20%">
</p>

## Installation
The following command installs all necessary packages:
```
pip install -r requirements.txt
```
The project was tested using `Python 3.7`.

## Construction of infoVerse
To construct infoVerse, one first needs to 1) train the vanilla classifiers. Then, using the trained classifiers, one can construct infoVerse by extracting the pre-defined meta-information (defined in `./src/scores_src`). We release the constructed infoVerse at [google drive](https://drive.google.com/file/d/1ARcXikAA7LMwWGEwEf2_rhbQIQyeblAk/view?usp=sharing). Please check out `run.sh`. 


1. Train the classifiers used for gathering meta-informations 
```
python train.py --train_type 0000_base --save_ckpt --epochs 10 --dataset sst2 --seed 1234 --backbone roberta_large
```
2. Construction of infoVerse 
```
python construct_infoverse.py --train_type 0000_base --seed_list "1234 2345 3456" --epochs 10 --dataset sst2 --seed 1234 --backbone roberta_large
```


In addition, one can visualize the constructed infoVerse and use it to analyize the given dataset using `visualize.ipynb`. For example, we provide a code to generate an interactive html file, as shown in the below figure. Pre-constructed tSNE and HTML files can be downloaded from the [google drive](https://drive.google.com/file/d/1N1aaQzUfCOfkmIvaR62FZ9HK0DdsGxNW/view?usp=sharing).

<p align="center" >
    <img src=assets/example_visualization.png width="50%">
</p>

```
visualize.ipynb
```

## Real-world Application #1: Data Pruning

After generating infoVerse, one can conduct data pruning as below. Please check the details in the corresponding repository `./data_pruning`.

```
ln -s transformers ./data_pruning/transformers
python ./data_pruning/train_pruning.py --train_type xxxx_infoverse_dpp --save_ckpt --data_ratio 0.xx --batch_size 16 --epochs 10 --dataset sst2 --seed 1234 --backbone roberta_large
```

## Real-world Application #2: Active Learning

Please see the repository `./active_learning`.

## Real-world Application #3: Data Annotation

Please see the repository `./data_annotation`.
