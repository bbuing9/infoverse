## Data Annotation with infoVerse

### Preliminary

One can download the used dataset (SST5 and IMP) from the [google drive](https://drive.google.com/file/d/1F1giIVdzHrRcib9NZwJCdba1xyBQ71Xi/view?usp=sharing). Then, one should move a folder for each dataset (named `sst5` and `imp`) into the `dataset` folder at the parent folder (`infoverse`).

### Experiments

1. Train the classifiers used for gathering meta-informations of unlabeled samples. 
```
python ./train_anno.py --train_type xxxx_base --save_ckpt --epochs 10 --dataset imp --seed 1234 --backbone roberta_large
```

2. Selecting unlabeled samples from the unlabeled data pool (`data_annotation_before_AMT.ipynb`). Then, using [AMT](https://www.mturk.com/) or other crowd-sourced labeling system, labeling those samples (`data_annotation_after_AMT.ipynb`). One can download the all materials (unlabeled data pool, selection index, crowd-sourcing results, finally annotated datasets) from the [google drive](https://drive.google.com/file/d/1M919ZhkWIMZVAPRIsexJJuZCGwD9Pann/view?usp=sharing).   
```
data_annotation_before_AMT.ipynb
data_annotation_after_AMT.ipynb
```

3. Train the classifier with annotated samples  
```
python ./train_anno.py --annotation infoverse --save_ckpt --train_type xxxx --epochs 10 --dataset imp --seed 1234 --backbone roberta_large
```
