## Data pruning with infoVerse

**Remark**. First, one needs to construct infoVerse following the procedures in `../infoverse`. For the tested datasets in our paper, one can download the pre-generated infoverse and selecting indices from the google drive. 

After that, one can conduct data pruning by controlling `data_ratio` (0.0 to 1.0). Please check out `./run_pruning.sh`. 
```
python ./train_pruning.py --train_type xxxx_infoverse_dpp --save_ckpt --data_ratio 0.xx --batch_size 16 --epochs 10 --dataset sst2 --seed 1234 --backbone roberta_large
```
