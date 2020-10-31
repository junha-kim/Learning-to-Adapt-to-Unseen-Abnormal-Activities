Learning to Adapt to Unseen Abnormal Activities under Weak Supervision
=====

This repo is official PyTorch implementation of Learning to Adapt to Unseen Abnormal Activities under Weak Supervision (ACCV 2020).

Jaeyoo Park, Junha Kim, [Bohyung Han](https://cv.snu.ac.kr/index.php/bhhan/)

## Data

* Download following data [link](https://drive.google.com/file/d/1HlOD5eVLXIa_sAz3IqS8w5b4z2SeKrsP/view?usp=sharing) and unzip under your $DATA_ROOT_DIR.
* You can set 'data_root_dir' as an argument in 'options.py'.
* We extract I3D features from raw UCF-Crime videos.
* We follow [this](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018) to make video features into 32 segment features.
* GT_anomaly.pkl: Temporal annotations for all videos.
* exclustion.pkl: We find some of duplicate videos (e.g. same videos but different video name)
* frames.pkl: Number of frames for all videos


You need to follow directory structure of dataset as below.
```  
{$DATA_ROOT_DIR}
|-- {$DATASET NAME}
|   |-- pkl_files
|   |-- {all_rgbs}  
|   |   |-- {$CLASS_NAME}  
|   |   |-- |-- video feature files (.npy)  
|   |-- {all_flows}  
|   |   |-- same structures as {all_rgbs}  
|   |-- {splits(only for UCF-Crime)}  
```  
For details, please check the downloaded data.

## Run
* 'seed' is used for selecting target class (e.g. 1 for Abuse) of UCF-Crime dataset
* All arguments are in options.py.
* Simple running command is as follows.
1. pretrain: python main.py --mode pretrain --dataset $DATASET_NAME --seed $CLASS_NUM --save_chpt
2. meta-train: python main.py --mode meta_train --dataset $DATASET_NAME --seed $CLASS_NUM --save_chpt
3. meta-test
  * Scratch: python main.py --mode eval --dataset $DATASET_NAME --seed $CLASS_NUM
  * Pretrain: python main.py --mode eval --dataset $DATASET_NAME --seed $CLASS_NUM --chpt $NAME_OF_CHECKPOINT_BY_PRETRAIN
  * Meta-train: python main.py --mode eval --dataset $DATASET_NAME --seed $CLASS_NUM --chpt $NAME_OF_CHECKPOINT_BY_METATRAIN --sampling
    * For meta-test, chpt format is like '{}epochs_exp0_seed1_lr1e-5_split1.pkl'.
    
## Reference
```  
@InProceedings{,  
author = {Park, Jaeyoo, Kim, Junha, and Han, Bohyoung},  
title = {Learning to Adapt to Unseen Abnormal Activities under Weak Supervision},  
booktitle = {Asian Conference on Computer Vision (ACCV)},  
year = {2020}  
}  
```
