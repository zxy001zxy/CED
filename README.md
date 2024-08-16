<<<<<<< HEAD

## 1. Dataset Download

1. Go to the [1 MEGAPIXEL Event Based Dataset](https://www.prophesee.ai/2020/11/24/automotive-megapixel-event-based-dataset/) and [Prophesee GEN1 Automotive DetectionDataset](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/) to download the datasets. 
2. Unzip the files to get the directory in the following form: 

    ```shell
    # 1 MEGAPIXEL Dataset
    ├── root_for_1MEGAPIXEL_Dataset
    │   ├── Large_Automotive_Detection_Dataset
    │   │   ├── train
    │   │   │   ├── EVENT_STREAM_NAME_td.dat
    │   │   │   ├── EVENT_STREAM_NAME_bbox.npy
    │   │   │   └── ...
    │   │   ├── val
    │   │   │   ├── EVENT_STREAM_NAME_td.dat
    │   │   │   ├── EVENT_STREAM_NAME_bbox.npy
    │   │   │   └── ...
    │   │   ├── test
    │   │   │   ├── EVENT_STREAM_NAME_td.dat
    │   │   │   ├── EVENT_STREAM_NAME_bbox.npy
    │   │   │   └── ...
    
    # GEN1 Dataset
    ├── root_for_GEN1_Dataset
    │   ├── ATIS_Automotive_Detection_Dataset
    │   │	├── detection_dataset_duration_60s_ratio_1.0
    │   │	│   ├── train
    │   │   │	│	├── EVENT_STREAM_NAME_td.dat
    │	│	│   │   ├── EVENT_STREAM_NAME_bbox.npy
    │	│	│   │	└── ...
    │   │	│   ├── val
    │   │   │	│	├── EVENT_STREAM_NAME_td.dat
    │	│	│   │   ├── EVENT_STREAM_NAME_bbox.npy
    │	│	│   │	└── ...
    │   │	│   ├── test
    │   │   │	│	├── EVENT_STREAM_NAME_td.dat
    │	│	│   │   ├── EVENT_STREAM_NAME_bbox.npy
    │	│	│   │	└── ...
    ```

## 3. Dataset Sampling (for 1MEGAPIXEL Dataset)

```shell
python sampling_dataset.py -raw_dir root_for_1MEGAPIXEL_Dataset/Large_Automotive_Detection_Dataset -target_dir root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_sampling
```

## 4. Preprocess

### 4.1 Generate Event Representation

```shell
#Generating Event Representation for 1MEGAPIXEL Dataset(Subset)
python PREPROCESS_FOOTAGE -raw_dir root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_sampling -label_dir root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_sampling -target_dir root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_processed -dataset gen4

#Generating Event Representation for GEN1 Dataset
python PREPROCESS_FOOTAGE -raw_dir root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0 -label_dir root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0 -target_dir root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset_processed -dataset gen1
```



```


## 5. Evaluation


    # Evaluation on 1MEGAPIXEL Dataset(Subset)
    CUDA_VISIBLE_DEVICES="0", python -m torch.distributed.launch --master_port 1403 --nproc_per_node 1 test.py --record True --bbox_path root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_sampling --dataset gen4 --resume_exp EXP_NAME --exp_type EXP_TYPE --event_volume_bins EVENT_VOLUME_BINS  --data_path root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_processed/DATA_DIR 
    
    # Evaluation on GEN1 Dataset
    CUDA_VISIBLE_DEVICES="0", python -m torch.distributed.launch --master_port 1403 --nproc_per_node 1 test.py --record True --bbox_path root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0 --dataset gen1 --resume_exp EXP_NAME --exp_type EXP_TYPE --event_volume_bins EVENT_VOLUME_BINS  --data_path root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset_processed/DATA_DIR 
    ```




## 6. Training from Scratch

```shell
# Training on 1MEGAPIXEL Dataset(Subset)
CUDA_VISIBLE_DEVICES="0", python -m torch.distributed.launch --master_port 1403 --nproc_per_node 1 train.py --bbox_path root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_sampling --dataset gen4 --batch_size 16 --augmentation True --exp_name EXP_NAME --exp_type EXP_TYPE --event_volume_bins EVENT_VOLUME_BINS  --data_path root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_processed/DATA_DIR --nodes 1

# Training on GEN1 Dataset
CUDA_VISIBLE_DEVICES="0", python -m torch.distributed.launch --master_port 1403 --nproc_per_node 1 train.py --bbox_path root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0 --dataset gen1 --batch_size 30 --augmentation True --exp_name EXP_NAME --exp_type EXP_TYPE --event_volume_bins EVENT_VOLUME_BINS  --data_path root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset_processed/DATA_DIR --nodes 1
```

* Resume training: Change "--exp_name EXP_NAME" to" --resume_exp EXP_NAME"
* Distribute training (4 GPUs for example): 
  1. Change "CUDA_VISIBLE_DEVICES="0"" to "CUDA_VISIBLE_DEVICES="0,1,2,3""
  2. Change "--nproc_per_node 1" to "--nproc_per_node 4"
  3. Change "--nodes 1" to "--nodes 4"

## 7. Visualization

```shell
# Visualization on 1MEGAPIXEL Dataset(Subset)
python visualization.py -item EVENT_STREAM_NAME -end ANNOTATION_TIMESTAMP -volume_bins VOLUME_BINS -ecd DATA_DIR -bbox_path root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_sampling -data_path root_for_1MEGAPIXEL_Dataset(Subset)/Large_Automotive_Detection_Dataset_processed -result_path log/EXP_NAME/summarise.npz -datatype DATA_TYPE -suffix DATADIR -dataset gen4

# Visualization on GEN1 Dataset
python visualization.py -item EVENT_STREAM_NAME -end ANNOTATION_TIMESTAMP -volume_bins VOLUME_BINS -ecd DATA_DIR -bbox_path root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0 -data_path root_for_GEN1_Dataset/ATIS_Automotive_Detection_Dataset_processed/DATA_DIR -result_path log/EXP_NAME/summarise.npz -datatype DATA_TYPE -suffix DATADIR -dataset gen1
```


=======
# CED
>>>>>>> 14c69bb6c7a6e2192033d13140823c1d23419735
