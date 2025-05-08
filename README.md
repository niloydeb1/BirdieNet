# BirdieNet
BirdieNet: Fine-grained Classification of Bird Species with Multi-Branch Attention Network

## Directory:
/
│</br>
├── cvt-setupA.py # Training with Setup A
├── cvt-setupB.py # Training with Setup B
├── eval_bird525.py # Evaluation script for Birdsnap dataset
├── eval_bird525.sh # Shell script to run eval_bird525.py with arguments
├── Datasets/ # Folder containing CUB-200-2011 and Birdsnap dataset
├── utils/ # Folder containing helper scripts
│ ├── attention_cvt4_all.py
│ ├── attention_cvt4.py
│ ├── object_crops.py
│ └── part_crops.py


## Install the required packages:
pip install torch torchvision transformers timm scikit-learn datasets tqdm

## Training with Setup A
python cvt-setupA.py <model_version> <learning_rate>

- model_version: 13, 21, or 24 (CvT variant)
- learning_rate: e.g., 0.0001 or 0.001

## Training with Setup B
python cvt-setupB.py <model_version> <stride_key_value> <clssification_dropout_rate>

- model_version: 13, 21, or 24
- stride_kv: 1 or 2 (stride for key/value projections)
- cls_drop: Dropout rate in the classification head (e.g., 0.5)

## Datasets
You can either download or use the datasets online from the following links:
1. https://www.kaggle.com/datasets/wenewone/cub2002011
2. https://huggingface.co/datasets/sasha/birdsnap
3. https://huggingface.co/docs/hub/datasets-usage

*Uploading the dataset was taking an awfully lot of times, therefore we skipped uploading.*
