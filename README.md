# Setting up environment
## Install conda.
Install miniconda or anaconda, please refer to this webpage for installation: 
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
## Create the virtual environment using conda.
The environment requirements are specified in `environment.yml` file. Please modify the path
before running the following command.

```
conda env create -f environment.yml
```

# The basics
The project uses the attention-based Bi-LSTM model introduced in the paper 
"ATTENTION-BASED LSTM FOR PSYCHOLOGICAL STRESS DETECTION FROM SPOKEN LANGUAGE USING DISTANT SUPERVISION"
IEEE ICASSP 2018 (Genta Indra Winata, Onno Pepijn Kampman, Pascale Fung).


### Datasets
1. Download UCI dataset from Kaggle https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset, and save the dataset to folder `data/UCI`.

2. JP Morgan fraud detection dataset which does not contain sensitive attributes, download data from https://www.jpmorgan.com/technology/artificial-intelligence/initiatives/synthetic-data/payments-data-for-fraud-detection
and save it to `data/JP`. 
3. Fraud detection dataset from Kaggle has sensitive attributes like age and gender, download
 from https://www.kaggle.com/datasets/ealaxi/banksim1?datasetId=1565&sortBy=voteCount and save it to `data/kaggle_fraud_detection_age_gender`.

### Data preprocessing
First need to do some data preprocessing.
1. UCI dataset: `Home_Credit_data_preprocess.py`.
2. JP Morgan fraud detection  dataset: `JP_data_preprocess.py`.
3. Fraud detection dataset from Kaggle: there is not much to preprocess.

### Code
The attention-based Bi-LSTM model is defined in `lstm_models.py`.
Since we are dealing with tabular data which contains a lot of numbers, we need to 
convert the numbers to strings, to do this, we convert numbers to different buckets. 
For example, we create a bucket for every $20000, then numbers like $20510 or $$20010
will be converted to the same bucket, and $1 or $ 501 will be in another bucket. 
This might not be the best way, but this is what we are using for now.

To code for training and testing the model on different dataset:
1. UCI dataset: `lstm_attn/UCI_train_and_get_attn.py`.
2. JP Morgen fraud dataset: `lstm_attn/JP_train_and_get_attn.py`
3. Fraud detection dataset: `fraud_detect_age_gender_lstm_attn.py`.

### Run the code
Example:
```
python fraud_detect_age_gender_lstm_attn.py \
--input_data_path=../data/UCI/feature_prefix_bin_stride_20000 \
--attn_data_path=../results/UCI/feature_prefix_bin_stride_20000_new/attn_scores.npy \
--model_type=LSTM --is_attention=True --is_finetune=False \
--hidden_units=64 --num_layers=1 --is_bidirectional=True \
--max_sequence_length=23 \
--validation_split=0.2 \
--save_path=../checkpoint/UCI/feature_prefix_bin_stride_20000_new_attn_no_personal_feature/ckpt \
--epochs 100 \
--result_dir=../results/UCI/feature_prefix_bin_stride_20000_new_attn_no_personal_feature
```
This trains/test the model on the fraud detection dataset which contains the sensitive
attributes. 
`--input_data_path` specifies the input path of the preprocessed data.

`--attn_data_path` specifies where the attention scores are saved to.

`--max_sequence_length` should match with the number of features.

