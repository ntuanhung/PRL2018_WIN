# PRL2018_WIN
## Overview
- This repository contains source code of the paper "Text-Independent Writer Identification using Convolutional Neural Network".
Read paper [here](https://www.sciencedirect.com/science/article/pii/S0167865518303180)

- If you want to refer to our paper, please use citation as follows:

Hung Tuan Nguyen, Cuong Tuan Nguyen, Takeya Ino, Bipin Indurkhya, Masaki Nakagawa,
*Text-independent writer identification using convolutional neural network*,
Pattern Recognition Letters, 2018, ISSN 0167-8655, https://doi.org/10.1016/j.patrec.2018.07.022.

## Notes
- Inside *images* folder, all image file should be named by <writer_id>_<image_name> such as 0_1234.

- In *configs* folder, there are different configuration files which are used to train/evaluate model.
Each configuration has three text files for train/valid/test sets.

```
train-files_<number_of_writers>users_<number_of_patterns_per_writer>patPerUser_SAME_<database_name>.txt
valid-files_<number_of_writers>users_<number_of_patterns_per_writer>patPerUser_SAME_<database_name>.txt
 test-files_<number_of_writers>users_<number_of_patterns_per_writer>patPerUser_SAME_<database_name>.txt
```

- Each configuration file has <number_of_writers> lines. For every line, the first value is <writer_id> which is followed by <number_of_patterns_per_writer> values represented for <image_name>.

## Train model
Run the following command to train network

```
python win5_subimage_classification.py --num_writers 10 --num_training_patterns 100 --agg_mode average --n_tuple 10
```

If you want to train using GPU, append the following argument

```
--gpu <gpu_id>
```

## Evaluate model
Run the following command to evaluate the trained model in *models* folder
For example, the evaluate model contains these following files:
```
  win5_subimg_average@2017.09.22-01.02.03_bestAcc.ckpt-100.data-00000-of-00001
  win5_subimg_average@2017.09.22-01.02.03_bestAcc.ckpt-100.meta
  win5_subimg_average@2017.09.22-01.02.03_bestAcc.ckpt-100.index
```
<model_filename> should be win5_subimg_average@2017.09.22-01.02.03 and <global_step_eval> is 100.

```
python win5_subimage_classification.py --training False --model_name <model_filename> --global_step_eval 100 --num_writers 10 --num_training_patterns 100 --agg_mode average --n_tuple 10
```

## Description of arguments
Argument | Type | Description
-------- | ---- | -----------
**Fix arguments** | |
--author_name | str | author name
--directory_path, -dp | str | path to PRL2018_WIN
--dataset_name, -dn | str | dataset name
--img_size, -im | int | image size
--image_type, -it | str | image type includes BIN, RGB
--selection_mode, -sm | str | selection mode such as SAME or DIFF
**Configuring arguments** |  | 
--num_writers, -nw|int| number of writers
--num_training_patterns, -ntp|int| number of training patterns per writer
--local_feature, -lf|str|type of local feature type: subimg
--agg_mode, -am|str|different aggregation modes: average, max, kmax
--kmax, -k|int|value of *k* in kmax aggregation mode
--n_tuple, -nt|int|tuple size
--train_num_permutations, -trnp|int| number of permutations per epoch during training
--valid_num_permutations, -vanp|int| number of permutations per epoch during validating
--test_num_permutations, -tenp|int| number of permutations per epoch during testing
--eval_num_permutations', -evnp|int| number of permutations per epoch during evaluating
**Training arguments**||
--lr, -l|float|learning rate
--max_epochs, -me|int| maximum number of epochs for training
--max_no_best, -mnb|int| early stop if accuracy does not increase during <max_no_best> epochs
--writer_per_batch, -wpb|int|number of writer per minibatch
--gpu, -g|int| gpu_id is used 
--gpu_mem_ratio, -gmr| float| memory ratio of gpu from 0.0 to 1.0
**Evaluation/Training arguments**||
--training, -t|bool|if *evaluate* model using False, else using default (True) for *training*
--resume, -r|str|if *resume* training process, pass the model filename
--global_step_start, -gss|int| if *resume* training process, pass the model *global_step* to continue training
--model_name, -mn|str|if *evaluate* model, pass the model filename.
--global_step_eval, -gse|int|if *evaluate* model, pass the model *global_step* to evaluate
--use_valid_data, -uvd|bool|use valid data or not while training model
--use_test_data, -utd|bool|use test data or not while training model
--eval_test_data, -etd|bool|use test data or not while evaluating model
