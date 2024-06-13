# Gait disorder classification based on effective feature selection and unsupervised methodology


### windows installation: <br />
- pipenv install <br />


### set config_data 
- set parameters and path of data for data prepration

### prepare dataset
- 1) to make files with features run: python prepare_dataset/1_get_features_frames.py <br />
- 2) to concatenate three exercises in a file run: python prepare_dataset/2_concat_exercises.py <br />
- 3) to split data for train and test sets run: python prepare_dataset/3_data_split.py <br />
- 4) to make final datassets run: python prepare_dataset/4_data_for_train.py <br />


### set config_train
- set parameters and model name for train model

### train 
- check there are Xtrain.File, Xtest.File, ytrain.File, ytest.File in data folder <br />
- run: python src/train.py <br />

### taining results
- saved_models folder will included trained models

### test trained models
- check the trained models are in saved_models folder <br />
- for testing proposed model run: python src/test.py <br />
- for testing vit model run: python src/test.py <br />
- for testing deepInsight trained with proposed model run: python src/test_deepInsight.py <br />
- for testing deepInsight trained with resNet152 run: python src/test_resnet152.py <br />
- for testing genomap trained with proposed model run: python src/test_genomap.py <br />
- for testing genomap trained with genoNet model run: python src/test_gnClassify.py <br />

### testing results
- results folder will included the reuslts files