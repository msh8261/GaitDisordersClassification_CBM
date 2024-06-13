""" export CUBLAS_WORKSPACE_CONFIG=:4096:8 """
 
params = dict(
    train_dataset_path="./data/70_zero_padding_no",  # 70 features
    sequences=300,
    # 70 features
    input_size=70,
    # three exercise as three classes
    num_class=3,
    # models_name = ["gan", "gan_genomaps", "gan_deepInsight", 'deepInsight', 'gnClassify', "vit_mlp"],
    # gan: for proposed model, gan_genomaps: for genomap train with proposed model, gan_deepInsight: deepInsight train with proposed model
    # gnClassify: for genomap train with genoNet, deepInsight: for deepInsight train with resNet
    models_name=["vit_mlp"],
    # "18", "50", "152"
    resnet_type="152",  # only for deepInsight train with resNet
    # for deepInsight only
    img_size1=70,
    img_size2=70,
    # for genomap models only [36, 50] 
    # for gan_genomaps: 50 and for gnClassify: 36
    genmap_colNum=36,
    genmap_rowNum=36,
    # 0-6 -->> 0 means no augmentation, 5 is best
    num_augmentation=5,
    batch_size=16,
    epochs=5,  
    ## 'relu', 'sigmoid', 'tanh', 'selu','elu',
    ## 'LeakyReLU','rrelu'
    acf_indx="LeakyReLU",
    ## 'Adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad',
    ## 'Adamax','Adamw', 'AdaBound'
    opt_indx="Adamw",
    last_layer=64,
    dropout=0.1,
    bottleneck=70,
    learning_rate=1e-3,  # 2e-3 for deepinsigt image with gan
    weight_decay=1e-5,
    lr_decay=0.1,
    n_folds=5,
    random_state_list=[21],
)
