# import 
import argparse
from easydict import EasyDict as edict
import numpy as np
import random
import yaml
from collections import Counter
from sktime.classification.deep_learning import InceptionTimeClassifier, ResNetClassifier, CNNClassifier
from sktime.regression.deep_learning import CNNRegressor, FCNRegressor
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError



# tensorflow
import tensorflow as tf



# custom
from train import Trainer
from dataloader import show_batch, DataLoader
from model import get_model


# Parameters
with open("/Users/doheonkim/Desktop/LBBB/Etc/config.yaml", "r") as f:
    cfg = edict(yaml.safe_load(f))
RANDOM_SEED = cfg.dataloader.random_seed
xml_folder = cfg.dataloader.xml_folder
csv_file = cfg.dataloader.csv_file

#init_lr = cfg.train_param.init_lr
num_epochs = cfg.train_param.num_epochs
batch_size = cfg.train_param.batch_size
save_path = cfg.train_param.save_path
model_name = cfg.train_param.model
types = cfg.train_param.types

# args
parser = argparse.ArgumentParser()
parser.add_argument("--phase")
parser.add_argument("--model", default="gbm", choices=["gbm", "lstm", "transformer", "inception", "hivecote", "lightgbm"])
parser.add_argument("--checkpoint-dir", type=str)
args = parser.parse_args()
print(args)



# Seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# hyperparameter
model = {
            "resnet" : get_model(2),
        }

def calculate_receptive_field(layers):
    receptive_field = 1
    stride = 1
    print(f"{'Layer Name':<20} {'Kernel Size':<15} {'Stride':<10} {'Receptive Field'}")
    print("="*60)
    for layer in layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            kernel_size = layer.kernel_size[0]
            layer_stride = layer.strides[0]
            receptive_field = receptive_field + (kernel_size - 1) * stride
            stride *= layer_stride
            print(f"{layer.name:<20} {kernel_size:<15} {layer_stride:<10} {receptive_field}")
        elif isinstance(layer, tf.keras.layers.AveragePooling1D):
            pool_size = layer.pool_size[0]
            pool_stride = layer.strides[0]
            receptive_field = receptive_field + (pool_size - 1) * stride
            stride *= pool_stride
            print(f"{layer.name:<20} {pool_size:<15} {pool_stride:<10} {receptive_field}")
    return receptive_field
if args.phase == "train":

    # data load and confirm
    data_loader = DataLoader(xml_folder, csv_file, phase=args.phase, types=types, random_state=RANDOM_SEED)
    for idx in range(5) :
        (train_data, train_labels), (val_data, val_labels) = data_loader.get_fold_data(idx)
        print(train_data)
        print(train_data.shape, Counter(train_labels))
        print(val_data.shape, Counter(val_labels))
        
        # for ecgs, label in zip((train_data, train_labels)) :
        #     print(ecgs.shape)
        #     show_batch(ecgs, label)
        #     break

        # # model train and validation
        model_inc = CNNRegressor(activation='linear', loss='mean_squared_error')#ResNetClassifier()#InceptionTimeClassifier()#model[model_name]
        final_model = model_inc.build_model(input_shape = (250,12))
        
        print(final_model.summary())

        # 모델의 수용 영역 계산
        receptive_field = calculate_receptive_field(final_model.layers)
        print(f'Receptive field: {receptive_field}')
                
        
        trainer = Trainer(model=final_model, epochs=num_epochs, types=types, idx=idx, save_path=save_path)
        trainer.train(train_data, train_labels, val_data, val_labels)
        break

#elif args.phase == "int test" :
#    data_loader = DataLoader(xml_folder, csv_file, phase=args.phase, types=types, random_state=RANDOM_SEED)
#    (int_test_data, int_test_labels) = data_loader.get_test_data()
#    results = []
#    for idx in range(5) :                 
        # # model train and validation
#        model_inc = CNNRegressor()#ResNetClassifier()#InceptionTimeClassifier()#model[model_name]
#        final_model = model_inc.build_model(input_shape = (250,12), n_classes=1)
        
#        Tester = Trainer(model=final_model, epochs=num_epochs, types=types, idx=idx, save_path=save_path)

#        Tester.test(model=final_model, test_dl=int_test_data, METRICS=[MeanSquaredError(), MeanAbsoluteError()])

        #trainer.test(int_test_data, int_test_labels, model[model_name], num_epochs, args.phase, idx, save_path)

#elif args.phase == "ext test" :
#    data_loader = DataLoader(xml_folder, csv_file, phase=args.phase, types=types, random_state=RANDOM_SEED)
#    (ext_test_data, ext_test_labels) = data_loader.get_test_data()
#    for idx in range(5) :
        
        # # model train and validation
#        model_inc = CNNRegressor()#ResNetClassifier()#InceptionTimeClassifier()#model[model_name]
#        final_model = model_inc.build_model(input_shape = (250,12), n_classes=1)
        
#        trainer = Trainer(model=final_model, epochs=num_epochs, types=types, idx=idx, save_path=save_path)

#       trainer.test(ext_test_data, ext_test_labels, model[model_name], num_epochs, args.phase, idx, save_path)
else : 
    raise ValueError("You entered the phase incorrectly")
