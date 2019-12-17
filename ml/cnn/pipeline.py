import argparse
import datasets as datasets
import models as models
import pandas as pd
import glob
import tensorflow as tf
import ast
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

print('Configurando TF')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


def main():
    parser = argparse.ArgumentParser()
    
    # Read Arguments
    parser.add_argument("--model", "-m", help="Model")
    parser.add_argument("--dataset", "-d", help="Dataset")
    parser.add_argument("--save", "-s", help="Saving suffix path")
    parser.add_argument("--input", "-i", help="Input Path")
    parser.add_argument("--classes", "-c", help="Num classes")
    parser.add_argument("--batch", "-b", help="Batch size")
    parser.add_argument("--pixels", "-p", help="Pixels")
    parser.add_argument("--train", "-t", help="Train files")
    parser.add_argument("--val", "-u", help="Validation files")
    parser.add_argument("--epochs", "-e", help="Number of epochs to train de model")
    parser.add_argument("--verbose", "-v", help="Verbosity")
    parser.add_argument("--load", "-l", help="Load model path")
    parser.add_argument("--weights", "-w", help="Load model weights")
    parser.add_argument("--lr", "-n", help="Learning Rate")
    parser.add_argument("--augment", "-a", help="Augmentation")
    parser.add_argument("--class_weight", "-g", help="Class weights")


    # read arguments from the command line
    args = parser.parse_args()

    dataset_mode = args.dataset if args.dataset else 'generator'
    num_classes = int(args.classes) if args.classes else 1
    save_suffix = args.save if args.save else ''
    input_path = args.input if args.input else ''
    batch_size = int(args.batch) if args.batch else 32
    pixels = int(args.pixels) if args.pixels else 144
    num_train = int(args.train) if args.train else None
    num_validation = int(args.val) if args.val else None
    epochs = int(args.epochs) if args.epochs else 5
    verbose = int(args.verbose) if args.verbose else 1
    load_path = args.load if args.load else None
    load_weights = args.weights if args.weights else None
    lr = float(args.lr) if args.lr else 0.001
    augment = True if args.augment else False
    class_weight = ast.literal_eval(args.class_weight) if args.class_weight else None

    #Print summary of params
    print("Dataset {0}".format(dataset_mode))
    print("Model {0}".format(args.model))
    print("Save {0}".format(save_suffix))
    print("Input path {0}".format(input_path))
    print("Num classes {0}".format(str(num_classes)))
    print("Batch size {0}".format(str(batch_size)))
    print("Pixels {0}".format(str(pixels)))
    print("Number of Train files {0}".format(str(num_train)))
    print("Number of Validation files {0}".format(str(num_validation)))
    print("Number of Epochs {0}".format(str(epochs)))
    print("Learning Rate {0}".format(str(lr)))
    print("Class Weights {0}".format(str(class_weight)))

    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)
    mc = ModelCheckpoint('best_' + save_suffix + '.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    callbacks = [es,mc]


    # Get Dataset
    if dataset_mode == 'local':
        X_train, y_train, X_validation, y_validation = datasets.local(input_path, num_train=num_train, num_validation=num_validation, augment=augment)
        shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    if dataset_mode == 'generator':
        files_train = glob.glob(input_path+'train/*')
        files_validation = glob.glob(input_path+'validation/*')
        gen_train, gen_validation = datasets.generators(files_train=files_train, files_validation=files_validation,
                                                        num_classes=num_classes, batch_size=batch_size)
        shape = (gen_train.get_shape()[1], gen_train.get_shape()[2], gen_train.get_shape()[3])

    if dataset_mode == 'mixed':
        files_train = glob.glob(input_path+'train/*')
        files_validation = glob.glob(input_path+'validation/*')
        gen_train, X_validation, y_validation = datasets.mixed(files_train=files_train, files_validation=files_validation,
                                                               num_classes=num_classes, batch_size=batch_size)
        shape = (gen_train.get_shape()[1], gen_train.get_shape()[2], gen_train.get_shape()[3])

    # Load/Create Model
    model = models.model(args.model, shape=shape, num_classes=num_classes, load_path=load_path, lr=lr)
    if not load_weights is None:
        model.load_weights(load_weights)


    # Train Model
    if dataset_mode == 'local':
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_validation,y_validation), class_weight=class_weight, callbacks=callbacks)

    if dataset_mode == 'generator':
        history = model.fit_generator(generator=gen_train, validation_data=gen_validation, epochs=epochs, verbose=verbose, use_multiprocessing=False, class_weight=class_weight, callbacks=callbacks)

    if dataset_mode == 'mixed':
        history = model.fit_generator(generator=gen_train, validation_data=(X_validation,y_validation), epochs=epochs, verbose=verbose, use_multiprocessing=False, class_weight=class_weight, callbacks=callbacks)


    # Save Model
    path = 'model_' + save_suffix + '.h5'
    model.save('model_' + save_suffix + '.h5')
    print("Model saved to {0}".format(str(path)))

    # Weights
    path = 'weights_' + save_suffix + '.h5'
    model.save_weights('weights_' + save_suffix + '.h5')
    print("Weights saved to {0}".format(str(path)))

    # History
    path = 'history_' + save_suffix + '.csv'
    pd.DataFrame(history.history).to_csv(path)
    print("History saved to {0}".format(str(path)))


if __name__ == '__main__':
    main()