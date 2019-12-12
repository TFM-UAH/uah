import argparse
import pandas as pd
import tensorflow as tf
import ast
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler

print('Configurando TF')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


#Get the data and splits in input X and output Y, by spliting in `n` past days as input X
#and `m` coming days as Y.
def processData(data, look_back, forward_days,jump=1, scale=True, label=False):
    serie = data['ADJ_CLOSE'].values.reshape(data.shape[0],1)
    X, Y, labels = [],[],[]
    for i in range(0,len(serie) -look_back -forward_days +1, jump):
        d = serie[i:i+look_back+forward_days]
        if scale == True:
            d = MinMaxScaler().fit_transform( d )
        X.append(d[:look_back])
        Y.append(d[look_back:look_back+forward_days])
        labels.append(data.iloc[i+look_back-1]['LABEL'])
    return X,Y,labels


def main():
    parser = argparse.ArgumentParser()

    # Read Arguments
    parser.add_argument("--model", "-m", help="Model")
    parser.add_argument("--save", "-s", help="Saving suffix path")
    parser.add_argument("--input", "-i", help="Input Path")
    parser.add_argument("--output", "-o", help="Output Path")
    parser.add_argument("--classes", "-c", help="Num classes")
    parser.add_argument("--batch", "-b", help="Batch size")
    parser.add_argument("--epochs", "-e", help="Number of epochs to train de model")
    parser.add_argument("--verbose", "-v", help="Verbosity")
    parser.add_argument("--load", "-l", help="Load model path")
    parser.add_argument("--weights", "-w", help="Load model weights")
    parser.add_argument("--lr", "-n", help="Learning Rate")
    parser.add_argument("--class_weight", "-g", help="Class weights")
    parser.add_argument("--look_back", "-p", help="Look Back Periods")
    parser.add_argument("--forward_days", "-f", help="Forward Days")
    parser.add_argument("--offset", "-d", help="Offset")

    # read arguments from the command line
    args = parser.parse_args()

    num_classes = int(args.classes) if args.classes else 2
    save_suffix = args.save if args.save else ''
    input_path = args.input if args.input else ''
    output_path = args.output if args.output else ''
    batch_size = int(args.batch) if args.batch else 32
    epochs = int(args.epochs) if args.epochs else 10
    verbose = int(args.verbose) if args.verbose else 1
    load_path = args.load if args.load else None
    load_weights = args.weights if args.weights else None
    lr = float(args.lr) if args.lr else 0.001
    class_weight = ast.literal_eval(args.class_weight) if args.class_weight else None
    look_back = int(args.look_back) if args.look_back else 100
    forward_days = int(args.forward_days) if args.forward_days else 30
    num_periods = int(args.offset) if args.offset else 20


    # Print summary of params
    print("Model {0}".format(args.model))
    print("Save {0}".format(save_suffix))
    print("Input path {0}".format(input_path))
    print("Output path {0}".format(output_path))
    print("Num classes {0}".format(str(num_classes)))
    print("Batch size {0}".format(str(batch_size)))
    print("Number of Epochs {0}".format(str(epochs)))
    print("Learning Rate {0}".format(str(lr)))
    print("Class Weights {0}".format(str(class_weight)))
    print("Look Back {0}".format(str(look_back)))
    print("Forward Days {0}".format(str(forward_days)))
    print("Offset {0}".format(str(num_periods)))


    df_train = pd.read_feather(input_path + "train/train.feather")

    df_train = df_train[['SYMBOL','ADJ_CLOSE','LABEL']]
    X = []
    y = []
    actual_labels = []
    for s, group in df_train.groupby('SYMBOL'):
        #array = group['ADJ_CLOSE'].values.reshape(group.shape[0], 1)
        X_t, y_t, labels = processData(group, look_back, forward_days, jump=num_periods)
        X = X + X_t
        y = y + y_t
        actual_labels = actual_labels + labels

    df = pd.DataFrame({'X': X, 'Y': y, 'LABEL': actual_labels})
    g = df.groupby('LABEL')
    g = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)
    df = g.sample(frac=1).reset_index(drop=True)

    X = df['X'].values
    Y = df['Y'].values

    X = np.array([a for a in X])
    Y = np.array([a for a in Y])

    print(len(Y))
    y = np.array([list(a.ravel()) for a in Y])
    #appl = df_train[df_train['SYMBOL'] == 'AAPL']
    #array = appl['ADJ_CLOSE'].values.reshape(appl.shape[0], 1)

    # Preprocess data
    #X, y = processData(array, look_back, forward_days, jump=num_periods)
    #y = np.array([list(a.ravel()) for a in y])

    from sklearn.model_selection import train_test_split
    print(len(X))
    print(len(y))
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.10, random_state=42)

    print("Training on {0} samples".format(str(len(X_train))))
    print("Validating on {0} samples".format(str(len(X_validate))))

    # Generate Model
    # Training the LSTM
    NUM_NEURONS_FirstLayer = 50
    NUM_NEURONS_SecondLayer = 30

    # Build the model
    model = Sequential()
    model.add(LSTM(NUM_NEURONS_FirstLayer, input_shape=(look_back, 1), return_sequences=True))
    model.add(LSTM(NUM_NEURONS_SecondLayer, input_shape=(NUM_NEURONS_FirstLayer, 1)))
    model.add(Dense(forward_days))
    model.compile(loss='mean_squared_error', optimizer='adam')


    # Train Model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(output_path + 'best_' + save_suffix + '.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    callbacks = [es, mc]
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_validate, y_validate), shuffle=True,
                        batch_size=batch_size, verbose=verbose, callbacks=callbacks)


    # Save Model
    path = output_path + 'model_' + save_suffix + '.h5'
    model.save('model_' + save_suffix + '.h5')
    print("Model saved to {0}".format(str(path)))

    # Weights
    path = output_path + 'weights_' + save_suffix + '.h5'
    model.save_weights('weights_' + save_suffix + '.h5')
    print("Weights saved to {0}".format(str(path)))

    # History
    path = output_path + 'history_' + save_suffix + '.csv'
    pd.DataFrame(history.history).to_csv(path)
    print("History saved to {0}".format(str(path)))


if __name__ == '__main__':
    main()