from google.cloud import storage
from google.cloud.storage.blob import Blob
import pandas as pd
import numpy as np
from io import StringIO
from keras.utils import to_categorical
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import save_img
from tqdm import tqdm
import os
import gc


# Devuelve la lista de ficheros de un bucket
def file_list(bucket, date, time, window, ahead, offset, std):
    path = 'data/' + date + '_' + time + '_W' + window + '_A' + ahead + '_O' + offset + '_S' + std + '/images/part'
    l = bucket.list_blobs(prefix=path)
    return [x for x in l if x.size > 0]


# Descarga una lista de ficheros de GS a una carpeta local
def download_all(files, target_folder='', format=None):
    for i in tqdm(range(0, len(files))):
        filename = target_folder + 'file_' + str(i)
        if format is None:
            files[i].download_to_filename(filename + '.csv')
        if format == 'feather':
            df = toDF(files[i])
            df.to_feather(filename + '.feather')


# Crea un dataframe a partir de un fichero o de un Blob
def toDF(file):
    if isinstance(file, Blob):
        s = str(file.download_as_string(), 'utf-8')
        s = StringIO(s)
        return pd.read_csv(s)
    extension = file.split('.')[1]
    if extension == 'csv':
        return pd.read_csv(file)
    if extension == 'feather':
        return pd.read_feather(file)


def label_binario_up(x):
    return 1 if x == 1 else 0

def label_binario_down(x):
    return 1 if x == -1 else 0


# Crea un dataframe a partir de todos los ficheros
def toDF_all(files, reb=True, f=None):
    dfs = []
    for i in range(0, len(files)):
        df = toDF(files[i])
        if not f is None:
            df['LABEL'] = df['LABEL'].apply(f)
        if reb:
            df = rebalance(df)
        dfs.append(df)
    df_final = pd.concat(dfs)
    del dfs
    gc.collect()
    return df_final


# Crea un unico fichero a partir de todos los CSV de una lista
def merge_all_CSV(filelist, reb=False):
    filename = 'df_all.csv'
    for i in tqdm(range(0, len(filelist))):
        df = pd.read_csv(filelist[i])
        if reb:
            df = rebalance(df)
        if i == 0:
            df.to_csv(filename, mode='w', header=True, index=None)
        else:
            df.to_csv(filename, mode='a', header=False, index=None)


# Crea un unico fichero HDF a partir de todos los CSV de una lista
def merge_all_HDF(filelist, reb=False):
    filename = 'df_all.h5'
    for i in tqdm(range(0, len(filelist))):
        df = pd.read_csv(filelist[i])
        if reb:
            df = rebalance(df)
        if i == 0:
            df.to_hdf(filename, 'df', mode='w', header=True, index=None)
        else:
            df.to_hdf(filename, 'df', mode='a', header=False, index=None)



# Un dataframe lo convierte en X,y listo para pasarselo a la CNN
def split(df, categories=3, augment_data=True):
    X = df.iloc[:, 21:]
    pixels = int(np.sqrt(X.shape[1]/3))
    print(len(df))
    X = np.reshape(X.values, (len(df), pixels, pixels, 3))
    y = df['LABEL'].values
    if categories > 2:
        y = to_categorical(y, categories)
    if augment_data:
        X = augment(X, pixels=pixels)
        y = np.concatenate((y,y))
    return X, y


def augment(X, pixels=144):
    X2 = np.array([[255 - a for a in x.ravel()] for x in X]).reshape(X.shape[0], pixels, pixels, 3)
    return np.concatenate((X,X2))


# Devuelve un dataframe que tiene el mismo numero de elementos de cada etiqueta
def rebalance(df, shuffle=True):
    g = df.groupby('LABEL')
    g = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)
    if shuffle:
        g = g.sample(frac=1).reset_index(drop=True)
    return g


def readfile(blob):
    df = toDF(blob)
    df['LABEL'] = df['LABEL'].apply(label_binario_up)
    df = rebalance(df)
    X, y = split(df)
    X = X / 255
    return X, y


def byHDF(dfs, remove=True):
    store = pd.HDFStore('df_all.h5')
    for df in dfs:
        store.append('df', df)
    # del dfs
    df = store.select('df')
    store.close()
    if remove:
        os.remove('df_all.h5')
    return df


def byCSV(dfs, remove=True):
    md, hd = 'w', True
    for df in dfs:
        df.to_csv('df_all.csv', mode=md, header=hd, index=None)
        md, hd = 'a', False
    # del dfs
    df_all = pd.read_csv('df_all.csv', index_col=None)
    if remove:
        os.remove('df_all.csv')
    return df_all


def generate_png(data, path='test_image_cnn.png'):
    save_img(path, array_to_img(data))


# Funcion generadora para entrenar en batches desde GS
def generator(filenames, batch_size):
    print('Reading new file', str(filenames[0]))
    X, y = readfile(filenames[0])
    fileindex = 1
    i = 0
    while True:
        if batch_size > len(X):
            print('Reading new file', str(fileindex))
            if fileindex > len(filenames) - 1:
                print('End of data')
                print('Set', str(i))
                i = i + 1
                yield X, y
            X_aux, y_aux = readfile(filenames[fileindex])
            X = np.concatenate((X, X_aux), axis=0)
            y = np.concatenate((y, y_aux), axis=0)
            fileindex = fileindex + 1

        batch_x = X[:batch_size]
        batch_y = y[:batch_size]

        X = X[batch_size:]
        y = y[batch_size:]
        print('Set', str(i))
        i = i + 1
        yield batch_x, batch_y