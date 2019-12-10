from google.cloud import storage
from tensorflow.keras.utils import Sequence
from io import StringIO
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
import glob
import math


class Reader():
    def __init__(self, mode):
        self.mode = mode

    # a String or a Blob, depending on the mode
    def read(self, file):
        if self.mode == 'gs':
            s = str(file.download_as_string(), 'utf-8')
            s = StringIO(s)
            return pd.read_csv(s)

        if self.mode == 'feather':
            return pd.read_feather(file)


class GeneratorCNN(Sequence):
    def __init__(self, filenames=None, mode=None, folder=None, bucket_name=None, credentials_path=None, num_files=None, categories=2, batch_size=32, reb=False, pixels=72):

        if filenames is None:
            if (mode != 'gs') & (mode != 'feather'):
                raise Exception('Mode must be gs or feather')

            if mode == 'gs':
                # Read Client
                storage_client = storage.Client() if credentials_path is None else storage.Client.from_service_account_json \
                    (json_credentials_path=credentials_path)
                # Creates the new bucket
                bucket = storage_client.get_bucket(bucket_name)
                filenames = self.__file_list__(bucket, '20191021', '221842', '30', '10', '5', '2.0')
                self.filenames = filenames[:(len(filenames) if num_files is None else num_files)]

            if mode =='feather':
                filenames = glob.glob(folder + '/*.feather')
                self.filenames = filenames[:(len(filenames) if num_files is None else num_files)]
        else:
            self.filenames = filenames

        self.reader = Reader(mode)

        self.pixels = pixels
        self.reb = reb
        self.categories = categories
        self.batch_size = batch_size
        self.len = 0
        
        self.indices = {}
        idx = 0
        carry = 0
        for f in self.filenames:
            print('Reading {0}'.format(f))
            df = self.reader.read(f)
            size = len(df)
            #for i in range(carry, size, self.batch_size):
                #if i + self.batch_size > size:
                    #carry = i + self.batch_size - size
                #self.indices[idx] = (f,i)
                #idx = idx + 1
            self.len = self.len + size
        print("Read {0} samples".format(str(self.len)))

        self.on_epoch_end()

    def __file_list__(self, bucket, date, time, window, ahead, offset, std):
        path = 'data/' + date + '_' + time + '_W' + window + '_A' + ahead + '_O' + offset + '_S' + std + '/images/part'
        l = bucket.list_blobs(prefix=path)
        return [x for x in l if x.size > 0]

    def __readfile__(self ,blob):
        self.current_file = blob
        df = self.reader.read(blob)
        if self.reb == True:
            df['LABEL'] = df['LABEL'].apply(lambda x: 1 if x == 1 else 0)
            df = self.__rebalance__(df)
        X ,y = self.__split__(df)
        X = X / 255
        return X, y
    
    def __rebalance__(self, df, shuffle=True):
        g = df.groupby('LABEL')
        g = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)
        if shuffle:
            g = g.sample(frac=1).reset_index(drop=True)
        return g

    # Un dataframe lo convierte en X,y listo para pasarselo a la CNN
    def __split__(self, df, layers=3):
        X = df.iloc[: ,21:]
        X = np.reshape(X.values ,(len(df), self.pixels , self.pixels ,layers))
        y = df['LABEL'].values
        if self.categories > 2:
            y = to_categorical(y, self.categories)
        return X ,y

    def __len__(self) :
        return math.ceil(self.len / self.batch_size)

    def __getitem_old__(self, idx_ext):

        #print(idx)
        f, row = self.indices[self.idx]
        if f != self.current_file:
            self.X, self.y = self.__readfile__(f)
            
        #print(self.name, 'Indice:',str(idx),'File:', f, 'Row:',str(row))
        
        if row + self.batch_size > len(self.X):
            if self.idx >= len(self.indices) -1:
                return self.X[row:], self.y[row:]

            
            X_aux, y_aux = self.__readfile__(self.indices[self.idx+1][0])
            batch_x = np.concatenate((self.X[row:], X_aux[:row + self.batch_size - len(self.X)]), axis=0)
            batch_y = np.concatenate((self.y[row:], y_aux[:row + self.batch_size - len(self.y)]), axis=0)
            #t = 'idx ' + str(idx) + ' ['+ str(row) + '] ' + ' / idx ' + str(idx+1)+ ' ['+ str(row) + '] '
            
            self.X = X_aux
            self.y = y_aux
        else:         
            batch_x = self.X[row : row + self.batch_size]
            batch_y = self.y[row : row + self.batch_size]

        self.idx = self.idx + 1

        return batch_x, batch_y

    def __getitem__(self, idx_ext):

        if self.batch_size > len(self.X):
            # Last batch
            if self.idx >= len(self.filenames) - 1:
                batch_x = self.X
                batch_y = self.y
                self.on_epoch_end()
                return batch_x, batch_y

            # Read new file
            X_aux, y_aux = self.__readfile__(self.filenames[self.idx + 1])
            self.idx = self.idx + 1
            self.X = np.concatenate((self.X, X_aux), axis=0)
            self.y = np.concatenate((self.y, y_aux), axis=0)

        # Extract batch
        batch_x = self.X[: self.batch_size]
        batch_y = self.y[: self.batch_size]

        self.X = self.X[self.batch_size:]
        self.y = self.y[self.batch_size:]

        return batch_x, batch_y

    def on_epoch_end(self):
        self.idx = 0
        self.X, self.y = self.__readfile__(self.filenames[0])
    
    def get_shape(self):
        return self.X.shape