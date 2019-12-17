from google.cloud import storage
from generatorCNN import GeneratorCNN
import glob
from sklearn.model_selection import train_test_split
import functions as func
from tqdm import tqdm


def local(path='', num_train=None, num_validation=None, num_classes=2, augment=True):
    trainpath = path + 'train/*'
    validationpath = path + 'validation/*'

    print("Reading training from {0}".format(trainpath))
    files = glob.glob(trainpath)
    files = files if num_train is None else files[:num_train]
    df = func.toDF_all(files, reb=False)
    X_train, y_train = func.split(df, categories=num_classes, augment_data=augment)
    X_train = X_train / 255
    print('X train shape', X_train.shape)
    print('y train shape', y_train.shape)

    print("Reading validation from {0}".format(validationpath))
    files = glob.glob(validationpath)
    files = files if num_validation is None else files[:num_validation]
    df = func.toDF_all(files, reb=False)
    X_validation, y_validation = func.split(df, categories=num_classes, augment_data=augment)
    X_validation = X_validation / 255
    print('X validation shape', X_validation.shape)
    print('y validation shape', y_validation.shape)

    return X_train, y_train, X_validation, y_validation

def generators(files_train = None, files_validation = None, path = '', suffix='simple', batch_size=32, pixels=144, num_classes=2, augment=True):
    folder_train = path + 'train/*'
    folder_test = path + 'validation/*'
    gen_train = GeneratorCNN(filenames=files_train, mode='feather', folder=folder_train, categories=num_classes, batch_size=batch_size,
                             pixels=pixels)
    print('Training Generator loaded')
    gen_validation = GeneratorCNN(filenames = files_validation, mode='feather', folder=folder_test, categories=num_classes,
                                  batch_size=batch_size, pixels=pixels)
    print('Validation Generator loaded')
    return gen_train, gen_validation

def mixed(files_train = None, files_validation = None, path = '', suffix='simple', batch_size=32, pixels=144, num_classes=2, augment=False):
    gen_train = GeneratorCNN(filenames=files_train, mode='feather', categories=num_classes, batch_size=batch_size, pixels=pixels)
    print('Training Generator loaded')

    df = func.toDF_all(files_validation, reb=False)
    X_validation, y_validation = func.split(df, categories=num_classes, augment_data=augment)
    X_validation = X_validation / 255
    print('X validation shape', X_validation.shape)
    print('y validation shape', y_validation.shape)
    print('Validation Generator loaded')
    return gen_train, X_validation, y_validation

def copy_feather(files, suffix='simple', start=-1, grouping=5, reb=True):
    start = start if start >= 0 else grouping
    j = start - grouping
    for i in tqdm(range(start, len(files), grouping)):
        df = func.toDF_all(files[j:i], reb=False)
        j = i
        if reb:
            df['LABEL'] = df['LABEL'].apply(lambda x: 1 if x == 1 else 0)
            df = func.rebalance(df)
        if len(df) >= 10:
            df_train, df_test = train_test_split(df, shuffle=True, stratify=df[['LABEL']], test_size=0.2)
            df_train.reset_index(drop=True).to_feather('train_' + suffix + '/file_' + str(i) + ('.feather'))
            df_validation, df_test = train_test_split(df_test, shuffle=True, stratify=df_test[['LABEL']], test_size=0.5)
            df_validation.reset_index(drop=True).to_feather('validation_' + suffix + '/file_' + str(i) + ('.feather'))
            df_test.reset_index(drop=True).to_feather('test_' + suffix + '/file_' + str(i) + ('.feather'))

def gs(bucket_name, credentials_path=None):
    if credentials_path is None:
        storage_client = storage.Client()
    else:
        storage_client = storage.Client.from_service_account_json(json_credentials_path=credentials_path)

    # Creates the new bucket
    bucket = storage_client.get_bucket(bucket_name)
    print('Bucket {} read.'.format(bucket.name))
    return bucket