from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import applications
from tensorflow.keras.models import load_model

def model(name, **kwargs):
    if name == 'binario':
        return cnn_binario(kwargs['shape'])
    if name == 'lenet5':
        return lenet5(kwargs['shape'], kwargs['num_classes'])
    if name == 'lenet':
        return lenet5_custom(kwargs['shape'], kwargs['num_classes'], lr=kwargs['lr'])
    if name == 'resnet':
        return resnet(kwargs['shape'], kwargs['num_classes'])
    if name == 'vgg16':
        return vgg16(kwargs['shape'], kwargs['num_classes'])
    if name == 'vgg19':
        return vgg19(kwargs['shape'], kwargs['num_classes'])
    if name == 'load':
        model = load_model(kwargs['load_path'])
        print('Loaded model from {0}'.format(str(kwargs['load_path'])) )
        return model

def cnn_binario(shape, num_classes=1):
    output_cells = num_classes if num_classes > 2 else 1
    loss = 'categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'

    # Initialising the CNN
    classifier = Sequential()

    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape=shape, activation='relu'))

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    #classifier.add(BatchNormalization())

    # Adding a second convolutional layer
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    #classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a third convolutional layer
    classifier.add(Conv2D(128, (3, 3), activation='relu'))
    #classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a fourth convolutional layer
    classifier.add(Conv2D(128, (3, 3), activation='relu'))
    #classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Step 3 - Flattening
    classifier.add(Flatten())

    #kernel_regularizer = regularizers.l2(0.0001)
    # Step 4 - Full connection
    classifier.add(Dense(units=64, activation='relu'))
    #classifier.add(Dropout(0.3))
    classifier.add(Dense(units=32, activation='relu'))
    #classifier.add(Dropout(0.3))
    classifier.add(Dense(units=output_cells, activation='sigmoid'))

    # Compiling the CNN
    classifier.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    return classifier


def lenet5_custom(shape, num_classes=3, lr=0.001):
    output_cells = num_classes if num_classes > 2 else 1
    loss = 'categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    model_lenet5 = Sequential([
        Conv2D(4, (5, 5), activation='relu', input_shape=shape, data_format="channels_last"),
        BatchNormalization(),
        AveragePooling2D(),
        Conv2D(9, (3, 3), activation='relu'),
        BatchNormalization(),
        AveragePooling2D(),
        Flatten(),
        Dense(80, activation='relu'),
        Dropout(0.5),
        Dense(24, activation='relu'),
        Dense(output_cells, activation='softmax' if num_classes > 2 else 'sigmoid')]
    )

    model_lenet5.compile(loss=loss,
                         optimizer=Adam(lr=lr),
                         #               regularization='L2',
                         metrics=['accuracy'])
    print('LeNet5 model loaded')

    return model_lenet5


def lenet5(shape, num_classes=3):
    output_cells = num_classes if num_classes > 2 else 1
    loss = 'categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    model_lenet5 = Sequential([
        Conv2D(6, (5, 5), activation='relu', input_shape=shape, data_format="channels_last"),
        BatchNormalization(),
        AveragePooling2D(),
        Conv2D(16, (3, 3), activation='relu'),
        BatchNormalization(),
        AveragePooling2D(),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(output_cells, activation='softmax' if num_classes > 2 else 'sigmoid')]
    )

    model_lenet5.compile(loss=loss,
                         optimizer=SGD(),
                         #               regularization='L2',
                         metrics=['accuracy'])
    print('LeNet5 model loaded')

    return model_lenet5


def resnet(shape, num_classes=1):
    output_cells = num_classes if num_classes > 2 else 1
    activation = 'softmax' if num_classes > 2 else 'sigmoid'
    loss = 'categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    base_model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=shape )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(output_cells, activation= activation)(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(optimizer = 'adam', loss=loss, metrics=['accuracy'])
    print('ResNet50 model loaded')
    return model
    
def vgg19(shape, num_classes=1):
    output_cells = num_classes if num_classes > 2 else 1
    activation = 'softmax' if num_classes > 2 else 'sigmoid'
    loss = 'categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    base_model = applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=shape )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(output_cells, activation= activation)(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(optimizer = 'adam', loss=loss, metrics=['accuracy'])
    print('VGG19 model loaded')
    return model
    
def vgg16(shape, num_classes=1):
    output_cells = num_classes if num_classes > 2 else 1
    activation = 'softmax' if num_classes > 2 else 'sigmoid'
    loss = 'categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    base_model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=shape )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(output_cells, activation= activation)(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    model.compile(optimizer = 'adam', loss=loss, metrics=['accuracy'])
    print('VGG16 model loaded')
    return model