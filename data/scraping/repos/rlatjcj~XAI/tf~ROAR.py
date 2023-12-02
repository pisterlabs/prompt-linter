import os
import sys
import tqdm
import json
import random
import argparse
import threading

import keras
import tensorflow as tf
from keras.preprocessing.image import random_rotation

from model import Mymodel
from coherence import *

np.random.seed(42)

coherence_dict = {'saliency': saliency,
                  'smoothgrad': smoothgrad,
                  'smoothgrad2': smoothgrad_square,
                  'vargrad': vargrad,
                  'integratedgrad': integratedgrad}

class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def Generator(x, y, c,
              dataset_idx=None,
              ratio=0.,
              mode='train',
              input_shape=32,
              batch_size=64,
              classes=10,
              seed=42,
              shuffle=True,
              **kwargs):

    random.seed(seed)
    datalist = np.arange(x.shape[0])
    batch = 0
    X = np.zeros((batch_size, input_shape, input_shape, 3))
    Y = np.zeros((batch_size, classes))

    while True:
        if shuffle:
            random.shuffle(datalist)

        for data in datalist:
            img = x[data].astype(np.float32)
            img /= 255
            img[np.where(c[data] <= ratio)] = img.mean()

            if mode == 'train':
                if np.random.random() > .5:
                    # vertical flip
                    img = img[::-1]
                    
                if np.random.random() > .5:
                    # horizontal flip
                    img = img[:,::-1]
                    
                img = random_rotation(img, 10, row_axis=0, col_axis=1, channel_axis=2)
                
            X[batch] = img
            Y[batch, y[data]] += 1
            
            batch += 1
            if batch >= batch_size:
                yield X, Y
                batch = 0
                X = np.zeros((batch_size, input_shape, input_shape, 3))
                Y = np.zeros((batch_size, classes))

def make_saliency(saliency_func, coherence, callback_name, saliency_model, x_train, x_test):
    train_saliency = np.zeros_like(x_train, dtype=np.uint8)
    test_saliency = np.zeros_like(x_test, dtype=np.uint8)
    for i in tqdm.trange(len(x_train)):
        img = x_train[i].astype(np.float32)
        img /= 255.
        pred = saliency_model.predict_on_batch(img[np.newaxis,...])
        if coherence == 'saliency':
            result = saliency_func[np.argmax(pred[0])]([img[np.newaxis,...]])[0][0]
        else:
            result = coherence_dict[coherence](saliency_func[np.argmax(pred[0])], img[np.newaxis,...])[0]

        result_idx = np.argsort(-result.flatten()).tolist()
        ex_idx = np.zeros((train_saliency.shape[1]*train_saliency.shape[2]*train_saliency.shape[3]), dtype=np.uint8)
        for j in range(10):
            ex_idx[result_idx[int(len(result_idx)*(j/10)):int(len(result_idx)*((j+1)/10))]] = j+1
        train_saliency[i] = ex_idx.reshape((train_saliency.shape[1], train_saliency.shape[2], train_saliency.shape[3]))

    for i in tqdm.trange(len(x_test)):
        img = x_test[i].astype(np.float32)
        img /= 255.
        pred = saliency_model.predict_on_batch(img[np.newaxis,...])
        if coherence == 'saliency':
            result = saliency_func[np.argmax(pred[0])]([img[np.newaxis,...]])[0][0]
        else:
            result = coherence_dict[coherence](saliency_func[np.argmax(pred[0])], img[np.newaxis,...])[0]

        result_idx = np.argsort(-result.flatten()).tolist()
        ex_idx = np.zeros((test_saliency.shape[1]*test_saliency.shape[2]*test_saliency.shape[3]), dtype=np.uint8)
        for j in range(10):
            ex_idx[result_idx[int(len(result_idx)*(j/10)):int(len(result_idx)*((j+1)/10))]] = j+1
        test_saliency[i] = ex_idx.reshape((test_saliency.shape[1], test_saliency.shape[2], test_saliency.shape[3]))

    np.save('./saliency/{}/{}_train.npy'.format(callback_name, coherence), train_saliency)
    np.save('./saliency/{}/{}_test.npy'.format(callback_name, coherence), test_saliency)
    return train_saliency, test_saliency
            
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

def check_args(args):
    assert args.dataset, 'Dataset will use must be selected.'

    return args

def get_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str,   default=None)
    parser.add_argument("--seblock",    action='store_true')
    parser.add_argument("--cbamblock",  action='store_true')

    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--steps",      type=int,   default=0)
    parser.add_argument("--lr",         type=float, default=.0001)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--checkpoint", type=str,   default=None)
    parser.add_argument("--callbacks",  action='store_true')
    parser.add_argument("--summary",    action='store_true')

    parser.add_argument("--coherence",  type=str,   default='saliency', choices=['saliency', 
                                                                                 'smoothgrad', 
                                                                                 'smoothgrad2', 
                                                                                 'vargrad', 
                                                                                 'integratedgrad'])

    return check_args(parser.parse_args(args))

def main(args=None):
    """S. Hooker et al., A Behcnmark for Interpretability Methods in Deep Neural Networks, NeurIPS 2019.
    """
    if args is None:
        args = sys.argv[1:]
    
    args = get_arguments(args)
    get_session()

    assert args.checkpoint

    if args.dataset == 'cifar10':
        from keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        label_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        classes = 10

    else:
        raise ValueError()

    train_model = Mymodel(
        classes=classes, 
        _isseblock=args.seblock,
        _iscbamblock=args.cbamblock
    )

    if args.summary:
        train_model.summary()
        return

    callback_name = ''
    if args.seblock:
        callback_name += 'seblock_'
    elif args.cbamblock:
        callback_name += 'cbamblock_'
    else:
        callback_name += 'no_'

    if not os.path.isfile('./init_{}.h5'.format(callback_name+args.coherence)):
        train_model.save_weights('./init_{}.h5'.format(callback_name+args.coherence))
    else:
        train_model.load_weights('./init_{}.h5'.format(callback_name+args.coherence))

    init_weights = train_model.get_weights()

    if not os.path.isfile('./saliency/{}/{}_train.npy'.format(callback_name, args.coherence)):
        saliency_model = keras.models.clone_model(train_model)
        saliency_model.load_weights(args.checkpoint)
        saliency_model.trainable = False
        saliency_func = [saliency(c, saliency_model) for c in range(classes)]
        train_saliency, test_saliency = make_saliency(saliency_func, args.coherence, callback_name, saliency_model, x_train, x_test)
        del saliency_model
    else:
        train_saliency = np.load('./saliency/{}/{}_train.npy'.format(callback_name, args.coherence))
        test_saliency = np.load('./saliency/{}/{}_test.npy'.format(callback_name, args.coherence))

    # find the latest model weights
    ckpt = None
    flag = False
    for prev_ratio in range(1, 11):
        try:
            ckpt_list = sorted(os.listdir('./checkpoint/ROAR/{}/{}'.format(callback_name+args.coherence, int(prev_ratio*10))),
                               key=lambda x: int(x.split('/')[-1].split('_')[0]))
            print(len(ckpt_list))
            if len(ckpt_list) < 100:
                ckpt = ckpt_list[-1]
                flag = True
                break
        except:
            break

    for ratio in range(prev_ratio,11):
        print('#################### ratio : {} ####################'.format(int(ratio*10)))
        if ckpt and flag:
            ckpt_path = './checkpoint/ROAR/{}/{}/{}'.format(callback_name+args.coherence, int(prev_ratio*10), ckpt)
            train_model.load_weights(ckpt_path)
            prev_epoch = int(ckpt.split('_')[0])
            flag = False
            print('Load the checkpoint at {}.'.format(ckpt_path))

        else:
            prev_epoch = 0

        train_model.compile(optimizer=keras.optimizers.Adam(lr=args.lr, clipnorm=.001),
                            loss=keras.losses.categorical_crossentropy,
                            metrics=['acc'])

        train_generator = Generator(
            x=x_train,
            y=y_train,
            c=train_saliency,
            ratio=ratio,
            mode='train',
            batch_size=args.batch_size
        )
        test_generator = Generator(
            x=x_test,
            y=y_test,
            c=test_saliency,
            ratio=ratio,
            mode='test',
            batch_size=args.batch_size,
            shuffle=False
        )

        if not os.path.isdir('./checkpoint/ROAR/{}/{}'.format(callback_name+args.coherence, int(ratio*10))):
            os.makedirs('./checkpoint/ROAR/{}/{}'.format(callback_name+args.coherence, int(ratio*10)))

        if args.callbacks:
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    filepath='./checkpoint/ROAR/{}/{}'.format(callback_name+args.coherence, int(ratio*10))+'/{epoch:04d}_{val_acc:.4f}_{val_loss:.4f}.h5',
                    monitor='val_acc',
                    verbose=1,
                    mode='max',
                    save_best_only=False,
                    save_weights_only=True
                ),
                keras.callbacks.CSVLogger(
                    filename='./history/ROAR_{}_{}.csv'.format(callback_name+args.coherence, int(ratio*10)),
                    separator=',', append=True
                )
            ]
        else:
            callbacks = []

        history = train_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=args.steps if args.steps else int(len(y_train)//args.batch_size),
            epochs=args.epochs,
            validation_data=test_generator,
            validation_steps=int(len(y_test)//args.batch_size),
            callbacks=callbacks
        )

        train_model.set_weights(init_weights)

        



if __name__ == "__main__":
    main()