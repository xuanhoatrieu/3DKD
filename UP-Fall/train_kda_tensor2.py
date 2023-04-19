import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

import argparse
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import BinaryAccuracy, SensitivityAtSpecificity, SpecificityAtSensitivity
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_new_kda import get_data, generator_training_data, data_augmentation, generator_test_data
from resnet_tensor2copy import Resnet_3D
from epochcheckpoint_kda import EpochCheckpoint, ModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Training Falling Detection')
    parser.add_argument('--clip_len', type=int, default=16, help='clip length')
    parser.add_argument('--crop_size', type=int, default=224, help='crop size')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    parser.add_argument('--gpu', type=str, default='1', help='GPU id')
    parser.add_argument('--use_mse', type=int, default=1, help='use mse')
    parser.add_argument('--using_attention', type=int, default=1, help='set 1 to use attention otherwise is 0')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--drop_rate', type=float, default=0.5, help='drop rate')
    parser.add_argument('--reg_factor', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--start_epoch', type=int, default=1, help='manual epoch number (useful on restarts)')

    args = parser.parse_args()
    return args


class KD_Approach(Model):
    def __init__(self, teacher_model, student_model, use_mse):
        super(KD_Approach, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.use_mse = use_mse

    def compile(self, optimizer, ce_loss, alpha=0.1):
        super(KD_Approach, self).compile()
        self.optimizer = optimizer
        self.ce_loss = ce_loss
        self.alpha = alpha
        self.mse_loss = MeanSquaredError()

        self.accuracy_t = BinaryAccuracy()
        self.accuracy_s = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.Sensitivity = SensitivityAtSpecificity(0.5)
        self.Specificity = SpecificityAtSensitivity(0.5)

    @tf.function
    def train_step(self, data):
        # Unpack data
        X, y = data

        # ---------------------------------- Training teacher--------------------------------
        with tf.GradientTape() as t_tape:
            # Forward pass
            [predict_t, z_t] = self.teacher_model(X, training=True)

            # Compute ce loss
            ce_loss = self.ce_loss(y, predict_t)
            t_loss = ce_loss + sum(self.teacher_model.losses)

        # Compute gradients
        trainable_vars = (self.teacher_model.trainable_variables)
        gradients = t_tape.gradient(t_loss, trainable_vars)
        # Update weights for model
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()
        self.accuracy_t.update_state(y, predict_t)

        # ---------------------------------- Training student--------------------------------
        with tf.GradientTape() as s_tape:
            # Forward pass
            # [predict_t, z_t] = self.teacher_model(X, training=False)
            [predict_s, z_s] = self.student_model(X, training=True)

            # Compute ce loss
            ce_loss = self.ce_loss(y, predict_s)

            # Compute KLD loss
            mse0 = self.mse_loss(tf.stop_gradient(z_t[0]), z_s[0])
            mse1 = self.mse_loss(tf.stop_gradient(z_t[1]), z_s[1])
            mse2 = self.mse_loss(tf.stop_gradient(z_t[2]), z_s[2])
            mse3 = self.mse_loss(tf.stop_gradient(z_t[3]), z_s[3])
            # s_loss = ce_loss + self.alpha * (mse0 + mse1 + mse2 + mse3) + sum(self.student_model.losses)
            # s_loss = ce_loss + mse0 + mse1 + mse2 + mse3 + sum(self.student_model.losses)
            s_loss = ce_loss + self.alpha * mse0 + self.alpha * mse1 + self.alpha * mse2 + self.alpha * mse3 + sum(self.student_model.losses)

        # Compute gradients
        trainable_vars = (self.student_model.trainable_variables)
        gradients = s_tape.gradient(s_loss, trainable_vars)
        # Update weights for model
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()
        self.accuracy_s.update_state(y, predict_s)

        # Return a dict of performance
        results = {}
        # results = {m.name: m.result() for m in self.metrics}
        results.update({"Teacher_loss": t_loss, "Teacher_accuracy": self.accuracy_t.result(),
                        "Student_loss": s_loss, "Student_accuracy": self.accuracy_s.result()})
        return results

    @tf.function
    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        [predict, _] = self.student_model(x, training=False)

        # Calculate the loss for student 1
        celoss = self.ce_loss(y, predict)
        loss = celoss + sum(self.student_model.losses)

        # Update the metrics.
        self.val_accuracy.update_state(y, predict)
        self.Sensitivity.update_state(y, predict)
        self.Specificity.update_state(y, predict)

        # Update the metrics.
        # self.compiled_metrics.update_state(y, predict)

        # Return a dict of performance
        results = {}
        # results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss, "accuracy": self.val_accuracy.result(),
                        'sensitivity': self.Sensitivity.result(), 'specificity': self.Specificity.result()})
        return results


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def build_callbacks(save_path, startAt):
    jsonPath = os.path.join(save_path, "output")
    jsonName = 'log_results.json'
    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, mode='max')
    Checkpoint = ModelCheckpoint(folderpath=save_path, jsonPath=jsonPath, jsonName=jsonName, startAt=startAt, monitor='val_accuracy',
                                 mode='max', verbose=1)
    outputPath = os.path.join(save_path, "checkpoints")
    epoch_checkpoint = EpochCheckpoint(outputPath=outputPath, every=1, startAt=startAt)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='max', min_lr=1e-8, startAt=startAt)
    return [earlyStopping, reduce_lr, Checkpoint, epoch_checkpoint]


def training(train_dataset, test_dataset, input_shape, lr_init=0.01, use_mse_loss=True, batch_size=32, reg_factor=5e-4,
             drop_rate=0.5, alpha=0.1, use_attention = False,
             epochs=200, start_epoch = 1, save_path='save_model'):
    # Prepare data for training phase
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds_train = tf.data.Dataset.from_generator(generator_training_data,
                                              (tf.float32, tf.float32),
                                              (tf.TensorShape(input_shape),
                                               tf.TensorShape([])),
                                              args=[train_dataset, input_shape[0], input_shape[1]])
    ds_train = ds_train.map(data_augmentation, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)

    ds_test = tf.data.Dataset.from_generator(generator_test_data, (tf.float32, tf.float32),
                                             (tf.TensorShape(input_shape), tf.TensorShape([])),
                                             args=[test_dataset, input_shape[0], input_shape[1]])
    ds_test = ds_test.batch(batch_size).prefetch(AUTOTUNE)

    # Build model
    resnet_50 = Resnet_3D(input_shape=input_shape, num_classes=1, repetitions=[3, 4, 6, 3], name_model='resnet50',
                          using_KD=True, using_attention=use_attention, drop_rate=drop_rate)
    resnet_18 = Resnet_3D(input_shape=input_shape, num_classes=1, repetitions=[2, 2, 2, 2], name_model='resnet18',
                          using_KD=True, using_attention=use_attention, drop_rate=drop_rate)
    # base_model.summary(line_length=150)
    # Load weight for student model
    if start_epoch > 1:
        print('--------------------------START LOAD WEIGHT FROM CURRENT EPOCH---------------------------------')
        path1 = os.path.join(save_path, 'checkpoints', 'kda_t_epoch_' + str(start_epoch) + '.h5')
        resnet_50.load_weights(path1)
        path2 = os.path.join(save_path, 'checkpoints', 'kda_s_epoch_' + str(start_epoch) + '.h5')
        resnet_18.load_weights(path2)
        print('--------------------------LOAD WEIGHT COMPLETED---------------------------------')
    # Build callbacks
    callback = build_callbacks(save_path, startAt=start_epoch)

    # Build optimizer and loss function
    optimizer = SGD(learning_rate=lr_init, momentum=0.9, nesterov=True)
    ce_loss = BinaryCrossentropy()

    # Build model
    Model_KD = KD_Approach(teacher_model=resnet_50, student_model=resnet_18, use_mse=use_mse_loss)
    Model_KD.compile(
        optimizer,
        ce_loss,
        alpha=alpha,
    )

    # Training
    history = Model_KD.fit(ds_train, epochs=epochs - start_epoch + 1, verbose=1,
                 steps_per_epoch=(len(train_dataset)*7 // batch_size),
                #  steps_per_epoch=10,
                 validation_data=ds_test,
                 validation_steps=len(test_dataset)*7 // (batch_size),
                 callbacks=callback
                 )
    # Convert the history dictionary to a dataframe
    history_df = pd.DataFrame(history.history)
    # Save the dataframe to a CSV file
    history_df.to_csv('kda_up_history.csv', index=False)
    
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='upper left')             
    # plt.savefig("KD_model.jpg")

def testing(test_dataset, args, save_path='save_model', lr_init=0.01, use_attention = False, drop_rate=0.5):
    input_shape = (args.clip_len, args.crop_size, args.crop_size, 3)
    # Prepare data for training phase
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds_test = tf.data.Dataset.from_generator(generator_test_data, (tf.float32, tf.float32),
                                             (tf.TensorShape(input_shape), tf.TensorShape([])),
                                             args=[test_dataset, input_shape[0], input_shape[1]])
    ds_test = ds_test.batch(args.batch_size).prefetch(AUTOTUNE)
    file_weight = os.path.join(save_path, "kda_best_s_model.h5")
    model = Resnet_3D(input_shape=input_shape, num_classes=1, repetitions=[2, 2, 2, 2], name_model='resnet18',
                          using_KD=False, using_attention=use_attention, drop_rate=drop_rate)
    model.summary(line_length=150)
    model.load_weights(file_weight)

    # Build callbacks
    # optimizer = Adam(lr=0.001)
    optimizer = SGD(learning_rate=lr_init, momentum=0.9, nesterov=True)
    ce_loss = BinaryCrossentropy()
    # metrics = [Accuracy(), Recall(), Precision()] --> Xem cach' tinh' cac' metrics Accuracy, Precision, Recall o duoi'
    model.compile(optimizer=optimizer, loss=ce_loss, metrics=['accuracy', precision_m, recall_m])
    # testing
    _, acc, precision, recall = model.evaluate(ds_test, verbose=1, steps=(len(test_dataset) // args.batch_size))
    # kq = model.evaluate(ds_test, verbose=1, steps=(len(test_dataset) // args.batch_size))
    # print(kq)
    # Compute and print F1 score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    print('---------------Results on the test dataset--------------')
    print('Accuracy: ', acc)
    print('Precision: ', precision)
    print('Recal: ', recall)
    print('F1 score: ', f1_score)

def predic(test_dataset, args, input_shape, save_path, use_attention, drop_rate, lr_init = 0.01):
    input_shape = (args.clip_len, args.crop_size, args.crop_size, 3)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    y_test=[]
    for row in test_dataset:
        # print(len(row[0]))
        if str(row[0])[0] == 'N':
            y_test.append(0)
        else:
            y_test.append(1)
    # print(y_test)    

    ds_test = tf.data.Dataset.from_generator(generator_test_data, (tf.float32, tf.float32),
                                             (tf.TensorShape(input_shape), tf.TensorShape([])),
                                             args=[test_dataset, input_shape[0], input_shape[1]])
    ds_test = ds_test.batch(1).prefetch(AUTOTUNE)
    file_weight = os.path.join(save_path, "kda_best_s_model.h5")
    model = Resnet_3D(input_shape=input_shape, num_classes=1, repetitions=[2, 2, 2, 2], name_model='resnet18',
                          using_KD=False, using_attention=use_attention, drop_rate=drop_rate)
    # model.summary(line_length=150)
    model.load_weights(file_weight)
    # Build callbacks
    optimizer = SGD(learning_rate=lr_init, momentum=0.9, nesterov=True)
    ce_loss = BinaryCrossentropy()
    # metrics = [Accuracy(), Recall(), Precision()] --> Xem cach' tinh' cac' metrics Accuracy, Precision, Recall o duoi'
    model.compile(optimizer=optimizer, loss=ce_loss, metrics=['accuracy', precision_m, recall_m])
    
    y_predict = model.predict(ds_test, steps=len(y_test))
    y_predict[y_predict>0.5] = 1
    y_predict[y_predict<=0.5] = 0
    np.savez('kda_up_y_predict.npz', data=y_predict)

    # x = np.load('y_predict.npz')['data']

    y_test = np.asarray(y_test)
    cm = confusion_matrix(y_test, y_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('kda_up_confusion_matrix.png')
    plt.show()

def cm_cal():
    y_predict = np.load('kda_up_y_predict.npz')['data']
    test_dataset = get_data('test.csv')
    y_test=[]
    for row in test_dataset:
        y_test.append(int(row[-1]))    
    y_test = np.asarray(y_test)
    cm = confusion_matrix(y_test, y_predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('kda_up_confusion_matrix.png')
    plt.show()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # Choose GPU for training

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    input_shape = (args.clip_len, args.crop_size, args.crop_size, 3)
    reg_factor = args.reg_factor
    batch_size = args.batch_size
    epochs = args.epochs
    start_epoch = args.start_epoch
    lr_init = args.lr
    alpha = args.alpha
    drop_rate = args.drop_rate
    temp = args.use_mse
    if temp > 0:
        use_mse_loss = True
    else:
        use_mse_loss = False

    if args.using_attention == 1:
        use_attention = True
    else:
        use_attention = False

    # Read dataset
    if use_mse_loss:
        save_path = './save_model/'
    else:
        save_path = './save_normal_model/'
    train_dataset = get_data('train.csv')
    test_dataset = get_data('test.csv')
    # random.shuffle(test_dataset)
    random.shuffle(train_dataset)
    print('Train set:', len(train_dataset))
    print('Test set:', len(test_dataset))

    # Create folders for callback
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(os.path.join(save_path, "output")):
        os.mkdir(os.path.join(save_path, "output"))
    if not os.path.exists(os.path.join(save_path, "checkpoints")):
        os.mkdir(os.path.join(save_path, "checkpoints"))

    # Write all config to file
    f = open(os.path.join(save_path, 'config.txt'), "w")
    f.write('input shape: ' + str(input_shape) + '\n')
    f.write('reg factor: ' + str(reg_factor) + '\n')
    f.write('batch size: ' + str(batch_size) + '\n')
    f.write('numbers of epochs: ' + str(epochs) + '\n')
    f.write('start epoch: ' + str(start_epoch) + '\n')
    f.write('lr init: ' + str(lr_init) + '\n')
    f.write('alpha: ' + str(alpha) + '\n')
    f.write('Drop rate: ' + str(drop_rate) + '\n')
    f.close()

    # --------------------------------------Training ----------------------------------------
    # training(train_dataset, test_dataset, input_shape, lr_init, use_mse_loss=use_mse_loss, use_attention = use_attention,
    #          batch_size=batch_size, reg_factor=reg_factor, drop_rate=drop_rate, alpha=alpha,
    #          epochs=epochs, start_epoch=start_epoch, save_path=save_path)
    
    testing(test_dataset, args, save_path=save_path, lr_init=lr_init, use_attention = use_attention, drop_rate=drop_rate)
    # predic(test_dataset, args, input_shape=input_shape, save_path=save_path, lr_init=lr_init, use_attention = use_attention, drop_rate=drop_rate)
    # cm_cal()

if __name__ == '__main__':
    print(tf.__version__)
    main()

