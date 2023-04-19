# import the necessary packages
import os
import numpy as np
import json
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import BaseLogger


def confidence_schedule(epoch, confidence_init):
    if epoch < 30:
        return confidence_init
    if epoch >= 30 and epoch < 70:
        return confidence_init + 0.05
    if epoch >= 70 and epoch < 110:
        return confidence_init + 0.10
    if epoch >= 110 and epoch < 140:
        return confidence_init + 0.15
    if epoch >= 140 and epoch < 170:
        return confidence_init + 0.20
    if epoch >= 170:
        return confidence_init + 0.25

class LearningRateScheduler(Callback):
    def __init__(self, schedule, startAt=1, verbose=0, lr_init=0.001):
        # call the parent constructor
        super(Callback, self).__init__()
        self.schedule = schedule
        self.intEpoch = startAt
        self.verbose = verbose
        self.lr_init = lr_init

    def on_epoch_begin(self, epoch, logs=None):
        current_epoch = self.intEpoch + epoch
        lr = self.schedule(current_epoch, self.lr_init)
        K.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (current_epoch, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # logs['lr'] = K.get_value(self.model.optimizer_student1.lr)

class Confidence_Scheduler(Callback):
    def __init__(self, confidence_schedule, startAt=1, verbose=0, confidence_init=0.7):
        # call the parent constructor
        super(Callback, self).__init__()
        self.confidence_schedule = confidence_schedule
        self.intEpoch = startAt
        self.verbose = verbose
        self.confidence_init = confidence_init

    def on_epoch_begin(self, epoch, logs=None):
        current_epoch = self.intEpoch + epoch
        confidence = self.confidence_schedule(current_epoch, self.confidence_init)
        # K.set_value(self.model.confidence, confidence)
        self.model.confidence = confidence

        if self.verbose > 0:
            print('\nEpoch %05d: Confidence Scheduler reducing to %s.' % (current_epoch, confidence))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)

class WarmUpCosineDecayScheduler(Callback):
    """Cosine decay with warmup learning rate scheduler
    """
    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))

class EpochCheckpoint(Callback):
    def __init__(self, outputPath, every=5, startAt=0):
        # call the parent constructor
        super(Callback, self).__init__()

        # store the base output path for the model, the number of
        # epochs that must pass before the model is serialized to
        # disk and the current epoch value
        self.outputPath = outputPath
        self.every = every
        self.intEpoch = startAt

    def on_epoch_end(self, epoch, logs={}):
        # check to see if the model should be serialized to disk
        if (self.intEpoch) % self.every == 0:
            # Save current model weight
            p1 = os.path.sep.join([self.outputPath, "kda_t_epoch_{}.h5".format(self.intEpoch)])
            self.model.teacher_model.save_weights(p1, overwrite=True)
            p2 = os.path.sep.join([self.outputPath, "kda_s_epoch_{}.h5".format(self.intEpoch)])
            self.model.student_model.save_weights(p2, overwrite=True)

            # Delete old model weight
            old_p1 = os.path.sep.join([self.outputPath, "kda_s_epoch_{}.h5".format(self.intEpoch - self.every)])
            old_p2 = os.path.sep.join([self.outputPath, "kda_t_epoch_{}.h5".format(self.intEpoch - self.every)])
            if os.path.exists(old_p1):
                os.remove(old_p1)
            if os.path.exists(old_p2):
                os.remove(old_p2)

        # increment the internal epoch counter
        self.intEpoch += 1
# class Switch_Models(Callback):
#     def __init__(self, num_epoch_switch=1, startAt=1, verbose=0):
#         # call the parent constructor
#         super(Callback, self).__init__()
#         self.num_epoch_switch = num_epoch_switch
#         self.intEpoch = startAt
#         self.verbose = verbose

#     def on_epoch_end(self, epoch, logs={}):
#         if self.intEpoch % self.num_epoch_switch == 0:
#             temp_weights = self.model.student_model.get_weights()
#             self.model.student_model.set_weights(self.model.teacher_model.get_weights())
#             self.model.teacher_model.set_weights(temp_weights)
#             if self.verbose ==1:
#                 print('Switch model weights completed!')

#         self.intEpoch +=1


class ModelCheckpoint(BaseLogger):
    def __init__(self, folderpath, jsonPath=None, jsonName=None, startAt=0, monitor='val_accuracy', mode='max',
                 lr_init=0.01, verbose=1):
        super(ModelCheckpoint, self).__init__()
        self.filepath_t = os.path.join(folderpath, 'kda_best_t_model.h5')
        self.filepath_s = os.path.join(folderpath, 'kda_best_s_model.h5')
        self.jsonPath = jsonPath
        self.jsonName = jsonName
        self.jsonfile = os.path.join(self.jsonPath, self.jsonName)
        self.startAt = startAt
        self.monitor = monitor
        self.mode = mode
        self.count_epoch = 0
        self.lr_init = lr_init
        self.verbose = verbose

        if self.mode == 'max':
            self.monitor_op = np.greater
            self.current_best = -np.Inf
        else:
            self.monitor_op = np.less
            self.current_best = np.Inf

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}
        K.set_value(self.model.optimizer.lr, self.lr_init)
        # if the JSON history path exists, load the training history
        if self.jsonfile is not None:
            if os.path.exists(self.jsonfile)  and self.startAt > 1:
                self.H = json.loads(open(self.jsonfile).read())
                if self.mode == 'max':
                    self.current_best = max(self.H[self.monitor])
                    self.idx = self.H[self.monitor].index(self.current_best)
                else:
                    self.current_best = min(self.H[self.monitor])
                    self.idx = self.H[self.monitor].index(self.current_best)
                for k in self.H.keys():
                    self.H[k] = self.H[k][:self.startAt]
                self.count_epoch = self.startAt - self.idx

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        current = logs.get(self.monitor)
        lr = K.get_value(self.model.optimizer.lr)
        print('\nCurrent best accuracy: ', self.current_best)
        print('Current LR: %0.5f and count epoch: %03d.' % (lr, self.count_epoch))

        if self.monitor_op(current, self.current_best):
            self.count_epoch = 0
            if self.verbose > 0:
                print('Count epoch is reduced to 0.')
                print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                      ' saving model to %s'
                      % (epoch + 1, self.monitor, self.current_best, current, self.filepath_s))

            self.model.teacher_model.save_weights(self.filepath_t, overwrite=True)
            self.model.student_model.save_weights(self.filepath_s, overwrite=True)
            self.current_best = current
        else:
            self.count_epoch += 1

        # # Reduce LR
        # if self.count_epoch == self.patience:
        #     new_lr = lr * self.factor
        #     if new_lr < self.min_lr:
        #         new_lr = self.min_lr
        #     K.set_value(self.model.optimizer.lr, new_lr)
        #     self.count_epoch = 0
        #     if self.verbose > 0:
        #         print('Epoch %05d: LearningRateScheduler reducing learning '
        #               'rate to %s.' % (epoch + self.startAt, new_lr))

        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(float(v))
            self.H[k] = l

        # check to see if the training history should be serialized
        # to file
        if self.jsonfile is not None:
            f = open(self.jsonfile, "w")
            f.write(json.dumps(self.H))
            f.close()
