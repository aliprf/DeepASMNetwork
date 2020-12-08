from configuration import DatasetName, DatasetType, \
    AffectnetConf, IbugConf, W300Conf, InputDataSize, LearningConfig, CofwConf, WflwConf
from tf_record_utility import TFRecordUtility
from clr_callback import CyclicLR
from cnn_model import CNNModel
from custom_Losses import Custom_losses
from Data_custom_generator import CustomHeatmapGenerator

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray
import csv
from skimage.io import imread
import pickle
from image_utility import ImageUtility
import img_printer as imgpr


class Train:
    def __init__(self, dataset_name, asm_accuracy=90):
        self.dataset_name = dataset_name
        self.asm_accuracy = asm_accuracy

        if dataset_name == DatasetName.ibug or dataset_name == DatasetName.w300:
            self.num_landmark = IbugConf.num_of_landmarks * 2
            # self.img_path = IbugConf.no_aug_train_image
            # self.annotation_path = IbugConf.no_aug_train_annotation
            self.img_path = IbugConf.augmented_train_image
            self.annotation_path = IbugConf.augmented_train_annotation

        if dataset_name == DatasetName.cofw:
            self.num_landmark = CofwConf.num_of_landmarks * 2
            self.img_path = CofwConf.augmented_train_image
            self.annotation_path = CofwConf.augmented_train_annotation

        if dataset_name == DatasetName.wflw:
            self.num_landmark = WflwConf.num_of_landmarks * 2
            self.img_path = WflwConf.augmented_train_image
            self.annotation_path = WflwConf.augmented_train_annotation

    # def train_on_batch(self, arch, weight_path):
    #     _lr = 1e-2
    #     model = self.make_model(arch=arch, w_path=weight_path)
    #     '''create optimizer'''
    #     optimizer = self._get_optimizer(lr=_lr, beta_1=0.5, beta_2=0.999, decay=1e-6)
    #
    #     '''create sample generator'''
    #     x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = self._create_generators()
    #
    #     '''create train configuration'''
    #     step_per_epoch = len(x_train_filenames) // LearningConfig.batch_size
    #
    #     for epoch in range(LearningConfig.epochs):
    #         for batch_index in range(step_per_epoch):

    def train(self, arch, weight_path):
        '''create loss'''
        c_loss = Custom_losses(dataset_name=self.dataset_name, accuracy=90)

        '''create summary writer'''
        summary_writer = tf.summary.create_file_writer(
            "./train_logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

        '''making models'''
        _lr=0.1
        model = self.make_model(arch=arch, w_path=weight_path)
        '''create optimizer'''
        optimizer = self._get_optimizer(lr=_lr)

        '''create sample generator'''
        x_train_filenames, y_train_filenames = self._create_generators()

        '''create train configuration'''
        step_per_epoch = len(x_train_filenames) // LearningConfig.batch_size

        '''create highlighted points'''
        # todo: need to complete this:
        bold_landmarks_point_map = self.create_FL_highligted_points_map(batch_size=LearningConfig.batch_size,
                                                                        num_of_landmark=self.num_landmark,
                                                                        ds_name=self.dataset_name)
        '''start train:'''
        for epoch in range(LearningConfig.epochs):
            x_train_filenames, y_train_filenames = self._create_generators()
            for batch_index in range(step_per_epoch):
                '''load annotation and images'''
                images, annotation_gr, annotation_asm, annotation_asm_prime = self._get_batch_sample(
                    batch_index=batch_index, x_train_filenames=x_train_filenames,
                    y_train_filenames=y_train_filenames)
                '''convert to tensor'''
                images = tf.cast(images, tf.float32)
                annotation_gr = tf.cast(annotation_gr, tf.float32)
                '''train step'''
                self.train_step(epoch=epoch, step=batch_index, total_steps=step_per_epoch, images=images, model=model,
                                annotation_gr=annotation_gr, annotation_asm=annotation_asm,
                                annotation_asm_prime=annotation_asm_prime,
                                optimizer=optimizer, summary_writer=summary_writer, c_loss=c_loss,
                                bold_landmarks_point_map=bold_landmarks_point_map)
            '''save weights'''
            model.save('./models/asm_fw_model_' + str(epoch) + '_' + self.dataset_name + '_.h5')
            model.save_weights('./models/asm_fw_weight_' + '_' + str(epoch) + self.dataset_name + '_.h5')
            if epoch != 0 and epoch % 100 == 0:
                _lr -= _lr * 0.4
                optimizer = self._get_optimizer(lr=_lr)

    # @tf.function
    def train_step(self, epoch, step, total_steps, images, model, annotation_gr,
                   annotation_asm, annotation_asm_prime,
                   optimizer, summary_writer, c_loss, bold_landmarks_point_map):
        with tf.GradientTape() as tape:
            '''create annotation_predicted'''
            annotation_predicted = model(images,  training=True)
            '''calculate loss'''
            loss_total, loss_main, loss_asm, loss_fw = c_loss.asm_assisted_loss(x_pr=annotation_predicted,
                                                                                x_gt=annotation_gr,
                                                                                x_asm=annotation_asm,
                                                                                x_asm_prime=annotation_asm_prime,
                                                                                main_loss_weight=1.5,
                                                                                asm_loss_weight=0.9,
                                                                                fw_loss_weight=0.9,
                                                                                ds_name=self.dataset_name,
                                                                                bold_landmarks_point_map=bold_landmarks_point_map)
        '''calculate gradient'''
        gradients_of_model = tape.gradient(loss_total, model.trainable_variables)
        '''apply Gradients:'''
        optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))
        '''printing loss Values: '''
        tf.print("->EPOCH: ", str(epoch), "->STEP: ", str(step) + '/' + str(total_steps), ' -> : LOSS: ', loss_total,
                 ' -> : loss_main: ', loss_main, ' -> : loss_asm: ', loss_asm, ' -> : loss_fw: ', loss_fw)
        # print('==--==--==--==--==--==--==--==--==--')
        with summary_writer.as_default():
            tf.summary.scalar('LOSS', loss_total, step=epoch)
            tf.summary.scalar('loss_main', loss_main, step=epoch)
            tf.summary.scalar('loss_asm', loss_asm, step=epoch)
            tf.summary.scalar('loss_fw', loss_fw, step=epoch)

    def create_FL_highligted_points_map(self, ds_name, batch_size, num_of_landmark):
        # todo
        weight_map = np.ones(shape=[batch_size, num_of_landmark])
        if ds_name == DatasetName.cofw:
            indices = [16, 17, 18, 19, 20, 21, 22, 23, 44, 45, 46, 47, 56, 57, 42, 43]
            for index in indices: weight_map[:, index] = 10

        elif ds_name == DatasetName.ibug:
            indices = [0, 1, 32, 33, 8, 9, 24, 25, 114, 115, 102, 103, 66, 67, 72, 73, 90, 91, 78, 79, 84, 85, 54, 55,
                       34, 35, 52, 53, 42, 43, 44, 45]
            for index in indices: weight_map[:, index] = 10

        elif ds_name == DatasetName.wflw:
            indices = [0, 1, 64, 65, 14, 15, 50, 51, 32, 33, 179, 171, 158, 159, 152, 153, 164, 165, 114, 115, 102, 103,
                       120, 121, 144, 145, 128, 129, 136, 137, 66, 67, 92, 93, 76, 77, 100, 101]
            for index in indices: weight_map[:, index] = 10

        return weight_map

    def make_model(self, arch, w_path):
        cnn = CNNModel()
        model = cnn.get_model(arch=arch, output_len=self.num_landmark)
        if w_path is not None:
            model.load_weights(w_path)
        return model

    def _get_optimizer(self, lr=1e-2, beta_1=0.9, beta_2=0.999, decay=1e-4):
        return tf.keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
        # return tf.keras.optimizers.SGD(lr=lr)

    def _create_generators(self):
        fn_prefix = './file_names/' + self.dataset_name + '_'
        x_trains_path = fn_prefix + 'x_train_fns.npy'
        x_validations_path = fn_prefix + 'x_val_fns.npy'

        tf_utils = TFRecordUtility(number_of_landmark=self.num_landmark)

        filenames, labels = tf_utils.create_image_and_labels_name(img_path=self.img_path,
                                                                  annotation_path=self.annotation_path)
        filenames_shuffled, y_labels_shuffled = shuffle(filenames, labels)
        # x_train_filenames, x_val_filenames, y_train, y_val = train_test_split(
        #     filenames_shuffled, y_labels_shuffled, test_size=0.01, random_state=1)

        # save(x_trains_path, filenames_shuffled)
        # save(x_validations_path, y_labels_shuffled)

        # save(x_trains_path, x_train_filenames)
        # save(x_validations_path, x_val_filenames)
        # save(y_trains_path, y_train)
        # save(y_validations_path, y_val)

        return filenames_shuffled, y_labels_shuffled
        # return x_train_filenames, x_val_filenames, y_train, y_val

    def _get_batch_sample(self, batch_index, x_train_filenames, y_train_filenames):
        img_path = self.img_path
        pn_tr_path = self.annotation_path
        tf_utils = TFRecordUtility(self.num_landmark)
        '''create batch data and normalize images'''
        batch_x = x_train_filenames[
                  batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
        batch_y = y_train_filenames[
                  batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
        '''create img and annotations'''
        img_batch = np.array([imread(img_path + file_name) for file_name in batch_x]) / 255.0
        # if self.dataset_name == DatasetName.cofw:  # this ds is not normalized
        #     pn_batch = np.array([load(pn_tr_path + file_name) for file_name in batch_y])
        #     '''creating y_asm'''
        #     pn_batch_asm = np.array([tf_utils.get_asm(input=load(pn_tr_path + file_name),
        #                                               dataset_name=self.dataset_name, accuracy=self.asm_accuracy)
        #                              for file_name in batch_y])
        # else:
        pn_batch = np.array([self._load_and_normalize(pn_tr_path + file_name) for file_name in batch_y])
        '''creating y_asm'''
        pn_batch_asm = np.array([tf_utils.get_asm(input=self._load_and_normalize(pn_tr_path + file_name),
                                                  dataset_name=self.dataset_name, accuracy=self.asm_accuracy)
                                     for file_name in batch_y])

        pn_batch_asm_prim = np.array(
            [pn_batch[i] - np.sign(pn_batch_asm[i] - pn_batch[i]) * abs(pn_batch_asm[i] - pn_batch[i]) for i in
             range(len(pn_batch_asm))])
        '''test: print'''
        # image_utility = ImageUtility()
        # for i in range(LearningConfig.batch_size):
        #     gr_s, gr_px_1, gr_Py_1 = image_utility.create_landmarks_from_normalized(pn_batch[i], 224, 224, 112, 112)
        #     asm_p_s, asm_px_1, asm_Py_1 = image_utility.create_landmarks_from_normalized(pn_batch_asm[i], 224, 224, 112,
        #                                                                                  112)
        #     asm_p_s, asm_p_px_00, asm_p_Py_00 = image_utility.create_landmarks_from_normalized(pn_batch_asm_prim[i],
        #                                                                                        224, 224, 112, 112)
        #     # imgpr.print_image_arr_multi(str(batch_index+1*(i+1))+'pts_gt', img_batch[i],
        #     #                           [gr_px_1, asm_px_1, asm_p_px_00], [gr_Py_1, asm_Py_1, asm_p_Py_00])
        #     imgpr.print_image_arr(str(batch_index + 1 * (i + 1)) + 'pts_gt', img_batch[i], gr_px_1, gr_Py_1)
        #     print(str(batch_index + 1 * (i + 1)))
        #     # imgpr.print_image_arr(str(batch_index + 1 * (i + 1)) + 'pts_asm', img_batch[i], asm_px_1, asm_Py_1)
        #     # imgpr.print_image_arr(str(batch_index + 1 * (i + 1)) + 'pts_asm_prim', img_batch[i], asm_p_px_00,
        #     #                       asm_p_Py_00)

        return img_batch, pn_batch, pn_batch_asm, pn_batch_asm_prim

    def _load_and_normalize(self, point_path):
        annotation = load(point_path)

        """for training we dont normalize COFW"""

        '''normalize landmarks based on hyperface method'''
        width = InputDataSize.image_input_size
        height = InputDataSize.image_input_size
        x_center = width / 2
        y_center = height / 2
        annotation_norm = []
        for p in range(0, len(annotation), 2):
            annotation_norm.append((x_center - annotation[p]) / width)
            annotation_norm.append((y_center - annotation[p + 1]) / height)
        return annotation_norm
