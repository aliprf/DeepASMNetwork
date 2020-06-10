from tf_record_utility import TFRecordUtility
from configuration import DatasetName, DatasetType, AffectnetConf, IbugConf, W300Conf, InputDataSize, CofwConf, WflwConf
from cnn_model import CNNModel
from pca_utility import PCAUtility
from image_utility import ImageUtility
import numpy as np
from train import Train
from test import Test

# from Train_Gan import TrainGan

if __name__ == '__main__':
    tf_record_util = TFRecordUtility(IbugConf.num_of_landmarks*2)
    pca_utility = PCAUtility()
    cnn_model = CNNModel()
    image_utility = ImageUtility()

    # tf_record_util.test_hm_accuracy()
    # tf_record_util.create_adv_att_img_hm()

    '''--> Preparing Test Data process:'''
    # tf_record_util.crop_and_save(dataset_name=DatasetName.cofw_test)
    # tf_record_util.normalize_points_and_save(dataset_name=DatasetName.cofw_test)
    # tf_record_util.detect_pose_and_save(dataset_name=DatasetName.cofw_test)
    # tf_record_util.create_tf_record(dataset_name=DatasetName.cofw_test, dataset_type=None, heatmap=False, accuracy=100)

    '''--> Preparing Train Data process:'''
    '''     augment, normalize, and save pts'''
    # tf_record_util.rotaate_and_save(dataset_name=DatasetName.cofw)
    # tf_record_util.random_augment_from_rotated(dataset_name=DatasetName.cofw)
    '''     normalize the points and save'''
    # tf_record_util.normalize_points_and_save(dataset_name=DatasetName.cofw)
    '''     generate pose using hopeNet'''
    # tf_record_util.detect_pose_and_save(dataset_name=DatasetName.cofw_test)
    '''     create and save PCA objects'''
    # pca_utility.create_pca_from_points(DatasetName.ibug, 95)
    # pca_utility.create_pca_from_npy(DatasetName.cofw, 95)
    '''     create tfRecord:'''
    # tf_record_util.create_tf_record(dataset_name=DatasetName.wflw, dataset_type=None, heatmap=False, accuracy=95)

    '''--> retrive and test tfRecords'''
    # tf_record_util.test_tf_record()

    '''--> Train GAN:'''
    # trg = TrainGan()
    # trg.create_seq_model()

    '''--> Evaluate Results'''
    # test = Test(dataset_name=DatasetName.ibug, arch='ASMNet', num_output_layers=2, weight_fname='weights-41-0.00429.h5')
    # test = Test(dataset_name=DatasetName.cofw, arch='mobileNetV2', num_output_layers=2, weight_fname='weights-120-0.00021.h5')
    #

    '''--> Train Model'''
    # trainer = Train(use_tf_record=True,
    #                 dataset_name=DatasetName.cofw,
    #                 custom_loss=True,
    #                 arch='ASMNet',
    #                 # arch='mobileNetV2',
    #                 inception_mode=False,
    #                 num_output_layers=2,
    #                 # weight='00-w-dasm.h5',
    #                 weight=None,
    #                 train_on_batch=False,
    #                 accuracy=95)






