from tf_record_utility import TFRecordUtility
from configuration import DatasetName, DatasetType, AffectnetConf, IbugConf, W300Conf, InputDataSize
from cnn_model import CNNModel
from pca_utility import PCAUtility
from image_utility import ImageUtility
import numpy as np
from train import Train
from test import Test

# from Train_Gan import TrainGan

if __name__ == '__main__':
    tf_record_util = TFRecordUtility()
    pca_utility = PCAUtility()
    cnn_model = CNNModel()
    image_utility = ImageUtility()

    # tf_record_util.test_hm_accuracy()

    # tf_record_util.create_adv_att_img_hm()

    '''augment, normalize, and save pts'''

    # tf_record_util.rotaate_and_save(dataset_name=DatasetName.ibug)
    # tf_record_util.random_augment_from_rotated(dataset_name=DatasetName.ibug)

    '''normalize the points and save'''
    # tf_record_util.normalize_points_and_save(dataset_name=DatasetName.ibug)
    '''generate pose using hopeNet'''
    # tf_record_util.detect_pose_and_save(dataset_name=DatasetName.ibug)
    '''create tfRecord:'''
    # tf_record_util.create_tf_record(dataset_name=DatasetName.ibug, dataset_type=None, heatmap=False)
    '''retrive and test tfRecords'''
    # tf_record_util.test_tf_record()

    '''create and save PCA objects'''
    # pca_utility.create_pca_from_points(DatasetName.ibug, 95)
    # pca_utility.create_pca_from_npy(DatasetName.ibug, 90)

    '''generate heatmap with different accuracy'''

    # mat = np.random.randint(0, 10, size=10)
    # cnn_model.generate_distance_matrix(mat)

    # cnn_model.init_for_test()

    # trg = TrainGan()
    # trg.create_seq_model()

    # test = Test(dataset_name=DatasetName.ibug, arch='ASMNet', num_output_layers=2, weight_fname='weights-41-0.00429.h5')

    trainer = Train(use_tf_record=True,
                    dataset_name=DatasetName.ibug,
                    custom_loss=False,
                    # arch='ASMNet',
                    arch='mobileNetV2',
                    inception_mode=False,
                    num_output_layers=2,
                    # weight='00-weights-61-0.01995.h5',
                    weight=None,
                    train_on_batch=False,
                    accuracy=100)






