class DatasetName:
    affectnet = 'affectnet'
    w300 = 'w300'
    ibug = 'ibug'
    aflw = 'aflw'
    aflw2000 = 'aflw2000'



class DatasetType:
    data_type_train = 0
    data_type_validation = 1
    data_type_test = 2


class LearningConfig:
    weight_loss_heatmap_face = 4.0
    weight_loss_heatmap_all_face = 4.0
    weight_loss_regression_face = 2.0
    weight_loss_regression_pose = 1.0

    weight_loss_heatmap_face_inc1 = 0.5
    weight_loss_heatmap_all_face_inc1 = 0.5
    weight_loss_regression_face_inc1 = 0.25
    weight_loss_regression_pose_inc1 = 0.125

    weight_loss_heatmap_face_inc2 = 1.0
    weight_loss_heatmap_all_face_inc2 = 1.0
    weight_loss_regression_face_inc2 = 0.5
    weight_loss_regression_pose_inc2 = 0.25

    weight_loss_heatmap_face_inc3 = 1.5
    weight_loss_heatmap_all_face_inc3 = 1.5
    weight_loss_regression_face_inc3 = 0.75
    weight_loss_regression_pose_inc3 = 0.375

    loss_weight_inception_1_face = 2
    # loss_weight_inception_1_pose = 1

    loss_weight_inception_2_face = 5
    # loss_weight_inception_2_pose = 2

    loss_weight_inception_3_face = 8
    # loss_weight_inception_3_pose = 3

    loss_weight_pose = 0.5

    loss_weight_face = 1
    loss_weight_nose = 1
    loss_weight_eyes = 1
    loss_weight_mouth = 1

    CLR_METHOD = "triangular"
    MIN_LR = 1e-7
    MAX_LR = 1e-2
    STEP_SIZE = 10
    batch_size = 70
    steps_per_validation_epochs = 5

    epochs = 200
    landmark_len = 136
    point_len = 68
    pose_len = 3

    reg_term_ASM = 0.8



class InputDataSize:
    image_input_size = 224
    landmark_len = 136
    landmark_face_len = 54
    landmark_nose_len = 18
    landmark_eys_len = 24
    landmark_mouth_len = 40
    pose_len = 3


class AffectnetConf:
    csv_train_path = '/media/ali/extradata/facial_landmark_ds/affectNet/training.csv'
    csv_evaluate_path = '/media/ali/extradata/facial_landmark_ds/affectNet/validation.csv'
    csv_test_path = '/media/ali/extradata/facial_landmark_ds/affectNet/test.csv'

    tf_train_path = '/media/ali/extradata/facial_landmark_ds/affectNet/train.tfrecords'
    tf_test_path = '/media/ali/extradata/facial_landmark_ds/affectNet/eveluate.tfrecords'
    tf_evaluation_path = '/media/ali/extradata/facial_landmark_ds/affectNet/test.tfrecords'

    sum_of_train_samples = 200000  # 414800
    sum_of_test_samples = 30000
    sum_of_validation_samples = 5500

    img_path_prefix = '/media/ali/extradata/facial_landmark_ds/affectNet/Manually_Annotated_Images/'


class Multipie:
    lbl_path_prefix = '/media/ali/extradata/facial_landmark_ds/multi-pie/MPie_Labels/labels/all/'
    img_path_prefix = '/media/ali/extradata/facial_landmark_ds/multi-pie/'

    origin_number_of_all_sample = 2000
    origin_number_of_train_sample = 1950
    origin_number_of_evaluation_sample = 50
    augmentation_factor = 100


class W300Conf:
    tf_common = '/media/ali/data/test_common.tfrecords'
    tf_challenging = '/media/ali/data/test_challenging.tfrecords'
    tf_full = '/media/ali/data/test_full.tfrecords'

    img_path_prefix_common = '/media/ali/extradata/facial_landmark_ds/from_ibug/test_set/common/'
    img_path_prefix_challenging = '/media/ali/extradata/facial_landmark_ds/from_ibug/test_set/challenging/'
    img_path_prefix_full = '/media/ali/extradata/facial_landmark_ds/from_ibug/test_set/full/'

    number_of_all_sample_common = 554
    number_of_all_sample_challenging = 135
    number_of_all_sample_full = 689


class IbugConf:

    '''server_config'''
    Ibug_prefix_path = '/media/data3/ali/FL/ibug/'  # --> Zeus
    # _Ibug_prefix_path = '/media/data2/alip/fala/ibug/'  # --> Atlas

    img_path_prefix = Ibug_prefix_path + 'all/'
    rotated_img_path_prefix = Ibug_prefix_path + '0_rotated/'
    train_images_dir = Ibug_prefix_path + '1_train_images_pts_dir/'
    normalized_points_npy_dir = Ibug_prefix_path + '2_normalized_npy_dir/'
    normalized_pose_npy_dir = Ibug_prefix_path + '3_normalized_pose_npy_dir/'
    pose_npy_dir = Ibug_prefix_path + '4_pose_npy_dir/'
    tf_train_path = Ibug_prefix_path + 'train.tfrecords'
    tf_test_path = Ibug_prefix_path + 'test.tfrecords'
    tf_evaluation_path = Ibug_prefix_path + 'evaluation.tfrecords'

    # train_hm_dir = '/media/data2/alip/fala/ibug/train_hm_dir/'
    # train_hm_dir_85 = '/media/data2/alip/fala/ibug/train_hm_dir_85/'
    # train_hm_dir_90 = '/media/data2/alip/fala/ibug/train_hm_dir_90/'
    # train_hm_dir_97 = '/media/data2/alip/fala/ibug/train_hm_dir_97/'

    '''local'''
    # img_path_prefix = '/media/ali/data/train_set/'
    # rotated_img_path_prefix = '/media/ali/data/rotated/'
    # train_images_dir = '/media/ali/data/train_images_dir/'
    # normalized_points_npy_dir = '/media/ali/data/normalized_points_npy_dir/'
    # pose_npy_dir = '/media/ali/data/train_before_heatmap_pose/'
    # train_hm_dir = '/media/ali/data/train_before_heatmap_npy/'
    # train_hm_dir_85 = '/media/ali/data/train_hm_dir_85/'
    # train_hm_dir_90 = '/media/ali/data/train_hm_dir_90/'
    # train_hm_dir_97 = '/media/ali/data/train_hm_dir_97/'
    # tf_train_path = '/media/ali/data/train.tfrecords'
    # tf_test_path = '/media/ali/data/test.tfrecords'
    # tf_evaluation_path = '/media/ali/data/evaluation.tfrecords'


    # origin_number_of_all_sample = 3148  # afw, train_helen, train_lfpw
    # origin_number_of_train_sample = 2834  # 95 % for train
    # origin_number_of_evaluation_sample = 314  # 5% for evaluation

    origin_number_of_samples_after_rotate = 33672  # afw, train_helen, train_lfpw
    origin_number_of_all_sample = 6296   # afw, train_helen, train_lfpw
    origin_number_of_train_sample = origin_number_of_all_sample - origin_number_of_all_sample* 0.5  # 95 % for train
    origin_number_of_evaluation_sample = origin_number_of_all_sample - origin_number_of_train_sample # 5% for evaluation

    augmentation_factor = 4  # create . image from 1
    augmentation_factor_rotate = 20  # create . image from 1

    sum_of_train_samples = origin_number_of_train_sample * augmentation_factor
    sum_of_validation_samples = origin_number_of_evaluation_sample * augmentation_factor