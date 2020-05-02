# https://developers.google.com/machine-learning/practica/image-classification
"""" Extract features from pretrained cnn
"""
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing      import image

from timeit import default_timer as timer
import os, sys
import numpy as np

# Basics
#========
base_dir   = "/home/odul/Documents/ESIR2/ACI/Intel/data/intel-image-classification"
output_dir ="./xception_features/"
train_dir  = os.path.join(base_dir, 'seg_train/seg_train')
test_dir   = os.path.join(base_dir, 'seg_test/seg_test')
pred_dir   = os.path.join(base_dir, 'seg_pred/seg_pred')


# Train directories
#===================
train_building_dir = os.path.join(train_dir, 'buildings')
train_forest_dir   = os.path.join(train_dir, 'forest')
train_glacier_dir  = os.path.join(train_dir, 'glacier')
train_mountain_dir = os.path.join(train_dir, 'mountain')
train_sea_dir 	   = os.path.join(train_dir, 'sea')
train_street_dir   = os.path.join(train_dir, 'street')

# Test directories
#==================
test_building_dir = os.path.join(test_dir, 'buildings')
test_forest_dir   = os.path.join(test_dir, 'forest')
test_glacier_dir  = os.path.join(test_dir, 'glacier')
test_mountain_dir = os.path.join(test_dir, 'mountain')
test_sea_dir 	  = os.path.join(test_dir, 'sea')
test_street_dir   = os.path.join(test_dir, 'street')

# VGG16 config
#==============
#local_weights_file = "./xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
model_xception = Xception(include_top=False, weights='imagenet', pooling='max')
#model_xception.summary()


# Feature extractor
#===================
def extract_features(class_input_dir, class_id):

    class_xception_feature_list = []
    im_nb=0

    for fname in os.listdir(class_input_dir):
        print (str(im_nb+1) + " Compute descriptors for " + fname)
        img = image.load_img(os.path.join(class_input_dir, fname), target_size=(150, 150))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        xception_feature = model_xception.predict(img_data)
        xception_feature_np = np.array(xception_feature)
        class_xception_feature_list.append(xception_feature_np.flatten())

        im_nb=im_nb+1


    class_xception_feature_list_np = np.array(class_xception_feature_list)
    y_train=np.full(im_nb,class_id)

    return class_xception_feature_list_np, y_train



if __name__ == "__main__":

    start1 = timer()
    

    # # Extract features from TEST data set
    # #=====================================
    # building_xception_feature_list_np, building_y_train 	= extract_features(test_building_dir, 0)
    # forest_xception_feature_list_np,   forest_y_train   	= extract_features(test_forest_dir,   1)
    # glacier_xception_feature_list_np, 	glacier_y_train 	= extract_features(test_glacier_dir,  2)
    # mountain_xception_feature_list_np, mountain_y_train 	= extract_features(test_mountain_dir, 3)
    # sea_xception_feature_list_np, 		sea_y_train 		= extract_features(test_sea_dir, 	  4)
    # street_xception_feature_list_np, 	street_y_train 		= extract_features(test_street_dir,   5)

    # # concatenate extraction results
    # xception_feature_list_np = np.concatenate((building_xception_feature_list_np, forest_xception_feature_list_np, glacier_xception_feature_list_np, mountain_xception_feature_list_np, sea_xception_feature_list_np, street_xception_feature_list_np))
    # y_train = np.concatenate((building_y_train, forest_y_train, glacier_y_train, mountain_y_train, sea_y_train, street_y_train))

    # # save descriptors
    # np.save(output_dir + "xception_test_descriptors.npy", xception_feature_list_np)
    # np.save(output_dir + "xception_test_target.npy", y_train)



    # Extract features from TRAIN data set
    #=====================================
    building_xception_feature_list_np, building_y_train = extract_features(train_building_dir,0)
    forest_xception_feature_list_np, forest_y_train = extract_features(train_forest_dir,1)
    glacier_xception_feature_list_np, glacier_y_train = extract_features(train_glacier_dir,2)
    mountain_xception_feature_list_np, mountain_y_train = extract_features(train_mountain_dir,3)
    sea_xception_feature_list_np, sea_y_train = extract_features(train_sea_dir,4)
    street_xception_feature_list_np, street_y_train = extract_features(train_street_dir,5)

    # concatenate extraction results
    xception_feature_list_np = np.concatenate((building_xception_feature_list_np, forest_xception_feature_list_np, glacier_xception_feature_list_np, mountain_xception_feature_list_np, sea_xception_feature_list_np, street_xception_feature_list_np))
    y_train = np.concatenate((building_y_train, forest_y_train, glacier_y_train, mountain_y_train, sea_y_train, street_y_train))
    
    # save descriptors
    np.save(output_dir + "xception_train_descriptors.npy", xception_feature_list_np)
    np.save(output_dir + "xception_train_target.npy", y_train)


    # Extract features from PRED data set
    #=====================================
    pred_xception_feature_list_np, pred_y_train = extract_features(pred_dir,0)
    
    # concatenate extraction results
    xception_feature_list_np = np.concatenate((pred_xception_feature_list_np))
    #y_train = np.concatenate((pred_y_train))
    
    # save descriptors
    np.save(output_dir + "xception_pred_descriptors.npy", xception_feature_list_np)
    #np.save(output_dir + "xception_pred_target.npy", y_train)
    
    # End of extraction
    #===================

    
    end1 = timer()

    print("Train extraction Time: " + str(end1 - start1))

    model_xception.summary()
 
    sys.exit(0)

