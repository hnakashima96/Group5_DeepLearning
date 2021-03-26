import os, os.path
import shutil 
import glob

folder_path_train = r'C:\Users\hirom\Documents\GitHub\Group5_DeepLearning\CNN_example\data\train'
folder_path_test = r'C:\Users\hirom\Documents\GitHub\Group5_DeepLearning\CNN_example\data\test'

def split_classes(folder_path):
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for image in images:
        folder_name = image.split('.')[0]

        new_path = os.path.join(folder_path, folder_name)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        old_image_path = os.path.join(folder_path, image)
        new_image_path = os.path.join(new_path, image)
        shutil.move(old_image_path, new_image_path)

#split each dataset 
split_classes(folder_path_train)
split_classes(folder_path_test)