
import keras
import numpy as np

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import ImageDataGenerator
import sys
import pandas as pd


def loader(bs = 32):
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_data = '/data/celebA/CelebA/Anno/train_data.txt'
    train_data = pd.read_csv(train_data, delim_whitespace=True, header = None)
    train_data.columns = ['filename', 'label', 'gender']
    train_data.columns = train_data.columns.str.strip()
    train_data['label'] = train_data['label'].astype(str)

    train_data_female = '/data/celebA/CelebA/Anno/train_data_female.txt'
    train_data_female = pd.read_csv(train_data_female, delim_whitespace=True, header = None)
    train_data_female.columns = ['filename', 'label', 'gender']
    train_data_female.columns = train_data_female.columns.str.strip()
    train_data_female['label'] = train_data_female['label'].astype(str)

    train_data_male = '/data/celebA/CelebA/Anno/train_data_male.txt'
    train_data_male = pd.read_csv(train_data_male, delim_whitespace=True, header = None)
    train_data_male.columns = ['filename', 'label', 'gender']
    train_data_male.columns = train_data_male.columns.str.strip()
    train_data_male['label'] = train_data_male['label'].astype(str)

    val_male_data = '/data/celebA/CelebA/Anno/val_data_male.txt'
    val_male_data = pd.read_csv(val_male_data, delim_whitespace=True, header = None)
    val_male_data.columns = ['filename', 'label', 'gender']
    val_male_data.columns = val_male_data.columns.str.strip()
    val_male_data['label'] = val_male_data['label'].astype(str)

    val_data = '/data/celebA/CelebA/Anno/val_data.txt'
    val_data = pd.read_csv(val_data, delim_whitespace=True, header = None)
    val_data.columns = ['filename', 'label', 'gender']
    val_data.columns = val_data.columns.str.strip()
    val_data['label'] = val_data['label'].astype(str)

    val_female_data = '/data/celebA/CelebA/Anno/val_data_female.txt'
    val_female_data = pd.read_csv(val_female_data, delim_whitespace=True, header = None)
    val_female_data.columns = ['filename', 'label', 'gender']
    val_female_data.columns = val_female_data.columns.str.strip()
    val_female_data['label'] = val_female_data['label'].astype(str)

    test_data_male = '/data/celebA/CelebA/Anno/test_data_male.txt'
    test_data_male = pd.read_csv(test_data_male, delim_whitespace=True, header = None)
    test_data_male.columns = ['filename', 'label', 'gender']
    test_data_male.columns = test_data_male.columns.str.strip()
    test_data_male['label'] = test_data_male['label'].astype(str)

    test_data_female = '/data/celebA/CelebA/Anno/test_data_female.txt'
    test_data_female = pd.read_csv(test_data_female, delim_whitespace=True, header = None)
    test_data_female.columns = ['filename', 'label', 'gender']
    test_data_female.columns = test_data_female.columns.str.strip()
    test_data_female['label'] = test_data_female['label'].astype(str)


    train_generator = train_datagen.flow_from_dataframe(train_data, x_col = 'filename', y_col=['label', 'gender'], \
                                                        class_mode = 'multi_output',target_size=(128, 128), \
                                                        batch_size = bs, shuffle = True)
    train_male_generator = train_datagen.flow_from_dataframe(train_data_male, x_col = 'filename', y_col=['label', 'gender'], \
                                                        class_mode = 'multi_output',target_size=(128, 128), \
                                                        batch_size = bs, shuffle = True)
    train_female_generator = train_datagen.flow_from_dataframe(train_data_female, x_col = 'filename', y_col=['label', 'gender'], \
                                                        class_mode = 'multi_output',target_size=(128, 128), \
                                                        batch_size = bs, shuffle = True)
    val_generator = val_datagen.flow_from_dataframe(val_data, x_col = 'filename', y_col=['label', 'gender'], \
                                                    class_mode = 'multi_output',target_size=(128, 128), \
                                                    batch_size = bs, shuffle = True)
    val_male_generator = val_datagen.flow_from_dataframe(val_male_data, x_col = 'filename', y_col=['label', 'gender'], \
                                                    class_mode = 'multi_output',target_size=(128, 128), \
                                                    batch_size = bs, shuffle = True)
    val_female_generator = val_datagen.flow_from_dataframe(val_female_data, x_col = 'filename', y_col=['label', 'gender'], \
                                                    class_mode = 'multi_output',target_size=(128, 128), \
                                                    batch_size = bs, shuffle = True)
    male_generator = train_datagen.flow_from_dataframe(test_data_male, x_col = 'filename', y_col=['label', 'gender'], \
                                                        class_mode = 'multi_output',target_size=(128, 128), \
                                                        batch_size = bs, shuffle = True)
    female_generator = val_datagen.flow_from_dataframe(test_data_female, x_col = 'filename', y_col=['label', 'gender'], \
                                                    class_mode = 'multi_output',target_size=(128, 128), \
                                                    batch_size = bs, shuffle = True)

    return train_generator, train_male_generator, train_female_generator, val_generator, val_male_generator, val_female_generator, male_generator, female_generator