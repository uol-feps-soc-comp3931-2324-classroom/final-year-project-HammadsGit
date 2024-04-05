import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

from keras.models import load_model

# 0.1.1
# pip uninstall tensorflow-macos
# pip uninstall tensorflow-metal
# python -m pip install tensorflow-macos==2.9
# pip install tensorflow-metal==0.5.0

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


train_df = pd.read_csv("train-small.csv")
valid_df = pd.read_csv("valid-small.csv")
test_df = pd.read_csv("test.csv")

train_df.head()

labels = ['Cardiomegaly',
          'Emphysema',
          'Effusion',
          'Hernia',
          'Infiltration',
          'Mass',
          'Nodule',
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening',
          'Pneumonia',
          'Fibrosis',
          'Edema',
          'Consolidation']


# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def check_for_leakage(df1, df2, patient_col):
    """
    Return True if there any patients are in both df1 and df2.

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dataframe describing second dataset
        patient_col (str): string name of column with patient IDs

    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    df1_patient_values = df1[patient_col].values
    df2_patient_values = df2[patient_col].values

    df1_patients_unique = set(df1_patient_values)
    df2_patients_unique = set(df2_patient_values)

    patients_in_both_groups = list(set(df1_patients_unique).intersection(set(df2_patients_unique)))
    n_overlap = len(patients_in_both_groups)

    # leakage contains true if there is patient overlap, otherwise false.
    leakage = False

    if (n_overlap > 0): leakage = True

    ### END CODE HERE ###

    return leakage

print("leakage between train and valid: {}".format(check_for_leakage(train_df, valid_df, 'PatientId')))
print("leakage between train and test: {}".format(check_for_leakage(train_df, test_df, 'PatientId')))
print("leakage between valid and test: {}".format(check_for_leakage(valid_df, test_df, 'PatientId')))

def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w=320, target_h=320):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.

    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """
    print("getting train generator...")
    # normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True)

    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
        dataframe=df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        target_size=(target_w, target_h))

    return generator



def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=8,
                                 seed=1, target_w=320, target_h=320):
    """
    Return generator for validation set and test set using
    normalization statistics from training set.

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.

    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df,
        directory=IMAGE_DIR,
        x_col="Image",
        y_col=labels,
        class_mode="raw",
        batch_size=sample_size,
        shuffle=True,
        target_size=(target_w, target_h))

    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)

    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
        dataframe=valid_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        target_size=(target_w, target_h))

    test_generator = image_generator.flow_from_dataframe(
        dataframe=test_df,
        directory=image_dir,
        x_col=x_col,
        y_col=y_cols,
        class_mode="raw",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        target_size=(target_w, target_h))
    return valid_generator, test_generator

IMAGE_DIR = "images-small/"
train_generator = get_train_generator(train_df, IMAGE_DIR, "Image", labels)
valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image", labels)

# valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image", labels)

# show image
# x, y = train_generator.__getitem__(0)
# plt.imshow(x[0]);

# show classes
# plt.xticks(rotation=90)
# plt.bar(x=labels, height=np.mean(train_generator.labels, axis=0))
# plt.title("Frequency of Each Class")
# plt.show()

def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # total number of patients (rows)
    N = len(labels)

    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies

    ### END CODE HERE ###
    return positive_frequencies, negative_frequencies

freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
freq_pos

# Create a DataFrame for positive values
data_pos = pd.DataFrame({
    "Class": labels,
    "Label": "Positive",
    "Value": freq_pos
})

# Create a DataFrame for negative values
data_neg = pd.DataFrame({
    "Class": labels,
    "Label": "Negative",
    "Value": freq_neg
})

# Combine the two DataFrames
data_combined = pd.concat([data_pos, data_neg], ignore_index=True)

# Plot using seaborn
# plt.figure(figsize=(10, 6))  # You can adjust the figure size
# sns.barplot(x="Class", y="Value", hue="Label", data=data_combined)
# plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
# plt.show()

pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights
neg_contribution = freq_neg * neg_weights

# Create a DataFrame to display the results

# # Create initial DataFrame
# data = pd.DataFrame({
#     "Class": labels,
#     "Label": "Positive",
#     "Value": pos_contribution
# })
#
# # Extend DataFrame with negative contributions
# neg_data = pd.DataFrame({
#     "Class": labels,
#     "Label": "Negative",
#     "Value": neg_contribution
# })
#
# # Combine positive and negative contributions
# data = pd.concat([data, neg_data], ignore_index=True)
#
# # Plotting
# plt.xticks(rotation=90)
# sns.barplot(x="Class", y="Value", hue="Label", data=data)
# plt.show()


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)

    Returns:
      weighted_loss (function): weighted loss function
    """

    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value.

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0

        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class
            loss_pos = -1. * K.mean(
                K.cast(pos_weights[i], 'float32') * K.cast(y_true[:, i], 'float32') * K.log(y_pred[:, i] + epsilon))
            loss_neg = -1 * K.mean(K.cast(neg_weights[i], "float32") * (1 - K.cast(y_true[:, i])) * K.log(
                1 - K.cast(y_pred[:, i]) + epsilon))

            loss += loss_pos + loss_neg
        return loss

        ### END CODE HERE ###

    return weighted_loss


base_model = DenseNet121(weights='densenet.hdf5', include_top=False)

x = base_model.output

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)

# and a logistic layer
predictions = Dense(len(labels), activation="sigmoid")(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss=get_weighted_loss(pos_weights, neg_weights))