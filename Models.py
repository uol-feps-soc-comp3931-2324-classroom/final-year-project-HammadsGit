#!/usr/bin/env python
# coding: utf-8

# In[5]:


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



import utils
from sklearn import utils
from public_test import *
from testutils import *

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



# In[6]:


IMAGE_DIR = "archive/images"


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




# In[7]:


train_df = pd.read_csv("archive/train_data.csv")
test_df = pd.read_csv("archive/test_data.csv")
valid_df = pd.read_csv("archive/val_data.csv")


# In[8]:


def detect_patient_overlap(dataframe1, dataframe2, patient_column):

    # Extracting patient IDs from both dataframes
    patients_df1 = set(dataframe1[patient_column].unique())
    patients_df2 = set(dataframe2[patient_column].unique())
    
    # Finding the intersection of patient IDs between the two sets
    common_patients = patients_df1 & patients_df2
    
    # Checking if there are any common patients
    has_overlap = len(common_patients) > 0
    
    return has_overlap


# In[9]:


print("Overlap detected between training and validation sets: {}".format(detect_patient_overlap(train_df, valid_df, 'PatientId')))
print("Overlap detected between training and test sets: {}".format(detect_patient_overlap(train_df, test_df, 'PatientId')))
print("Overlap detected between validation and test sets: {}".format(detect_patient_overlap(valid_df, test_df, 'PatientId')))


# In[10]:


def get_train_generator(dataframe, images_folder, feature_column, labels_columns, mix=True, size_per_batch=32, random_seed=1, width=320, height=320):

    print("Preparing training data loader...") 
    # Standardize images
    img_data_preprocessor = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True)
    
    # Generate data batch with defined size and image dimensions
    data_loader = img_data_preprocessor.flow_from_dataframe(
            dataframe=dataframe,
            directory=images_folder,
            x_col=feature_column,
            y_col=labels_columns,
            class_mode="raw",
            batch_size=size_per_batch,
            shuffle=mix,
            seed=random_seed,
            target_size=(width,height))
    
    return data_loader


# In[11]:


def get_test_and_valid_generator(validation_data, test_data, training_data, images_path, feature_column, target_columns, sampling_size=100, size_per_batch=32, random_seed=1, img_width=320, img_height=320):

    print("Preparing validation and test data loaders...")
    # Create a raw data generator for sampling
    initial_data_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=training_data,
        directory=images_path,
        x_col="Image",
        y_col=target_columns,
        class_mode="raw",
        batch_size=sampling_size,
        shuffle=True,
        target_size=(img_width, img_height))
    
    # Extract a data sample for normalization purposes
    data_batch = initial_data_generator.next()
    sample_data = data_batch[0]

    # Configure generator for standardized image inputs
    standardized_image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)
    
    # Standardize based on the training data sample
    standardized_image_generator.fit(sample_data)

    # Setup validation data loader
    validation_loader = standardized_image_generator.flow_from_dataframe(
            dataframe=validation_data,
            directory=images_path,
            x_col=feature_column,
            y_col=target_columns,
            class_mode="raw",
            batch_size=size_per_batch,
            shuffle=False,
            seed=random_seed,
            target_size=(img_width,img_height))

    # Setup test data loader
    test_loader = standardized_image_generator.flow_from_dataframe(
            dataframe=test_data,
            directory=images_path,
            x_col=feature_column,
            y_col=target_columns,
            class_mode="raw",
            batch_size=size_per_batch,
            shuffle=False,
            seed=random_seed,
            target_size=(img_width,img_height))
    
    return validation_loader, test_loader


# In[12]:


train_generator = get_train_generator(train_df, IMAGE_DIR, "Image", labels)


# In[13]:


valid_generator, test_generator= get_test_and_valid_generator(valid_df, test_df, train_df, IMAGE_DIR, "Image", labels)


# In[66]:


plt.xticks(rotation=90)
plt.bar(x=labels, height=np.mean(train_generator.labels, axis=0))
plt.title("Frequency of Each Class")
plt.show()


# In[14]:


def compute_class_freqs(label_array):
    # Determine the total number of entries
    total_entries = label_array.shape[0]
    
    # Calculate frequencies of the positive labels
    freq_of_positives = np.sum(label_array, axis=0) / total_entries
    # Derive frequencies of the negative labels
    freq_of_negatives = 1 - freq_of_positives

    return freq_of_positives, freq_of_negatives


# In[15]:


freq_pos, freq_neg = compute_class_freqs(train_generator.labels)


# In[16]:


data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})

# New DataFrame for negative labels, with red color
new_data = pd.DataFrame({"Class": labels, "Label": "Negative", "Value": freq_neg, "Color": "red"})

# Concatenating the original and new data
data = pd.concat([data, new_data], ignore_index=True)

# Plotting
plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label", data=data)


# In[17]:


pos_weights = freq_neg
neg_weights = freq_pos

pos_contribution = freq_pos * pos_weights
neg_contribution = freq_neg * neg_weights

pos_contribution, neg_contribution


# In[18]:


# Creating DataFrame for positive contributions
data_pos = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})

# Creating DataFrame for negative contributions
data_neg = pd.DataFrame({"Class": labels, "Label": "Negative", "Value": neg_contribution})

# Concatenating the positive and negative data
data = pd.concat([data_pos, data_neg], ignore_index=True)

# Plotting
plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label", data=data)
plt.show()


# In[19]:


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):

    def weighted_loss(y_true, y_pred):
        print("y_true shape:", K.shape(y_true))  # Debug print
        print("y_pred shape:", K.shape(y_pred))  # Debug print
        

        # initialize loss to zero
        loss = 0.0
        
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class 
            loss_pos = -1. * K.mean(K.cast(pos_weights[i], 'float32') * K.cast(y_true[:, i], 'float32') * K.log(K.cast(y_pred[:, i], 'float32') + epsilon))
            
            loss_neg = -1 * K.mean(K.cast(neg_weights[i], "float32") * (1 - K.cast(y_true[:, i], 'float32')) * K.log(1 - K.cast(y_pred[:, i], 'float32') + epsilon))

            
            loss += loss_pos + loss_neg
        return loss
        
        
    
        ### END CODE HERE ###
    return weighted_loss


# In[20]:


from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
import datetime

# create the base pre-trained model
base_model = DenseNet121(weights='models/densenet.hdf5', include_top=False)

x = base_model.output
# x = tf.cast(x, tf.float32)
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)

# and a logistic layer
predictions = Dense(len(labels), activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))

# Callbacks
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='min', min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)


# In[88]:


history = model.fit(train_generator, 
                              validation_data=valid_generator,
                              steps_per_epoch=100, 
                              validation_steps=25, 
                              epochs = 10, callbacks=[checkpoint, tensorboard, reduce_lr, early_stopping])

plt.plot(history.history['loss'])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training Loss Curve")
model.save("trained_model.h5")
plt.show()


# In[119]:


# print loss value
plt.plot(history.history['val_loss'])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Value Loss Curve")

plt.show()


# In[24]:


model.load_weights("trained_model.h5")


# In[25]:


predicted_vals = model.predict_generator(test_generator, steps = len(test_generator))

print(predicted_vals[0])


# In[92]:


# code to predict a single file
from keras.utils import load_img, img_to_array
from keras.models import load_model
import numpy as np

model = load_model('trained_model.h5', custom_objects={'weighted_loss': get_weighted_loss(pos_weights, neg_weights)})

# Function to prepare the image
def prepare_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims / 255.

# Replace 'path/to/your/image' with the path to the image you want to predict on
img_path = 'archive/images/00000003_000.png'
prepared_img = prepare_image(img_path, target_size=(320, 320))

predictions = model.predict(prepared_img)
print(predictions)

# Assuming your model predicts probabilities for each class
# and you have a list of class labels
labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']


# Print predictions for each label
for label, prediction in zip(labels, predictions[0]):
    print(f"{label}: {prediction*100:.2f}%")


# In[32]:


# creating file for the predictions

filenames = test_generator.filenames
true_labels = test_generator.labels
predicted_vals = predicted_vals  # This assumes 'predicted_vals' is already defined and contains predictions from your model

# Convert true labels to a DataFrame
df_labels = pd.DataFrame(true_labels, columns=labels)

# Add the image filenames to the DataFrame
df_labels['Image'] = filenames


df_labels['Hello'] = 1

# Convert predicted probabilities to a DataFrame
df_predictions = pd.DataFrame(predicted_vals, columns=[f"{label}_pred" for label in labels])

# Concatenate the true labels and predicted probabilities DataFrames
df_combined = pd.concat([df_labels, df_predictions], axis=1)

# Ensure the filenames are the first column
df_combined = df_combined[['Image'] + [col for col in df_combined.columns if col != 'Image']]

# Save the combined DataFrame to a CSV file
df_combined.to_csv('test_pred.csv', index=False)


# In[33]:


import pandas as pd
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

test_results = pd.read_csv("test_pred.csv")
pred_labels = [l + "_pred" for l in labels]



# In[34]:


y = test_results[labels].values
pred = test_results[pred_labels].values


# In[37]:


import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

def get_curve(true_labels, predictions, labels_names, curve_type='roc'):
    fig, ax = plt.subplots(figsize=(7, 7))  # Use ax for plotting

    plot_funcs = {
        'roc': lambda y_true, y_pred: (
            roc_curve(y_true, y_pred),
            {'label': f"{labels_names[i]} AUC: {roc_auc_score(y_true, y_pred):.3f}"}
        ),
        'prc': lambda y_true, y_pred: (
            precision_recall_curve(y_true, y_pred),
            {'label': f"{labels_names[i]} Avg. Prec.: {average_precision_score(y_true, y_pred):.3f}", 'where': 'post'}
        )
    }
    
    for i in range(len(labels_names)):
        scores, plot_args = plot_funcs[curve_type](true_labels[:, i], predictions[:, i])
        if curve_type == 'roc':
            fpr, tpr, _ = scores
            ax.plot(fpr, tpr, **plot_args)
        else:
            precision, recall, _ = scores
            ax.step(recall, precision, **plot_args)
            
    # Adding 'Chance' line for ROC
    if curve_type == 'roc':
        ax.plot([0, 1], [0, 1], 'k--', label='Chance')
        
    ax.set_xlabel("False Positive Rate" if curve_type == 'roc' else 'Recall')
    ax.set_ylabel("True Positive Rate" if curve_type == 'roc' else 'Precision')
    ax.set_title(f"{curve_type.upper()} Curve")
    ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1), fancybox=True, ncol=1)
    plt.show()


# In[39]:


get_curve(y, pred, labels, curve_type='roc')


# In[96]:


import matplotlib.pyplot as plt
plt.xticks(rotation=90)
plt.bar(x = labels, height= y.sum(axis=0));


# In[40]:


def true_positives(y, pred, th=0.5):
    TP = 0
    thresholded_preds = pred >= th
    TP = np.sum((y == 1) & (thresholded_preds == 1))
    
    return TP

def true_negatives(y, pred, th=0.5):
    thresholded_preds = pred < th  # Prediction is negative if below threshold
    TN = np.sum((y == 0) & (thresholded_preds == True))  # Both ground truth and prediction are negative
    return TN


def false_positives(y, pred, th=0.5):
    thresholded_preds = pred >= th  # Prediction is positive if equal to or above threshold
    FP = np.sum((y == 0) & (thresholded_preds == True))  # Ground truth is negative but prediction is positive
    return FP


def false_negatives(y, pred, th=0.5):
    thresholded_preds = pred < th  # Prediction is negative if below threshold
    FN = np.sum((y == 1) & (thresholded_preds == True))  # Ground truth is positive but prediction is negative
    return FN

def get_accuracy(y, pred, th=0.5):
    accuracy = 0.0
    
    # Threshold predictions to get binary outcomes
    thresholded_preds = pred >= th
    
    # Calculate TP, FP, TN, FN
    TP = np.sum((y == 1) & (thresholded_preds == 1))
    FP = np.sum((y == 0) & (thresholded_preds == 1))
    TN = np.sum((y == 0) & (thresholded_preds == 0))
    FN = np.sum((y == 1) & (thresholded_preds == 0))

    # Compute accuracy using TP, FP, TN, FN
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    
    return accuracy

def get_prevalence(y):
    prevalence = 0.0
    
    prevalence = np.sum(y == 1) / len(y)
    
    return prevalence


# In[41]:


true_positives(y, pred), false_positives(y, pred), false_negatives(y, pred), true_negatives(y, pred)

accuracy = (true_positives(y, pred) + true_negatives(y, pred)) / (y.shape[0] * y.shape[1])

print(accuracy * 100)


# In[44]:


get_curve(y, pred, labels, curve_type='roc')


# In[55]:


from sklearn.calibration import calibration_curve
def plot_calibration_curve(y, pred):
    plt.figure(figsize=(20, 20))
    for i in range(len(labels)):
        plt.subplot(4, 4, i + 1)
        fraction_of_positives, mean_predicted_value = calibration_curve(y[:,i], pred[:,i], n_bins=20)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(mean_predicted_value, fraction_of_positives, marker='.')
        plt.xlabel("Predicted Value")
        plt.ylabel("Fraction of Positives")
        plt.title(labels[i])
    plt.tight_layout()
    plt.show()


# In[56]:


plot_calibration_curve(y, pred)


# In[52]:


from sklearn.linear_model import LogisticRegression as LR 


pred_calibrated = np.zeros_like(pred)

for i in range(len(labels)):
    lr = LR(solver='liblinear', max_iter=10000)
    lr.fit(pred[:, i].reshape(-1, 1), y[:, i])    
    pred_calibrated[:, i] = lr.predict_proba(pred[:, i].reshape(-1, 1))[:,1]
    


# In[53]:


plot_calibration_curve(y[:,], pred_calibrated)


# In[49]:


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate performance metrics based on binary classification outcomes.
    """
    TP = np.sum((y_true == 1) & (y_pred >= threshold))
    TN = np.sum((y_true == 0) & (y_pred < threshold))
    FP = np.sum((y_true == 0) & (y_pred >= threshold))
    FN = np.sum((y_true == 1) & (y_pred < threshold))

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Sensitivity = TP / (TP + FN) if TP + FN > 0 else 0  # Also known as Recall
    Specificity = TN / (TN + FP) if TN + FP > 0 else 0
    PPV = TP / (TP + FP) if TP + FP > 0 else 0  # Also known as Precision
    NPV = TN / (TN + FN) if TN + FN > 0 else 0
    Prevalence = np.sum(y_true == 1) / len(y_true)
    AUC = roc_auc_score(y_true, y_pred)
    F1 = f1_score(y_true, y_pred >= threshold)

    return TP, TN, FP, FN, Accuracy, Prevalence, Sensitivity, Specificity, PPV, NPV, AUC, F1

def get_performance_metrics(y, pred, class_labels, thresholds=[]):
    if len(thresholds) != len(class_labels):
        thresholds = [.5] * len(class_labels)
    
    metrics_dict = {"": class_labels}
    metrics_list = ["TP", "TN", "FP", "FN", "Accuracy", "Prevalence", "Sensitivity", "Specificity", "PPV", "NPV", "AUC", "F1", "Threshold"]
    
    for metric in metrics_list:
        metrics_dict[metric] = []
    
    for i, label in enumerate(class_labels):
        TP, TN, FP, FN, Accuracy, Prevalence, Sensitivity, Specificity, PPV, NPV, AUC, F1 = calculate_metrics(y[:, i], pred[:, i], thresholds[i])
        values = [TP, TN, FP, FN, Accuracy, Prevalence, Sensitivity, Specificity, PPV, NPV, AUC, F1, thresholds[i]]
        rounded_values = [round(value, 3) if isinstance(value, float) else value for value in values]
        for metric, value in zip(metrics_list, rounded_values):
            metrics_dict[metric].append(value)
    
    return pd.DataFrame(metrics_dict).set_index("")


# In[55]:


get_performance_metrics(y, pred, labels)

# get average performance metrics


