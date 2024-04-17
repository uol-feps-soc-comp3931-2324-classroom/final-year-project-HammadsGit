from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
from keras.utils import load_img, img_to_array
import numpy as np
from flask import url_for

import cv2
import tensorflow as tf
from util import grad_cam, load_image

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras import backend as K


app = Flask(__name__)

def make_prediction(img_array):
    base_model = DenseNet121(weights='models/densenet.hdf5', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(14, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights('trained_model.h5')
    prediction = model.predict(img_array)
    return model, prediction

def overlay_heatmap(heatmap, image, alpha=0.5):
    """Overlay the heatmap on the original image."""
    # Heatmap needs to be rescaled to 0-255 and converted to color
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Combine the heatmap with the original image, using the specified alpha for the heatmap
    overlay_img = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
    return overlay_img

def prepare_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims / 255.


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file is provided
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Secure the filename
            filename = secure_filename(file.filename)
            file_path = 'static/images/' + filename
            file.save(file_path)
            
            # Preprocess the image
            img = load_img(file_path, target_size=(320, 320))  
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.
            image_url = url_for('static', filename='images/' + filename)

            prepare_imaged = prepare_image(file_path, target_size=(320, 320))


            # Make prediction
            model, prediction = make_prediction(img_array)
            
            print("Prediction:", prediction)
            print(filename)
            
            condition = ['Cardiomegaly', 
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
            
            predictions_with_conditions = [(condition, float(pred)) for condition, pred in zip(condition, prediction[0])]
            print(predictions_with_conditions)

            cam_images = []

            for idx in range(len(condition)):
                if (prediction[0][idx] > 0.50):
                    cam = grad_cam(model, img_array, idx, layer_name='bn')
                    original_img = cv2.imread(file_path)
                    original_img = cv2.resize(original_img, (cam.shape[1], cam.shape[0]))
                    superimposed_img = overlay_heatmap(cam, original_img)
                    output_path = 'static/heatmaps/' + filename[:-4] + "_" + condition[idx] + '_CAM.jpg'
                        # append the condition name and the path to the image
                    cam_images.append([condition[idx], output_path])
                    cv2.imwrite(output_path, superimposed_img)

            # Return results
            return render_template('results.html', prediction=prediction, condition=condition, predictions_with_conditions=predictions_with_conditions, 
                                   file_path=file_path, image_url=image_url, cam_images = cam_images)
            
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)