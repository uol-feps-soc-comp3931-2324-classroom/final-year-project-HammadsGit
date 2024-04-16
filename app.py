from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
from keras.utils import load_img, img_to_array
import numpy as np
from flask import url_for

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

app = Flask(__name__)


base_model = DenseNet121(weights='models/densenet.hdf5', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(14, activation="sigmoid")(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights('trained_model.h5')



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
            img = load_img(file_path, target_size=(224, 224))  # Use the correct target size for your model
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Scale image
            image_url = url_for('static', filename='images/' + filename)

            # Make prediction
            prediction = model.predict(img_array)
            
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


            # Return results
            return render_template('results.html', prediction=prediction, condition=condition, predictions_with_conditions=predictions_with_conditions, file_path=file_path, image_url=image_url)
            
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
