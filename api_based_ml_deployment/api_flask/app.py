from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import tensorflow as tf
import numpy as np
import cv2 as cv
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

ml_model = 'ml_model\Rice_ResNet50_50epochs.h5'
image_size = (224, 224)
class_names = [
        'Bacterial_Blight',
        'Blast',
        'Brown_Spot',
        'False_Smut',
        'Healthy',
        'Healthy_Flowers',
        'Hispa',
        'Leaf_Smut',
        'Sheath_Blight',
        'Tungro',
        'Unhealthy_Flowers'
    ]

@app.errorhandler(404)
def page_not_found(e):
    return "<p>404!</p>"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        try:
            # Read the image
            img = Image.open(BytesIO(file.read()))
            imageExpandedArrayShape = load_image(img)
            prediction_results = classify(imageExpandedArrayShape)
            # Prepare the response
            response = [prediction_results]
            return jsonify(response)
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        

def load_image(img):
    # Preprocess the image
    resized_img = img.resize(image_size)
    # Convert image to numpy array
    numpydata = np.asarray(resized_img)
    input_arr = np.array([numpydata])  # Convert single image to a batch.
    return input_arr
    # # ---Using OpenCV
    '''img_array = cv.imread(fileurl) # read the image
    img_array = cv.cvtColor(img_array, cv.COLOR_BGR2RGB) # convert to RGB
    new_img = cv.resize(img_array, image_size) # resize the image
    imageExpandedArrayShape = np.expand_dims(new_img,axis=0)
    return imageExpandedArrayShape'''
    # # ---Using Tensorflow
    '''img = load_img(fileurl, target_size=image_size) # Load and preprocess the image
    img_array = img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
    return img_array'''
    # # Note: it is advisable to use what is being used during the Model training/development (Opencv or Tensorflow)

def classify(imageExpandedArrayShape):
    # load the model
    model = tf.keras.models.load_model(ml_model)
    # class labels
    LABELS = class_names
    # PREDICTION
    predictions = model.predict(imageExpandedArrayShape) # all predictions
    rounded_predictions = list(np.array(np.around(predictions[0] * 100,2)))
    # predicted name
    predicted_class_name = LABELS[np.argmax(predictions)] # get the name/class label
    predicted_class_acuracy = float(np.array(predictions[0])[np.argmax(predictions)]) * 100
    predicted_infos = {
        'predictions' : [pred.tolist() for pred in predictions],
        'rounded_predictions' : [float(val) for val in rounded_predictions],
        'predicted_class_name' : predicted_class_name,
        'predicted_class_acuracy' : predicted_class_acuracy,
    }
    return predicted_infos


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
