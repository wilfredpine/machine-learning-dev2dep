from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import random
import string
import tensorflow as tf
from django.conf import settings
import os
import numpy as np
import cv2 as cv
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pathlib
import matplotlib.pyplot as plt
import io
import base64
import re

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

# Create your views here.

def capture(request):
    context = {}
    if request.method == 'POST':
        image = request.POST['image_data']
        extension = '.' + image.split(',')[0].split(';')[0].split('/')[1]
        renamed_file = rename_file(extension)
        uploaded_image = upload_photo_base64(image, renamed_file)
        
        loaded_image_converted = load_image(uploaded_image)
        predicted_infos = classify(loaded_image_converted)
        
        graphic_visualization = visualize_prediction(predicted_infos['rounded_predictions'])
        
        is_valid = False if (float(predicted_infos['predicted_class_acuracy'].rstrip('%')) / 100) < 0.75 else True
        
        context = {
            'image_name' : renamed_file,
            'predicted_class_name' : predicted_infos['predicted_class_name'],
            'predicted_class_acuracy' : predicted_infos['predicted_class_acuracy'],
            'graphic_visualization' : graphic_visualization,
            'is_valid' : is_valid
            }
        
    return render(request, 'index.html', context)

def upload(request):
    context = {}
    if request.method == 'POST' and request.FILES:
        image = request.FILES['file']
        extension = pathlib.Path(image.name).suffix
        
        renamed_file = rename_file(extension)
        uploaded_image = upload_file(image, renamed_file)
        
        loaded_image_converted = load_image(uploaded_image)
        predicted_infos = classify(loaded_image_converted)
        
        graphic_visualization = visualize_prediction(predicted_infos['rounded_predictions'])
        
        is_valid = False if (float(predicted_infos['predicted_class_acuracy'].rstrip('%')) / 100) < 0.75 else True
        
        context = {
            'image_name' : renamed_file,
            'predicted_class_name' : predicted_infos['predicted_class_name'],
            'predicted_class_acuracy' : predicted_infos['predicted_class_acuracy'],
            'graphic_visualization' : graphic_visualization,
            'is_valid' : is_valid
            }
        
    return render(request, 'upload.html', context)
    
def rename_file(extension):
    rand_name = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(20))
    return rand_name + extension

def upload_file(file, file_name):
    fs = FileSystemStorage() # create a new instance of FileSystemStorage
    file_path = os.path.join(settings.MEDIA_ROOT, file_name)
    fs.save(file_path, file)
    return file_path

def load_image(fileurl):
    # # ---Using OpenCV
    img_array = cv.imread(fileurl) # read the image
    img_array = cv.cvtColor(img_array, cv.COLOR_BGR2RGB) # convert to RGB
    new_img = cv.resize(img_array, image_size) # resize the image
    imageExpandedArrayShape = np.expand_dims(new_img,axis=0)
    return imageExpandedArrayShape
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
    predicted_class_acuracy = ("{:.2%}".format(np.array(predictions[0])[np.argmax(predictions)]))
    
    predicted_infos = {
        'predictions' : predictions,
        'rounded_predictions' : rounded_predictions,
        'predicted_class_name' : predicted_class_name,
        'predicted_class_acuracy' : predicted_class_acuracy,
    }
    
    return predicted_infos

def visualize_prediction(rounded_predictions):
    labels = class_names
    values = rounded_predictions
    x = np.arange(len(labels))  # the label locations
    width = 0.80                # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, values, width, label='Accuracy')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores of Prediction by Classes')
    ax.set_xticks(x, labels)
    plt.xticks(rotation=65)
    ax.legend()
    ax.bar_label(rects1, padding=2)
    fig.tight_layout()
    # plt.show()
    # Image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    # image_png = base64.b64encode(buffer.read())
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic_visualization = graphic.decode('utf-8')
    return graphic_visualization

def upload_photo_base64(file, file_name):
    strs = re.match('^data:image/(jpeg|png|gif);base64,', file) # Regular match to remove the previous file type
    image = file.replace(strs.group(), '')
    imgdata = base64.b64decode(image) #Convert to image object
    location = 'media/' + file_name
    file = open(location, 'wb')
    file.write(imgdata) # save the picture
    file.close()
    return location
    
