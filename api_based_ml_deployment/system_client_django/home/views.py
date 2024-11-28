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
from .models import RiceDisease
import subprocess
import json

api_link = 'http://127.0.0.1/predict'
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
        
        ''' ---Preprocessing--- '''
        image = cv.imread(uploaded_image, cv.IMREAD_UNCHANGED)      # Load the image with 4 channels
        # If the image has 4 channels (RGBA), convert it to 3 channels (RGB)
        if image.shape[2] == 4:
            image = cv.cvtColor(image, cv.COLOR_RGBA2RGB)           # Extract the RGB channels (discard the Alpha channel)
        resized_image = cv.resize(image, image_size)                # Resize the image to the desired dimensions (e.g., 128x128)
        normalized_image = resized_image.astype('float32') / 255.0  # Convert the image to float32 and normalize it
        # input_image = normalized_image[np.newaxis, ...]             # Expand dimensions to match the expected input shape of the model
        cv.imwrite(uploaded_image, (normalized_image * 255).astype(np.uint8)) # Save the image here
        
        
        # Execute the curl command to send the image to the Flask API
        curl_command = [
            'curl',
            '-X', 'POST',
            api_link,
            '-F', f'file=@{uploaded_image}'
        ]
        result = subprocess.run(curl_command, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Curl command failed: {result.stderr}")
        # Parse the response
        predicted_infos = json.loads(result.stdout)[0]

        graphic_visualization = visualize_prediction(predicted_infos['rounded_predictions'])
        
        is_valid = False if (predicted_infos['predicted_class_acuracy'] / 100) < 0.75 else True
        
        # get some information from the database (RiceDisease)
        infos = RiceDisease.objects.get(class_label=predicted_infos['predicted_class_name'])
        
        context = {
            'image_name' : renamed_file,
            'predicted_class_name' : predicted_infos['predicted_class_name'],
            'predicted_class_acuracy' : predicted_infos['predicted_class_acuracy'],
            'graphic_visualization' : graphic_visualization,
            'predicted_information' : infos,
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
        
        # Execute the curl command to send the image to the Flask API
        curl_command = [
            'curl',
            '-X', 'POST',
            api_link,
            '-F', f'file=@{uploaded_image}'
        ]
        result = subprocess.run(curl_command, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Curl command failed: {result.stderr}")
        # Parse the response
        predicted_infos = json.loads(result.stdout)[0]
        
        graphic_visualization = visualize_prediction(predicted_infos['rounded_predictions'])
        
        is_valid = False if (predicted_infos['predicted_class_acuracy'] / 100) < 0.75 else True
        
        # get some information from the database (RiceDisease)
        infos = RiceDisease.objects.get(class_label=predicted_infos['predicted_class_name'])
        
        context = {
            'image_name' : renamed_file,
            'predicted_class_name' : predicted_infos['predicted_class_name'],
            'predicted_class_acuracy' : predicted_infos['predicted_class_acuracy'],
            'graphic_visualization' : graphic_visualization,
            'predicted_information' : infos,
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
    location = os.path.join(settings.MEDIA_ROOT, file_name)
    file = open(location, 'wb')
    file.write(imgdata) # save the picture
    file.close()
    return location
    
