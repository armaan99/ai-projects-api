from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

import tensorflow as tf
import numpy as np
import cv2

# Load your pre-trained model
from django.conf import settings
import os
model_path = os.path.join(settings.BASE_DIR, 'hdr_api', 'HDR_model')
model = tf.keras.models.load_model(model_path)

@csrf_exempt
def hdr_predict(request):
    if request.method == 'POST':
        try:
            # Read the image data from the request
            image_data = request.FILES['image'].read()

            # Convert the image data to a NumPy array and decode it
            img_np_array = np.frombuffer(image_data, dtype=np.uint8)
            img = cv2.imdecode(img_np_array, cv2.IMREAD_GRAYSCALE)
            
            # Altering Black <-> White colors
            img = 255 - img

            # Resizing image to 28x28 pixels
            rescaled_img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

            # Normalizing the data
            image = rescaled_img / 255.0

            # Reshape the image to match the model's input shape
            image_reshaped = image.reshape(1, 28, 28)

            # Predicting the number
            prediction = model.predict(image_reshaped).argmax()

            # Return the result as a JSON response
            return HttpResponse(prediction)

        except Exception as e:
            return HttpResponse("Some Server Error Occured")
    return JsonResponse({'error': 'Invalid request method'}, status=400)