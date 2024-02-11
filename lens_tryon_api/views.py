from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

import numpy as np
import mediapipe as mp
import cv2
import json
import base64

import os
LENS_IMAGES_BASE_URL = os.path.join(settings.BASE_DIR, 'lens_tryon_api', 'lens_images')

def detect_landmarks(face_img):    
    # Initialize Mediapipe face detection
    faceMesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5
    )

    # Processing facemesh on input image
    result = faceMesh.process(face_img)

    # Fetching landmark coordinates
    landmarks = None
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = []
            for landmark in face_landmarks.landmark:
                img_h, img_w, img_c = face_img.shape
                x, y = int(landmark.x * img_w), int(landmark.y * img_h)
                landmarks.append((x,y))
    
    return landmarks


@csrf_exempt
def apply_facemesh(request):
    if request.method == 'POST':
        try:
            # Read the image data from the request
            image_data = request.FILES['image'].read()

            # Convert the image data to a NumPy array and decode it
            img_np_array = np.frombuffer(image_data, dtype=np.uint8)
            face_img = cv2.imdecode(img_np_array, flags=cv2.IMREAD_COLOR)

            landmarks = detect_landmarks(face_img)

            result = dict()
            result['landmarks'] = landmarks

            # Return the result as a JSON response
            json_string = json.dumps(result)
            return HttpResponse(json_string)

        except Exception as e:
            return HttpResponse("Some Server Error Occured")
        
    if request.method == 'GET':
        return HttpResponse("Server Connection Established. Kindly send POST Request for prediction")

    return JsonResponse({'error': 'Invalid request method'}, status=400)


@csrf_exempt
def apply_lens(request):
    if request.method == 'POST':
        try:
            # Read the image data from the request
            image_data = request.FILES['image'].read()

            # Convert the image data to a NumPy array and decode it
            img_np_array = np.frombuffer(image_data, dtype=np.uint8)
            face_img = cv2.imdecode(img_np_array, flags=cv2.IMREAD_COLOR)

            # Get the JSON data
            json_data = json.loads(request.POST['data'])
            landmarks = json_data['face_landmarks']['landmarks']

            # Landmark 9 corresponds to point slightly above nose tip
            nose_tip = landmarks[9]
            nose_tip_x, nose_tip_y = nose_tip
            face_width = landmarks[264][0] - landmarks[34][0]
            
            # Fetching Lens Image
            lens_id = json_data['lens_id']
            lens_img_path = LENS_IMAGES_BASE_URL + f'/lens{lens_id}.png'
            lens_img = cv2.imread(lens_img_path, -1)

            # Resizing Lens to fit on face
            height, width, _ = lens_img.shape
            new_width = face_width
            new_height = int(round(height * new_width / width))
            lens_img = cv2.resize(lens_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Setting offset value of coordinates for where to place the image
            offset_x = int((face_img.shape[1] - lens_img.shape[1]) / 2)
            offset_y = nose_tip_y

            # Overlay lens image on face image
            face_img_RGBA = cv2.cvtColor(face_img, cv2.COLOR_RGB2RGBA)
            h, w, _ = lens_img.shape
            for i in range(h):
                for j in range(w):
                    if lens_img[i][j][3] != 0: # Check if the alpha channel is not transparent
                        face_img_RGBA[offset_y + i, offset_x + j] = lens_img[i, j]


            # Encode the final image to base64
            _, img_encoded = cv2.imencode('.png', face_img_RGBA)
            res_img = base64.b64encode(img_encoded).decode()

            # Return the base64 encoded image as a JSON response
            return JsonResponse({'image': res_img})
            
        except Exception as e:
            return HttpResponse("Some Server Error Occured")
        
    if request.method == 'GET':
        return HttpResponse("Server Connection Established. Kindly send POST Request for prediction")

    return JsonResponse({'error': 'Invalid request method'}, status=400)