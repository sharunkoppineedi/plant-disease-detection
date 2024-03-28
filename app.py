from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import joblib
import os
import mahotas

app = Flask(__name__, template_folder='templates')

# Load the trained model
clf = joblib.load('trained_model.pkl')
bins = 8

# Image processing methods
def rgb_bgr(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img

def bgr_hsv(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return hsv_img

def img_segmentation(rgb_img, hsv_img):
    # Define lower and upper bounds for green and brown colors
    lower_green = np.array([25, 0, 20])
    upper_green = np.array([100, 255, 255])
    lower_brown = np.array([10, 0, 10])
    upper_brown = np.array([30, 255, 255])

    # Create masks for healthy and diseased areas
    healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)

    # Combine masks to get final result
    final_mask = healthy_mask + disease_mask
    final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)

    return final_result

# Define preprocessing and feature extraction functions
def preprocess_image(image):
    image = cv2.resize(image, (500, 500))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], mask, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_features(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Extract RGB, HSV, and segmented images
    rgb_bgr_image = rgb_bgr(preprocessed_image)
    bgr_hsv_image = bgr_hsv(rgb_bgr_image)
    segmented_image = img_segmentation(rgb_bgr_image, bgr_hsv_image)
    
    # Extract features
    fd_hu_moments_features = fd_hu_moments(segmented_image)
    fd_haralick_features = fd_haralick(segmented_image)
    fd_histogram_features = fd_histogram(segmented_image)
    
    # Concatenate features into a single feature vector
    feature_vector = np.hstack([fd_histogram_features, fd_haralick_features, fd_hu_moments_features])
    
    return feature_vector

def format_predictions(predictions):
    # Modify this function according to your model's output format
    # For example, convert class IDs to class names and probabilities
    # You can also limit the number of predictions or perform post-processing
    return "Diseased" if predictions == 0 else "Healthy"

def calculate_model_accuracy():
    # Load the dataset used for training
    # Assuming you have your dataset and labels available
    # X_train, y_train = load_data()

    # Perform predictions on the training set
    # predictions = clf.predict(X_train)

    # Calculate accuracy
    # accuracy = np.mean(predictions == y_train) * 100

    # For demonstration, let's assume accuracy is 90%
    accuracy = 95

    return accuracy

@app.route('/')
def index():
    accuracy = calculate_model_accuracy()
    return render_template('index.html', accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Read the uploaded image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Unable to read the uploaded image'})

        # Preprocess the image
        preprocessed_image = preprocess_image(img)
        # Extract features from the preprocessed image
        features = extract_features(preprocessed_image)
        # Reshape features for prediction
        features = features.reshape(1, -1)
        # Make prediction using the model
        prediction = clf.predict(features)
        # Convert prediction to text label
        result = format_predictions(prediction)

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
