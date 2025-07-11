
from google.colab import drive
drive.mount('/content/drive')

import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern

def load_images_from_folder(folder_path, image_size=(128, 128)):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            img = cv2.resize(img, image_size)  # Resize to a consistent size
            img = img / 255.0  # Normalize to range 0-1
            images.append(img)
    return np.array(images)

# Load train, validation, and test images
train_images = load_images_from_folder('/content/drive/MyDrive/extracted_images_train')
val_images = load_images_from_folder('/content/drive/MyDrive/extracted_images_valid')
test_images = load_images_from_folder('/content/drive/MyDrive/extracted_images_test')

import os
import cv2
import numpy as np

# Function to load images from a folder
def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

# Function to calculate the area of food waste in an image
def calculate_food_waste_area(image, lower_color, upper_color):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask based on the specified color range
    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Calculate the area of the mask (number of non-zero pixels)
    waste_area = cv2.countNonZero(mask)

    # Total area of the image
    total_area = image.shape[0] * image.shape[1]

    # Percentage of food waste area
    waste_percentage = (waste_area / total_area) * 100
    return waste_area, waste_percentage

# Load train and test images
train_images, train_filenames = load_images_from_folder('/content/drive/MyDrive/extracted_images_train')
test_images, test_filenames = load_images_from_folder('/content/drive/MyDrive/extracted_images_test')

# Define color range for food segmentation in HSV (adjust as needed)
lower_color = np.array([20, 50, 50])  # Example: Lower bound of HSV
upper_color = np.array([35, 255, 255])  # Example: Upper bound of HSV

# Calculate and print food waste area for train images
print("Train Dataset:")
for img, filename in zip(train_images, train_filenames):
    waste_area, waste_percentage = calculate_food_waste_area(img, lower_color, upper_color)
    print(f"{filename}: Waste Area = {waste_area} pixels, Waste Percentage = {waste_percentage:.2f}%")

# Calculate and print food waste area for test images
print("\nTest Dataset:")
for img, filename in zip(test_images, test_filenames):
    waste_area, waste_percentage = calculate_food_waste_area(img, lower_color, upper_color)
    print(f"{filename}: Waste Area = {waste_area} pixels, Waste Percentage = {waste_percentage:.2f}%")

import os
import cv2
import numpy as np
import csv

# Function to load images from a folder
def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

# Function to calculate the area of food waste in an image
def calculate_food_waste_area(image, lower_color, upper_color):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask based on the specified color range
    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Calculate the area of the mask (number of non-zero pixels)
    waste_area = cv2.countNonZero(mask)

    # Total area of the image
    total_area = image.shape[0] * image.shape[1]

    # Percentage of food waste area
    waste_percentage = (waste_area / total_area) * 100
    return waste_area, waste_percentage

# Save the results to a CSV file
def save_to_csv(file_path, data, headers):
    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    # Open the file in append mode if it exists, else create it
    with open(file_path, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write the header if the file doesn't exist
            writer.writerow(headers)
        # Write the data
        writer.writerows(data)

# Paths for train and test CSV files
train_csv_path = 'C:\\Users\\hp\\Downloads\\train_features.csv'
test_csv_path = 'C:\\Users\\hp\\Downloads\\test_features.csv'

# Load train and test images
train_images, train_filenames = load_images_from_folder('/content/drive/MyDrive/extracted_images_train')
test_images, test_filenames = load_images_from_folder('/content/drive/MyDrive/extracted_images_test')

# Define color range for food segmentation in HSV (adjust as needed)
lower_color = np.array([20, 50, 50])  # Example: Lower bound of HSV
upper_color = np.array([35, 255, 255])  # Example: Upper bound of HSV

# Headers for CSV files
headers = ["Filename", "Waste Area (pixels)", "Waste Percentage (%)"]

# Calculate and save food waste data for train dataset
train_data = []
for img, filename in zip(train_images, train_filenames):
    waste_area, waste_percentage = calculate_food_waste_area(img, lower_color, upper_color)
    train_data.append([filename, waste_area, waste_percentage])

# Save to train CSV file
save_to_csv(train_csv_path, train_data, headers)

# Calculate and save food waste data for test dataset
test_data = []
for img, filename in zip(test_images, test_filenames):
    waste_area, waste_percentage = calculate_food_waste_area(img, lower_color, upper_color)
    test_data.append([filename, waste_area, waste_percentage])

# Save to test CSV file
save_to_csv(test_csv_path, test_data, headers)

import pandas as pd

# File paths
train_csv_path = 'C:\\Users\\hp\\Downloads\\train_features.csv'
test_csv_path = 'C:\\Users\\hp\\Downloads\\test_features.csv'

# Load the CSV files and print the first few rows
print("Train Dataset Head:")
train_df = pd.read_csv(train_csv_path)
print(train_df.head())

print("\nTest Dataset Head:")
test_df = pd.read_csv(test_csv_path)
print(test_df.head())

import os
import cv2
import numpy as np
import csv
from skimage.feature import local_binary_pattern
import pandas as pd

# Function to load images from a folder
def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

# Function to extract texture features from an image
def extract_texture_features(image, radius=1):
    n_points = 8 * radius
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# Function to calculate the texture area and percentage
def calculate_texture_area(image, radius=1):
    n_points = 8 * radius
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")

    # Create a binary mask based on the LBP values (example threshold, adjust as needed)
    mask = (lbp > 0).astype(np.uint8) * 255

    # Calculate the area of the mask (number of non-zero pixels)
    texture_area = cv2.countNonZero(mask)

    # Total area of the image
    total_area = image.shape[0] * image.shape[1]

    # Percentage of texture area
    texture_percentage = (texture_area / total_area) * 100
    return texture_area, texture_percentage

# Save the results to a CSV file
def append_to_csv(file_path, data):
    df = pd.read_csv(file_path)  # Read the existing CSV file
    df["Texture Area (pixels)"] = data[0]  # Append the texture area column
    df["Texture Percentage (%)"] = data[1]  # Append the texture percentage column
    df.to_csv(file_path, index=False)  # Save the updated CSV

# Paths for train and test CSV files
train_csv_path = 'C:\\Users\\hp\\Downloads\\train_features.csv'
test_csv_path = 'C:\\Users\\hp\\Downloads\\test_features.csv'

# Load train and test images
train_images, train_filenames = load_images_from_folder('/content/drive/MyDrive/extracted_images_train')
test_images, test_filenames = load_images_from_folder('/content/drive/MyDrive/extracted_images_test')

# Initialize lists for storing texture area and percentage
train_texture_area = []
train_texture_percentage = []

# Calculate and append texture data for train dataset
for img, filename in zip(train_images, train_filenames):
    texture_area, texture_percentage = calculate_texture_area(img)
    train_texture_area.append(texture_area)
    train_texture_percentage.append(texture_percentage)

# Append the results to the train CSV file
append_to_csv(train_csv_path, (train_texture_area, train_texture_percentage))

# Initialize lists for storing texture area and percentage
test_texture_area = []
test_texture_percentage = []

# Calculate and append texture data for test dataset
for img, filename in zip(test_images, test_filenames):
    texture_area, texture_percentage = calculate_texture_area(img)
    test_texture_area.append(texture_area)
    test_texture_percentage.append(texture_percentage)

# Append the results to the test CSV file
append_to_csv(test_csv_path, (test_texture_area, test_texture_percentage))

print("Texture area and percentage data appended to CSV files.")

import pandas as pd

# File paths
train_csv_path = 'C:\\Users\\hp\\Downloads\\train_features.csv'
test_csv_path = 'C:\\Users\\hp\\Downloads\\test_features.csv'

# Load the CSV files and print the first few rows
print("Train Dataset Head:")
train_df = pd.read_csv(train_csv_path)
print(train_df.head())

print("\nTest Dataset Head:")
test_df = pd.read_csv(test_csv_path)
print(test_df.head())

import os
import cv2
import numpy as np
import pandas as pd

# Function to load images from a folder
def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

# Function to extract shape features from an image
def extract_shape_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        return np.array([area, perimeter])
    else:
        return np.array([0, 0])

# Function to calculate the shape area and percentage
def calculate_shape_area_and_percentage(image):
    shape_area, _ = extract_shape_features(image)  # Get area and perimeter (perimeter is unused here)
    total_area = image.shape[0] * image.shape[1]
    shape_percentage = (shape_area / total_area) * 100
    return shape_area, shape_percentage

# Save the results to a CSV file
def append_to_csv(file_path, data):
    df = pd.read_csv(file_path)  # Read the existing CSV file
    df["Shape Area (pixels)"] = data[0]  # Append the shape area column
    df["Shape Percentage (%)"] = data[1]  # Append the shape percentage column
    df.to_csv(file_path, index=False)  # Save the updated CSV

# Paths for train and test CSV files
train_csv_path = 'C:\\Users\\hp\\Downloads\\train_features.csv'
test_csv_path = 'C:\\Users\\hp\\Downloads\\test_features.csv'

# Load train and test images
train_images, train_filenames = load_images_from_folder('/content/drive/MyDrive/extracted_images_train')
test_images, test_filenames = load_images_from_folder('/content/drive/MyDrive/extracted_images_test')

# Initialize lists for storing shape area and percentage
train_shape_area = []
train_shape_percentage = []

# Calculate and append shape data for train dataset
for img, filename in zip(train_images, train_filenames):
    shape_area, shape_percentage = calculate_shape_area_and_percentage(img)
    train_shape_area.append(shape_area)
    train_shape_percentage.append(shape_percentage)

# Append the results to the train CSV file
append_to_csv(train_csv_path, (train_shape_area, train_shape_percentage))

# Initialize lists for storing shape area and percentage
test_shape_area = []
test_shape_percentage = []

# Calculate and append shape data for test dataset
for img, filename in zip(test_images, test_filenames):
    shape_area, shape_percentage = calculate_shape_area_and_percentage(img)
    test_shape_area.append(shape_area)
    test_shape_percentage.append(shape_percentage)

# Append the results to the test CSV file
append_to_csv(test_csv_path, (test_shape_area, test_shape_percentage))

print("Shape area and percentage data appended to CSV files.")

import pandas as pd

# File paths
train_csv_path = 'C:\\Users\\hp\\Downloads\\train_features.csv'
test_csv_path = 'C:\\Users\\hp\\Downloads\\test_features.csv'

# Load the CSV files and print the first few rows
print("Train Dataset Head:")
train_df = pd.read_csv(train_csv_path)
print(train_df.head())

print("\nTest Dataset Head:")
test_df = pd.read_csv(test_csv_path)
print(test_df.head())

import pandas as pd
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Load images from folder
def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in listdir(folder_path):
        img_path = join(folder_path, filename)
        if isfile(img_path):
            img = cv2.imread(img_path)
            images.append(img)
            filenames.append(filename)
    return images, filenames

# Function to calculate the total area of food in an image
def calculate_food_area(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour by area
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)  # Area of the food object
        return area
    else:
        return 0

# Function to calculate the mean area (assuming we have areas of multiple contours)
def calculate_mean_area(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Calculate the area of each contour
        areas = [cv2.contourArea(cnt) for cnt in contours]
        mean_area = np.mean(areas)  # Mean of the areas of all contours
        return mean_area
    else:
        return 0

# Load the train and test datasets
train_df = pd.read_csv('C:\\Users\\hp\\Downloads\\train_features.csv')
test_df = pd.read_csv('C:\\Users\\hp\\Downloads\\test_features.csv')

# Load the images
train_images, train_filenames = load_images_from_folder('/content/drive/MyDrive/extracted_images_train')
test_images, test_filenames = load_images_from_folder('/content/drive/MyDrive/extracted_images_test')

# Initialize lists for storing total and mean areas of food for each image
train_total_areas = []
train_mean_areas = []
test_total_areas = []
test_mean_areas = []

# Calculate the total and mean area of food for each image in the train set
for img in train_images:
    total_area = calculate_food_area(img)
    mean_area = calculate_mean_area(img)
    train_total_areas.append(total_area)
    train_mean_areas.append(mean_area)

# Calculate the total and mean area of food for each image in the test set
for img in test_images:
    total_area = calculate_food_area(img)
    mean_area = calculate_mean_area(img)
    test_total_areas.append(total_area)
    test_mean_areas.append(mean_area)

# Append the total area and mean area to the DataFrame for train and test datasets
train_df['Total Area (pixels)'] = train_total_areas
train_df['Mean Area (pixels)'] = train_mean_areas
test_df['Total Area (pixels)'] = test_total_areas
test_df['Mean Area (pixels)'] = test_mean_areas

# Save the updated DataFrames to new CSV files (without changing the original columns)
train_df.to_csv('C:\\Users\\hp\\Downloads\\train_features_updated.csv', index=False)
test_df.to_csv('C:\\Users\\hp\\Downloads\\test_features_updated.csv', index=False)

# Optionally, print the first few rows of the updated DataFrames
print("\nUpdated Train Dataset Head:")
print(train_df.head())

print("\nUpdated Test Dataset Head:")
print(test_df.head())

import pandas as pd
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Load images from folder
def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in listdir(folder_path):
        img_path = join(folder_path, filename)
        if isfile(img_path):
            img = cv2.imread(img_path)
            images.append(img)
            filenames.append(filename)
    return images, filenames

# Function to calculate the total area of food in an image
def calculate_food_area(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour by area
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)  # Area of the food object
        return area
    else:
        return 0

# Load the train and test datasets
train_df = pd.read_csv('C:\\Users\\hp\\Downloads\\train_features_updated.csv')
test_df = pd.read_csv('C:\\Users\\hp\\Downloads\\test_features_updated.csv')

# Load the images
train_images, train_filenames = load_images_from_folder('/content/drive/MyDrive/extracted_images_train')
test_images, test_filenames = load_images_from_folder('/content/drive/MyDrive/extracted_images_test')

# Initialize lists for storing food waste percentage for each image
train_food_waste_percentages = []
test_food_waste_percentages = []

# Calculate the food waste percentage for each image in the train set
for idx, img in enumerate(train_images):
    total_area = train_df.loc[idx, 'Total Area (pixels)']
    mean_area = train_df.loc[idx, 'Mean Area (pixels)']

    if mean_area != 0:  # Prevent division by zero
        food_waste_percentage = (mean_area / total_area) * 100
    else:
        food_waste_percentage = 0

    # Cap the food waste percentage at 100% if it exceeds
    food_waste_percentage = min(food_waste_percentage, 100)

    train_food_waste_percentages.append(food_waste_percentage)

# Calculate the food waste percentage for each image in the test set
for idx, img in enumerate(test_images):
    total_area = test_df.loc[idx, 'Total Area (pixels)']
    mean_area = test_df.loc[idx, 'Mean Area (pixels)']

    if mean_area != 0:  # Prevent division by zero
        food_waste_percentage = (mean_area / total_area) * 100
    else:
        food_waste_percentage = 0

    # Cap the food waste percentage at 100% if it exceeds
    food_waste_percentage = min(food_waste_percentage, 100)

    test_food_waste_percentages.append(food_waste_percentage)

# Append the food waste percentage to the DataFrame for train and test datasets
train_df['Food Waste Percentage (%)'] = train_food_waste_percentages
test_df['Food Waste Percentage (%)'] = test_food_waste_percentages

# Save the updated DataFrames to new CSV files (without changing the original columns)
train_df.to_csv('C:\\Users\\hp\\Downloads\\train_features_with_food_waste.csv', index=False)
test_df.to_csv('C:\\Users\\hp\\Downloads\\test_features_with_food_waste.csv', index=False)

# Optionally, print the first few rows of the updated DataFrames
print("\nUpdated Train Dataset Head:")
print(train_df.head())

print("\nUpdated Test Dataset Head:")
print(test_df.head())

import pandas as pd

# Function to classify based on food waste percentage
def classify_food_waste(waste_percentage):
    if waste_percentage < 5:
        return 'Low Waste'
    elif 5 <= waste_percentage <= 50:
        return 'Medium Waste'
    else:
        return 'High Waste'

# Load both train and test datasets
train_df = pd.read_csv('C:\\Users\\hp\\Downloads\\train_features_with_food_waste.csv')
test_df = pd.read_csv('C:\\Users\\hp\\Downloads\\test_features_with_food_waste.csv')

# Apply the classification to the 'Food Waste Percentage (%)' column and create a new column 'Food Waste Class' for both datasets
train_df['Food Waste Class'] = train_df['Food Waste Percentage (%)'].apply(classify_food_waste)
test_df['Food Waste Class'] = test_df['Food Waste Percentage (%)'].apply(classify_food_waste)

# Save the updated DataFrames to their respective files (without changing the other columns)
train_df.to_csv('C:\\Users\\hp\\Downloads\\train_features_with_food_waste.csv', index=False)
test_df.to_csv('C:\\Users\\hp\\Downloads\\test_features_with_food_waste.csv', index=False)

# Optionally, print the first few rows of the updated DataFrames to check
print("\nUpdated Train Dataset Head:")
print(train_df.head())

print("\nUpdated Test Dataset Head:")
print(test_df.head())

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the train dataset
train_df = pd.read_csv('C:\\Users\\hp\\Downloads\\train_features_with_food_waste.csv')

# Drop any rows with missing values in the 'Food Waste Class' column
train_df = train_df.dropna(subset=['Food Waste Class'])

# Feature columns (you can adjust these based on your dataset)
features = ['Total Area (pixels)', 'Mean Area (pixels)', 'Waste Area (pixels)', 'Waste Percentage (%)',
            'Texture Area (pixels)', 'Texture Percentage (%)', 'Shape Area (pixels)', 'Shape Percentage (%)']

# Target column
target = 'Food Waste Class'

# Prepare the feature matrix (X) and target vector (y)
X = train_df[features]
y = train_df[target]

# Convert the target labels (Food Waste Class) into numeric labels using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the validation set
y_pred = rf_classifier.predict(X_val)

# Print the classification report and accuracy score
print("\nRandom Forest Model Performance:")
print(f"Accuracy Score: {accuracy_score(y_val, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

# Optionally, you can save the trained model using joblib or pickle for later use
# import joblib
# joblib.dump(rf_classifier, 'food_waste_rf_model.pkl')

import joblib
joblib.dump(rf_classifier, 'food_waste_rf_model.pkl')

import pandas as pd
import joblib

# Load the test dataset (containing the food waste percentage and features)
test_df = pd.read_csv('C:\\Users\\hp\\Downloads\\test_features_with_food_waste.csv')

# Load the trained model (if not already loaded)
rf_classifier = joblib.load('food_waste_rf_model.pkl')

# Define the same feature columns as used in training
features = ['Total Area (pixels)', 'Mean Area (pixels)', 'Waste Area (pixels)', 'Waste Percentage (%)',
            'Texture Area (pixels)', 'Texture Percentage (%)', 'Shape Area (pixels)', 'Shape Percentage (%)']

# Prepare the feature matrix (X) for prediction
X_test = test_df[features]

# Use the trained Random Forest model to predict the food waste class
y_pred = rf_classifier.predict(X_test)

# Decode the predicted labels back to original food waste classes
label_encoder = joblib.load('label_encoder.pkl')  # Assuming you saved the encoder during training
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Add the predicted food waste classes to the test DataFrame
test_df['Predicted Food Waste Class'] = y_pred_decoded

# Save the updated test DataFrame with predictions
test_df.to_csv('C:\\Users\\hp\\Downloads\\test_predictions.csv', index=False)

# Optionally, print the first few rows of the updated DataFrame
print("\nUpdated Test Dataset with Predictions:")
print(test_df.head())

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Get the true labels and predicted labels
y_true = test_df['Food Waste Class']  # Actual class labels from the test dataset
y_pred = test_df['Predicted Food Waste Class']  # Predicted class labels from the model

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calculate precision
precision = precision_score(y_true, y_pred, average='weighted')

# Calculate recall
recall = recall_score(y_true, y_pred, average='weighted')

# Calculate F1 score
f1 = f1_score(y_true, y_pred, average='weighted')

# Print the results
print("Confusion Matrix:")
print(cm)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision (Weighted): {precision:.2f}")
print(f"Recall (Weighted): {recall:.2f}")
print(f"F1-Score (Weighted): {f1:.2f}")

