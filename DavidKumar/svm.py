#David Kumar
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm, metrics


# Function to read features from the .txt files
def read_features_from_txt(file_path):
    with open(file_path, 'r') as file:
        # Assuming features are stored in a text file, one feature per line
        features = [float(line.strip()) for line in file]
    return features

# Function to extract features from image pairs
def extract_features(image_pairs):
    features_list = []
    for pair in image_pairs:
        # Assuming pair is a tuple (fi, si)
        # Extract features from the pair and append to features_list
        features_fi = read_features_from_txt(pair[0])
        features_si = read_features_from_txt(pair[1])

        features_diff = np.abs(np.array(features_fi) - np.array(features_si))
        features_list.append(features_diff)
    return features_list

# Function to split the overall database into TRAIN and TEST sets
def split_dataset(image_pairs):
    train_pairs, test_pairs = train_test_split(image_pairs, test_size=500, random_state=42)
    return train_pairs, test_pairs

# Load image pairs and organize into {fi, si} pairs
image_folder = "database"
image_pairs = []

for i in range(1, 2001):
    f_image_path = os.path.join(image_folder, f'f{i:04d}.txt')
    s_image_path = os.path.join(image_folder, f's{i:04d}.txt')
    image_pairs.append((f_image_path, s_image_path))

# Split the dataset into TRAIN and TEST sets
train_pairs, test_pairs = split_dataset(image_pairs[:1500])

# Extract features from TRAIN set
train_features = extract_features(train_pairs)


labels = np.random.randint(2, size=len(train_features))


svm_classifier = svm.SVC()
scores_svm = cross_val_score(svm_classifier, train_features, labels, cv=5)
print("SVM Accuracy:", np.mean(scores_svm))
