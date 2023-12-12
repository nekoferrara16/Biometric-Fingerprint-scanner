"""Grey scale image pre processing using open CV with a sliding window
Reads gray scale fingerpint images using openCV (cv2) and then enhances the image contrast
to lalow for more accurate readings. Extracts features using skkeletonization to etract minutiae
from the fingerprints

Uses circular masks to focus on central region of interest then computes the centroid of the minutiae for each fingerprint
Measures the distance of each minutia to its centroid then compares two fingerprint images and returns
a similarity score based on the number of minutiae pairs with the distances
below a specified threshold. Inside of the comparison function it then
calculates the EER, FAR, and FRR on the ROC curve from skmetrics



"""
import os

import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from skimage.morphology import skeletonize
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn import metrics

# creates a circular border around the fingerprint and discards information outside of the circle,
# essentially focuses on one area at a time rather than all at once, breaking the fingerprint into chunks

def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask

# Increases contrast for more accurate finger print reading
def cleanup_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist(img)
    img = cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 181, 11)
    return img


def find_minutiae(path, display=False):
    # cleans up the image first
    img = cleanup_image(path)
    timg = img // 255
    # examine a 3x3x3 window of a pixel and sweep and thins the ridges to obtain a binary image of the
    # fingerprint skeleton
    img = skeletonize(timg, method='lee')

    # computers the center of mass of the skeletonized image
    com = ndimage.center_of_mass(img)
    cmask = create_circular_mask(512, 512, com, 224)
    img[cmask == 0] = 0

    # applies the circular mask to the skeletonized image

    if display:
        plt.imshow(255 - img, 'gray')
        plt.show()

    step_size = 3
    window_size = (3, 3)
    coords = []

    for x in range(0, img.shape[1] - window_size[0], step_size):
        for y in range(0, img.shape[0] - window_size[1], step_size):
            window = img[x:x + window_size[0], y:y + window_size[1]]
            win_mean = np.mean(window)
            if win_mean in (8 / 9, 1 / 9):
                coords.append((x, y))

    # minutiae detection
    coords = np.array(coords)
    coords_centr = centroid_np(coords)
    sort_coords = sorted(coords, key=lambda coord: np.linalg.norm(coord - coords_centr))
    return np.array(sort_coords[1:100])

# computes centroid of detected minutiae, sorts minutiae coordinates
def centroid_np(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x / length, sum_y / length



def compare_prints(path_a, path_b, threshold=7, debug=False):
    minutiae_a = find_minutiae(path_a, display=debug)
    minutiae_b = find_minutiae(path_b, display=debug)
    centroid_a = np.expand_dims(centroid_np(minutiae_a), 0)
    centroid_b = np.expand_dims(centroid_np(minutiae_b), 0)
    # calculate the distance of each minutia point from the centroid for both fingerprints
    dists_a = np.linalg.norm(minutiae_a - centroid_a[:, ], axis=1)
    dists_b = np.linalg.norm(minutiae_b - centroid_b[:, ], axis=1)
    sort_dists = np.array(dists_a) - np.array(dists_b)
    # count the number of minutiae pairs with distances below the threshold
    similarity = len(sort_dists[np.where(abs(sort_dists) < threshold)]) / len(sort_dists)

    # calculate equal error rate and equal acceptance rate (FAR = FRR )

    eer, threshold = calculate_eer(dists_a, dists_b)

    # equal acceptance rate

    x = far, frr = calculate_far_frr(dists_a, dists_b, threshold)

    return round(similarity, 2) , round(eer, 2), round(far, 2) , round(frr, 2)


def calculate_eer(scores_genuine, scores_impostor):
    # Calculate the EER and the corresponding threshold
    all_scores = np.concatenate([scores_genuine, scores_impostor])
    labels = np.concatenate([np.ones_like(scores_genuine), np.zeros_like(scores_impostor)])

    fpr, tpr, thresholds = metrics.roc_curve(labels, all_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, thresholds)(eer)

    return eer, threshold


def calculate_far_frr(scores_genuine, scores_impostor, threshold):
    # Calculate FAR and FRR at the given threshold
    far = np.mean(scores_impostor >= threshold)
    frr = np.mean(scores_genuine < threshold)

    return far, frr


def main(image_path_a, image_path_b):
    # Step 1: Compare fingerprints
    similarity_score = compare_prints(image_path_a, image_path_b, threshold=7, debug=False)

    # Step 2: Display the similarity score
    # print(f" {image_path_a} and {image_path_b}: {similarity_score}")


if __name__ == "__main__":
    # Replace these paths with the actual file paths of your images
    fingerprint = 'fingerprintset'
    subject = 'subjectSet'

    # Call the main function with the image paths
    # Get lists of all files in the folders
    image_files_a = [f for f in os.listdir(fingerprint) if f.endswith('.png')]
    image_files_b = [f for f in os.listdir(subject) if f.endswith('.png')]
    # print("                                   Similarity Score, Equal Error Rate, False Acceptance Rate, False Rejection rate                  ")
    # Iterate through pairs of images

    similarity_scores = []
    eer_scores = []
    far_scores = []
    frr_scores = []

    # Iterate through pairs of images
    for file_a in image_files_a:
        for file_b in image_files_b:
            image_path_a = os.path.join(fingerprint, file_a)
            image_path_b = os.path.join(subject, file_b)

            # Call the compare_prints function with the image paths
            similarity, eer, far, frr = compare_prints(image_path_a, image_path_b, threshold=7, debug=False)

            # Append scores to lists
            similarity_scores.append(similarity)
            eer_scores.append(eer)
            far_scores.append(far)
            frr_scores.append(frr)
            # Calculate and display averages
# Calculate and display averages
    avg_similarity = round(np.mean(similarity_scores), 2)
    avg_eer = round(np.mean(eer_scores), 2)
    avg_far = round(np.mean(far_scores), 2)
    avg_frr = round(np.mean(frr_scores), 2)

# Calculate the mins
    min_sim = round(min(similarity_scores), 2)
    min_eer = round(min(eer_scores), 2)
    min_far = round(min(far_scores), 2)
    min_frr = round(min(frr_scores), 2)

# Calculate the maxs
    max_sim = round(max(similarity_scores), 2)
    max_eer = round(max(eer_scores), 2)
    max_far = round(max(far_scores), 2)
    max_frr = round(max(frr_scores), 2)

    print("\nAverages:")
    print(f"Similarity Score: {avg_similarity}")
    print(f"Equal Error Rate: {avg_eer}")
    print(f"False Acceptance Rate: {avg_far}")
    print(f"False Rejection Rate: {avg_frr}")

    print("\nMinimums:")
    print(f"Similarity Score: {min_sim}")
    print(f"Equal Error Rate: {min_eer}")
    print(f"False Acceptance Rate: {min_far}")
    print(f"False Rejection Rate: {min_frr}")

    print("\nMaximums:")
    print(f"Similarity Score: {max_sim}")
    print(f"Equal Error Rate: {max_eer}")
    print(f"False Acceptance Rate: {max_far}")
    print(f"False Rejection Rate: {max_frr}")




