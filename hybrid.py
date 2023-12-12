#hybrid system



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
def compare_fingerprints(image_path1, image_path2):
    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # ORB detector for feature extraction
    orb = cv2.ORB_create()
    keypoint1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoint2, descriptors2 = orb.detectAndCompute(image2, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    similarity_score_orb = len(matches)

    return similarity_score_orb

def compare_prints_combined(image_path_a, image_path_b, threshold=7, debug=False):
    # Compare fingerprints using minutiae-based method
    similarity_score_minutiae, eer, far, frr = compare_prints(image_path_a, image_path_b, threshold=threshold, debug=debug)

    # Compare fingerprints using ORB-based method
    similarity_score_orb = compare_fingerprints(image_path_a, image_path_b)

    # Combine the scores (you can adjust the weights or choose a different combination method)
    combined_similarity_score = (similarity_score_minutiae + similarity_score_orb) / 2



    return combined_similarity_score, eer, far, frr


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
    fingerprint = 'Tester1Finger'
    subject = 'Tester2Subject'

    image_files_a = [f for f in os.listdir(fingerprint) if f.endswith('.png')]
    image_files_b = [f for f in os.listdir(subject) if f.endswith('.png')]

    similarity_scores_combined = []
    eer_scores_combined = []
    far_scores_combined = []
    frr_scores_combined = []

    for file_a in image_files_a:
        for file_b in image_files_b:
            image_path_a = os.path.join(fingerprint, file_a)
            image_path_b = os.path.join(subject, file_b)

            similarity_combined, eer_combined, far_combined, frr_combined = compare_prints_combined(image_path_a, image_path_b, threshold=7, debug=False)

            similarity_scores_combined.append(similarity_combined)
            eer_scores_combined.append(eer_combined)
            far_scores_combined.append(far_combined)
            frr_scores_combined.append(frr_combined)

    avg_similarity_combined = round(np.mean(similarity_scores_combined), 2)
    avg_eer_combined = round(np.mean(eer_scores_combined), 2)
    avg_far_combined = round(np.mean(far_scores_combined), 2)
    avg_frr_combined = round(np.mean(frr_scores_combined), 2)

    min_sim_combined = round(min(similarity_scores_combined), 2)
    min_eer_combined = round(min(eer_scores_combined), 2)
    min_far_combined = round(min(far_scores_combined), 2)
    min_frr_combined = round(min(frr_scores_combined), 2)

    max_sim_combined = round(max(similarity_scores_combined), 2)
    max_eer_combined = round(max(eer_scores_combined), 2)
    max_far_combined = round(max(far_scores_combined), 2)
    max_frr_combined = round(max(frr_scores_combined), 2)

    print("\nCombined Averages:")
    print(f"Combined Similarity Score: {avg_similarity_combined}")
    print(f"Combined Equal Error Rate: {avg_eer_combined}")
    print(f"Combined False Acceptance Rate: {avg_far_combined}")
    print(f"Combined False Rejection Rate: {avg_frr_combined}")

    print("\nCombined Minimums:")
    print(f"Combined Similarity Score: {min_sim_combined}")
    print(f"Combined Equal Error Rate: {min_eer_combined}")
    print(f"Combined False Acceptance Rate: {min_far_combined}")
    print(f"Combined False Rejection Rate: {min_frr_combined}")

    print("\nCombined Maximums:")
    print(f"Combined Similarity Score: {max_sim_combined}")
    print(f"Combined Equal Error Rate: {max_eer_combined}")
    print(f"Combined False Acceptance Rate: {max_far_combined}")
    print(f"Combined False Rejection Rate: {max_frr_combined}")



