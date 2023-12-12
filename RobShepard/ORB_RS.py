''' 
#Robert Shepard
#CSEC 472 - Lab 4

import cv2
import os

def compare_fingerprints(image_path1, image_path2):
    # loading fingerprint images
    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # ORB detector for feature extraction
    orb = cv2.ORB_create()
    keypoint1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoint2, descriptors2 = orb.detectAndCompute(image2, None)

    #  feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    similarity_score = len(matches) #how many feature matches there are between both fingerprints

    return similarity_score

def main():
    print("Comparing fingerprints for authentication...")

    fingerprint_dir = 'C:/Users/rober/Downloads/Tests_Subjects/Tester1Finger'
    subject_dir = 'C:/Users/rober/Downloads/Tests_Subjects/Tester2Subject'
    image_files_a = []
    image_files_b = []

    #load files for comparison 
    for file in os.listdir(fingerprint_dir):
        if file.endswith('.png'):
            image_files_a.append(file)
        else:
            print("file loading error")
        
    
    for file in os.listdir(subject_dir):
        if file.endswith('.png'):
            image_files_b.append(file)
        else:
            print("file loading error")
   

    # Compare fingerprints vars
    total_fingerprints = len(image_files_a)
    authenticated_fingerprints = []
    denied_fingerprints = []
    true_auth_counter = 0
    true_deny_counter = 0
    false_pos_auth_counter = 0
    false_reject_counter = 0
    similarity_comparison = 170
   
    # compare each index of fingerprints to every subject fingerprint to determine authentication: 
    for i in range(total_fingerprints):  
        fingerprint_image = os.path.join(fingerprint_dir, image_files_a[i])
        for i2 in range(total_fingerprints):
            subject_image = os.path.join(subject_dir, image_files_b[i2])
            similarity_score = compare_fingerprints(fingerprint_image, subject_image)
            
            if similarity_score > similarity_comparison:
                authenticated_fingerprints.append((fingerprint_image, subject_image)) #append tuple of authenticated fingerprints/images
                if fingerprint_image[-11:] == subject_image[-11:]:    #determine if they are the same fingerprint for successful authentication
                    true_auth_counter += 1
                else:
                    false_pos_auth_counter += 1
            else:
                denied_fingerprints.append((fingerprint_image, subject_image)) #append tuple of denied fingerprints/images
                if fingerprint_image[-11:] == subject_image[-11:]:
                    false_reject_counter += 1
                else:
                    true_deny_counter += 1

    #Calculate Auth Result Variables to determine technique usefulness 

    #FRR = False Reject Rate
    Frr = false_reject_counter / total_fingerprints
    #FAR = False Acceptance Rate
    Far = false_pos_auth_counter / total_fingerprints
    #EER = Equal Error Rate
    Eer = (Frr + Far) / 2

    print("True Authentication Counter: ",true_auth_counter) 
    print("True Deny Counter: ",true_deny_counter)
    print("False Positive Authentication Counter ", false_pos_auth_counter)
    print("False Reject Counter: ", false_reject_counter)
    print("FRR: ", Frr)
    print("FAR: ", Far)
    print("EER: ", Eer)


main()
'''