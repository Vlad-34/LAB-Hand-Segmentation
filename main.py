import sys
from math import sqrt

import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(histogram, title):
    plt.plot(histogram)
    plt.title(f'Histogramă {title}')
    plt.xlabel('Valoare')
    plt.ylabel('Frecvență')
    plt.show()


# Global variables to track mouse events
start_x, start_y = -1, -1
drawing = False
a_global_histogram = np.zeros(256)
b_global_histogram = np.zeros(256)


# noinspection PyUnusedLocal
def draw_rectangle(event, x, y, flags, param):
    global start_x, start_y, drawing, image, a_global_histogram, b_global_histogram

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            canvas = image.copy()
            cv2.rectangle(canvas, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow('Eșantionați imaginea', canvas)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(image, (start_x, start_y), (x, y), (0, 255, 0), 2)
        cv2.imshow('Eșantionați imaginea', image)

        # Create a mask for the selection
        mask = np.zeros_like(image)
        cv2.rectangle(mask, (start_x, start_y), (x, y), (255, 255, 255), -1)

        # Calculate the local histogram
        a_local_histogram = cv2.calcHist([image], [1], mask[:, :, 0], [256], [0, 256])
        b_local_histogram = cv2.calcHist([image], [2], mask[:, :, 0], [256], [0, 256])

        # Flatten the local histogram
        a_local_histogram = a_local_histogram.flatten()
        b_local_histogram = b_local_histogram.flatten()

        # Add the local histogram to the global histogram
        a_global_histogram += a_local_histogram
        b_global_histogram += b_local_histogram


def dilate_erode(input_image):
    kernel = np.ones((3, 3), np.uint8)
    output_image = cv2.dilate(input_image, kernel, iterations=2)
    output_image = cv2.erode(output_image, kernel, iterations=4)
    output_image = cv2.dilate(output_image, kernel, iterations=2)
    return output_image


# Segmenting constants
K_MANHATTAN = 1.19
K_EUCLID = 2.2
K_MAHALANOBIS = 1.5

K_MANHATTAN_A = 1.175
K_EUCLID_A = 1.15
K_MAHALANOBIS_A = 1.15

K_MANHATTAN_B = 1
K_EUCLID_B = 1
K_MAHALANOBIS_B = 1


def segment_image():
    global K_MANHATTAN, K_EUCLID, K_MAHALANOBIS, K_MANHATTAN_A, K_EUCLID_A, K_MAHALANOBIS_A, K_MANHATTAN_B, K_EUCLID_B, K_MAHALANOBIS_B

    sample_image = cv2.imread('Img/03.bmp')
    filtered = cv2.GaussianBlur(sample_image, (5, 5), 0)
    lab_image = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)

    with open("results/a_skin_model.txt", "r") as a_skin_model:
        global a_global_histogram
        a_global_histogram = a_skin_model.readlines()

    a_global_histogram = [line.strip() for line in a_global_histogram]
    a_global_histogram = np.array(a_global_histogram).astype(np.uint8)

    with open("results/b_skin_model.txt", "r") as b_skin_model:
        global b_global_histogram
        b_global_histogram = b_skin_model.readlines()

    b_global_histogram = [line.strip() for line in b_global_histogram]
    b_global_histogram = np.array(b_global_histogram).astype(np.uint8)

    for i in range(len(a_global_histogram)):
        if a_global_histogram[i] < 0.1 * a_global_histogram.max():
            a_global_histogram[i] = 0

    for i in range(len(b_global_histogram)):
        if b_global_histogram[i] < 0.1 * b_global_histogram.max():
            b_global_histogram[i] = 0

    a_mean_val, a_std_val = cv2.meanStdDev(a_global_histogram)
    b_mean_val, b_std_val = cv2.meanStdDev(b_global_histogram)

    segmented_image_manhattan = np.zeros_like(sample_image[:, :, 0])
    segmented_image_euclidian = np.zeros_like(sample_image[:, :, 0])
    segmented_image_mahalanobis = np.zeros_like(sample_image[:, :, 0])

    segmented_image_manhattan_a = np.zeros_like(sample_image[:, :, 0])
    segmented_image_euclidian_a = np.zeros_like(sample_image[:, :, 0])
    segmented_image_mahalanobis_a = np.zeros_like(sample_image[:, :, 0])

    segmented_image_manhattan_b = np.zeros_like(sample_image[:, :, 0])
    segmented_image_euclidian_b = np.zeros_like(sample_image[:, :, 0])
    segmented_image_mahalanobis_b = np.zeros_like(sample_image[:, :, 0])

    rows, cols = sample_image.shape[:2]
    for i in range(rows):
        for j in range(cols):
            euclidean_distance = sqrt((lab_image[i, j, 1] - a_mean_val[0][0]) ** 2 +
                                      (lab_image[i, j, 2] - b_mean_val[0][0]) ** 2)
            if euclidean_distance < (K_EUCLID / 2) * sqrt(a_std_val[0][0] ** 2 + b_std_val[0][0] ** 2):
                segmented_image_euclidian[i, j] = 255
            else:
                segmented_image_euclidian[i, j] = 0

            if (a_mean_val[0][0] - K_MANHATTAN * a_std_val[0][0] < lab_image[i][j][1] < a_mean_val[0][0] + K_MANHATTAN * a_std_val[0][0] and
                    b_mean_val[0][0] - K_MANHATTAN * b_std_val[0][0] < lab_image[i][j][2] < b_mean_val[0][0] + K_MANHATTAN * b_std_val[0][0]):
                segmented_image_manhattan[i][j] = 255
            else:
                segmented_image_manhattan[i][j] = 0

            mahalanobis_distance = sqrt(((lab_image[i, j, 1] - a_mean_val[0][0]) / a_std_val[0][0]) ** 2 + (
                    (lab_image[i, j, 2] - b_mean_val[0][0]) / b_std_val[0][0]) ** 2)
            if mahalanobis_distance < K_MAHALANOBIS:
                segmented_image_mahalanobis[i, j] = 255
            else:
                segmented_image_mahalanobis[i, j] = 0

            # Channel A
            euclidean_distance_a = sqrt((lab_image[i, j, 1] - a_mean_val[0][0]) ** 2)
            if euclidean_distance_a < K_EUCLID_A * sqrt(a_std_val[0][0] ** 2):
                segmented_image_euclidian_a[i, j] = 255
            else:
                segmented_image_euclidian_a[i, j] = 0

            if (a_mean_val[0][0] - K_MANHATTAN_A * a_std_val[0][0] < lab_image[i][j][1] <
                    a_mean_val[0][0] + K_MANHATTAN_A * a_std_val[0][0]):
                segmented_image_manhattan_a[i][j] = 255
            else:
                segmented_image_manhattan_a[i][j] = 0

            mahalanobis_distance_a = sqrt(((lab_image[i, j, 1] - a_mean_val[0][0]) / a_std_val[0][0]) ** 2)
            if mahalanobis_distance_a < K_MAHALANOBIS_A:
                segmented_image_mahalanobis_a[i, j] = 255
            else:
                segmented_image_mahalanobis_a[i, j] = 0

            # Channel B
            euclidean_distance_b = sqrt((lab_image[i, j, 2] - b_mean_val[0][0]) ** 2)
            if euclidean_distance_b < K_EUCLID_B * sqrt(b_std_val[0][0] ** 2):
                segmented_image_euclidian_b[i, j] = 255
            else:
                segmented_image_euclidian_b[i, j] = 0

            if (b_mean_val[0][0] - K_MANHATTAN_B * b_std_val[0][0] < lab_image[i][j][2] < b_mean_val[0][0] + K_MANHATTAN_B *
                    b_std_val[0][0]):
                segmented_image_manhattan_b[i][j] = 255
            else:
                segmented_image_manhattan_b[i][j] = 0

            mahalanobis_distance_b = sqrt(((lab_image[i, j, 2] - b_mean_val[0][0]) / b_std_val[0][0]) ** 2)
            if mahalanobis_distance_b < K_MAHALANOBIS_B:
                segmented_image_mahalanobis_b[i, j] = 255
            else:
                segmented_image_mahalanobis_b[i, j] = 0

    segmented_image_manhattan = dilate_erode(segmented_image_manhattan)
    segmented_image_euclidian = dilate_erode(segmented_image_euclidian)
    segmented_image_mahalanobis = dilate_erode(segmented_image_mahalanobis)

    segmented_image_manhattan_a = dilate_erode(segmented_image_manhattan_a)
    segmented_image_euclidian_a = dilate_erode(segmented_image_euclidian_a)
    segmented_image_mahalanobis_a = dilate_erode(segmented_image_mahalanobis_a)

    segmented_image_manhattan_b = dilate_erode(segmented_image_manhattan_b)
    segmented_image_euclidian_b = dilate_erode(segmented_image_euclidian_b)
    segmented_image_mahalanobis_b = dilate_erode(segmented_image_mahalanobis_b)

    cv2.imshow("segmented_image_manhattan", segmented_image_manhattan)
    cv2.imshow("segmented_image_euclidian", segmented_image_euclidian)
    cv2.imshow("segmented_image_mahalanobis", segmented_image_mahalanobis)

    cv2.imshow("segmented_image_manhattan_a", segmented_image_manhattan_a)
    cv2.imshow("segmented_image_euclidian_a", segmented_image_euclidian_a)
    cv2.imshow("segmented_image_mahalanobis_a", segmented_image_mahalanobis_a)

    cv2.imshow("segmented_image_manhattan_b", segmented_image_manhattan_b)
    cv2.imshow("segmented_image_euclidian_b", segmented_image_euclidian_b)
    cv2.imshow("segmented_image_mahalanobis_b", segmented_image_mahalanobis_b)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


while True:
    user_input = input("""Alegeți o opțiune:
0. Ieșire.
1. Construirea modelului.
2. Segmentarea imaginilor.
Opțiunea aleasă: """)
    if user_input == '0':
        sys.exit()

    elif user_input == '1':  # build model
        cv2.namedWindow('Eșantionați imaginea')
        cv2.setMouseCallback('Eșantionați imaginea', draw_rectangle)
        image = cv2.imread('Img/01.bmp')

        while True:
            cv2.imshow("Eșantionați imaginea", image)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # Press 'ESC' to build model
                break

        a_global_histogram[0], a_global_histogram[255], b_global_histogram[0], b_global_histogram[255] = 0, 0, 0, 0

        with open("results/a_skin_model.txt", "w") as file:
            for a_value in a_global_histogram:
                file.write(str(np.uint8(a_value)) + '\n')

        with open("results/b_skin_model.txt", "w") as file:
            for b_value in b_global_histogram:
                file.write(str(np.uint8(b_value)) + '\n')

        cv2.destroyAllWindows()

    elif user_input == '2':
        segment_image()
