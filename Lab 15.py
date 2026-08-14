import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
img = cv2.imread(
    r"C:\Users\mbala\Downloads\istockphoto-1152113369-612x612.jpg"
)

# Check if the image was loaded successfully
if img is None:
    print("Error: Could not load image.")
    print("Please check the file path.")
else:
    # Convert BGR image to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert image pixels into a 2D array
    pixels = np.float32(rgb_img.reshape((-1, 3)))

    # Define K-means criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    # Number of clusters
    K = 3

    # Apply K-means clustering
    _, labels, centers = cv2.kmeans(
        pixels,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    # Convert cluster centers to uint8
    centers = np.uint8(centers)

    # Create segmented image
    segmented_img = centers[labels.flatten()].reshape(rgb_img.shape)

    # Display original and segmented images
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Original Image")
    plt.axis("off")

    # Segmented image
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_img)
    plt.title("Segmented Image (K-means)")
    plt.axis("off")

    # Adjust layout
    plt.tight_layout()

    # Show result
    plt.show()
