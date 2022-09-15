import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import os


def compare_images(images, labels=None):
    """
    Use to visualize images

    :param images: list of image to visualize
    :param labels: list of label corresponding to each image. Default None for non-label images
    """
    num_img = len(images)

    fig, ax = plt.subplots(nrows=1, ncols=num_img, figsize=(18, 7))
    ax =ax.flatten()

    for i in range(num_img):
        ax[i].imshow(images[i], cmap='gray')
        if labels:
            ax[i].set_title(labels[i])
        ax[i].axis("off")
    plt.show()


def gaussian_blur(img, ksize=3, sigma=1):
    """
    Using gaussian blur for a given image to blur that image

    :param img: Original image used for gaussian blur
    :param ksize: Gaussian kernel size, default is 3
    :param sigma: Gaussian kernel standard deviation, default is 1

    :return: result image of gaussian blur
    """
    gauss_img = cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    return gauss_img


def sobel_filter(img):
    """
    Using sobel filters for a given image to extract edge in that image

    :param img: original image used for sobel edge detection. Regularly img is a blur image
     to reduce the noise

    :return: G_magnitude: Sobel gradient magnitude
             tangent: Gradient direction
    """

    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(float)
    filter_y = -filter_x.T

    Gx = convolve(img.astype(float), filter_x)
    Gy = convolve(img.astype(float), filter_y)
    G_magnitude = np.hypot(Gx, Gy)
    tangent = np.arctan2(Gy, Gx)

    # Normalize G_magnitude to range [0, 255]
    # G_magnitude = cv2.normalize(G_magnitude, None, 0, 255, norm_type=cv2.NORM_MINMAX)

    return G_magnitude.astype(int), tangent


def non_max_suppression(img, theta):
    """
    Using non-max-suppression to reject pixel that is not local maximum compare to it's neighbor
    those have the same direction (Assume 3x3 block as neighbor and direction is quantized in 4 main way.
    As the result, 4 pairs of result are extracted to compare with in-process pixel)
    ==> Output is an image that has thinner edges than original image

    :param img: Edge image that need to used non-max suppression for thinner edge
    :param theta: Gradient direction matrix,
                  using to define neighbors pixel that have same direction with in-process pixel
    :return: input image after apply non-max suppression
    """
    out = np.zeros_like(img)
    theta[theta < 0] = theta[theta < 0] + np.pi
    theta = theta * 180 / np.pi


    for i in range(1, len(out)-1):
        for j in range(1, len(out[0])-1):
            r = q = None
            if 0 <= theta[i][j] < 22.5 or 337.5 <= theta[i][j] or 157.5 <= theta[i][j] < 202.5:
                r = img[i, j - 1]
                q = img[i, j + 1]
            elif 22.5 <= theta[i][j] < 67.5 or 202.5 <= theta[i][j] < 247.5:
                r = img[i-1, j+1]
                q = img[i+1, j-1]
            elif 67.5 <= theta[i][j] < 112.5 or 247.5 <= theta[i][j] < 292.5:
                r = img[i-1, j]
                q = img[i+1, j]
            else:
                r = img[i-1][j-1]
                q = img[i+1][j+1]

            if img[i][j] >= r and img[i][j] >= q:
                out[i][j] = img[i][j]

    return out


def double_threshold(img, low_thresh_ratio=0.04, high_thresh_ratio=0.1, weak_edge=32, strong_edge=255):
    """
    Using double threshold to classify each edge in image into weak edge or strong edge

    :param img: edge image extracted from sobel filter algorithm above or others algorithms
    :param low_thresh_ratio: threshold ratio use for classify non-edge and weak edge. Default 0.04
    :param high_thresh_ratio: threshold ratio use for classify weak edge and strong edge. Default 0.1
    :param weak_edge: pixel value of weak edge in the output image. Default 32
    :param strong_edge: pixel value of strong edge in the output image. Default 255

    :return: image after thresholding, each pixel value is 1 of 3 value: (0, weak_edge, strong_edge)
    """
    lowThresh = img.max() * low_thresh_ratio
    highThresh = img.max() * high_thresh_ratio

    out = np.zeros_like(img)
    out[(lowThresh <= img) & (img < highThresh)] = weak_edge
    out[img > highThresh] = strong_edge

    return out


def hysteresis_tracking(img, weak_edge=32, strong_edge=255):
    """
    Using to transform weak edges, which maybe has potential to become a strong edge, into strong edge

    :param img: double thresholding edge image. Only have 3 pixel value: (0, weak_edge, strong_edge)
    :param weak_edge: weak edge pixel value. Default 32
    :param strong_edge: strong edge pixel value. Default 255

    :return: image after edge tracking by hysteresis
    """
    out = img.copy()
    for i in range(len(out)):
        for j in range(len(out[0])):
            if out[i][j] == weak_edge:
                if out[i+1][j] == strong_edge or out[i-1][j] == strong_edge or \
                   out[i][j-1] == strong_edge or out[i][j+1] == strong_edge or \
                   out[i-1][j-1] == strong_edge or out[i-1][j+1] == strong_edge or \
                   out[i+1][j-1] == strong_edge or out[i+1][j+1] == strong_edge:
                    out[i][j] = strong_edge
                else:
                    out[i][j] = 0
    return out


if __name__ == '__main__':
    # Run to visualize each step of Canny edge detection
    img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), r"Images/emma_stone.jpg")
    orig_img = cv2.imread(img_path, 0)

    # Gaussian blur to reduce noise
    blur_img = gaussian_blur(orig_img, 3, 1)

    # Sobel filter to extract edge
    G_mag, theta = sobel_filter(orig_img)

    # Non-max suppression to make edge thinner
    non_max = non_max_suppression(G_mag, theta)

    # Thresholding to classify edge to weak or strong edge
    threshold = double_threshold(non_max, low_thresh_ratio=5/255, high_thresh_ratio=30/255, weak_edge=32)

    # Hysteresis tracking to transform weak edge to strong edge if it passed the condition
    canny_img = hysteresis_tracking(threshold)

    # Visualize canny edge detection precess
    compare_images([orig_img, G_mag, non_max, threshold, canny_img], ["Original", "G_magnitude",
                                                                      "After non-max suppression",
                                                                      "Double Threshold", "Canny result"])

