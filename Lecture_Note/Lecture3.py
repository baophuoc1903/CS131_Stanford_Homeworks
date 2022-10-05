import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import os


def box_filter(image, kernel=3):
    """
    Blurring image using box filter

    :param image: image that need to blur
    :param kernel: kernel size using in blurring filter
    :return: blurring image
    """
    filter = np.ones((kernel, kernel)) / kernel ** 2
    filter_image = convolve(image, filter)

    return filter_image


def sharpen_filter(image, kernel=3):
    """
    Sharpening image

    :param image: image that need to sharpen
    :param kernel: kernel size using in sharpen filter
    :return: sharpen image
    """
    sharp_fil = np.zeros((kernel, kernel))
    sharp_fil[kernel // 2, kernel // 2] = 2

    filter = sharp_fil - np.ones((kernel, kernel)) / kernel ** 2
    detail = convolve(image, filter)

    return detail


def generate_image_for_characterize_edge():
    """
    Generate a image to visualize a characterize edge
    :return: an image
    """
    blur_cols = 25
    black = 50
    white = 35
    blur_cols = blur_cols // 2 * 2
    black = black // 2 * 2
    white = white // 2 * 2
    img_size = blur_cols * 2 + white * 2 + black

    img = np.zeros((img_size, img_size))
    zero_blur = np.linspace(0, 0.4, blur_cols // 2)
    one_blur = np.linspace(0.6, 1, blur_cols // 2)
    blur = np.hstack((zero_blur, one_blur)).reshape(1, -1)

    blur = np.repeat(blur, img_size, axis=0)

    img[:, :white] = 1
    img[:, -white:] = 1
    img[:, white:white + blur_cols] = blur[:, ::-1]
    img[:, img_size - white - blur_cols:img_size - white] = blur
    img = cv2.blur(img, ksize=(15, 15))

    return img


def characterize_edge(image, scanline=None):
    """
    Visualize a characterize edge of a horizontal scanline in image

    :param image: original image
    :param scanline: a horizontal line across image
    """
    if scanline is None:
        scanline = image.shape[0] // 2

    filter = np.array([[1, 0, -1]]) / 3
    filter = filter.repeat(3, axis=0)
    first_derivative = convolve(image, filter)

    plt.figure(figsize=(18, 6))

    plt.subplot(131)
    plt.imshow(cv2.line(image[:, :, None].repeat(3, axis=2),
                        pt1=(0, scanline), pt2=(image.shape[1], scanline),
                        color=(1.0, 0.0, 0.0), thickness=1),
               cmap='gray')
    plt.title("Image")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(132)
    plt.plot(image[scanline], color='k')
    plt.title("Intensity function\n(Along horizontal scanline)")

    plt.subplot(133)
    plt.plot(first_derivative[scanline], color='k')
    plt.title("First derivative")

    plt.show()


if __name__ == '__main__':
    # # ============ FILTERS ===============
    # img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), r"Images/iguana.jpg")
    # img = cv2.imread(img_path, 0)
    #
    # plt.figure(figsize=(10, 5))
    # plt.subplot(121)
    # plt.imshow(img, cmap='gray')
    # plt.title("Original")
    # plt.axis('off')
    # transform_img = box_filter(img, kernel=11)
    # plt.subplot(122)
    # plt.imshow(transform_img, cmap='gray')
    # plt.title("Box filtering (kernel=11)")
    # plt.axis('off')
    # plt.show()

    # # ======== CHARACTERIZE EDGE =========
    # img = generate_image_for_characterize_edge()
    # characterize_edge(img)

    # Image Histogram
    img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), r"Images/iguana.jpg")
    img = cv2.imread(img_path, 0)

    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(122)
    plt.plot(np.arange(256), hist)
    plt.xlabel("Pixel value")
    plt.ylabel("Count")
    plt.title("Pixel value histogram", weight="bold", color='k')
    plt.show()
