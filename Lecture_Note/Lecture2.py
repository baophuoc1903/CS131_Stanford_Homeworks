import matplotlib.pyplot as plt
import cv2
import numpy as np


def read_rgb(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_img(img):
    plt.figure(figsize=(15, 7))
    plt.imshow(img)
    plt.show()


def compare_before_after(before, after):
    plt.figure(figsize=(15, 7))
    plt.subplot(121)
    plt.imshow(before)
    plt.title("Before")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(after)
    plt.title("After white balance")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def von_kries(img):
    gray = img[279, 255]  # Choose a point by hand (Assume this point is true gray)
    img_min, img_max = img.min(), img.max()
    r, g, b = gray
    img = img.astype(float)
    img[:, :, 0] = img[:, :, 0] / r
    img[:, :, 1] = img[:, :, 1] / g
    img[:, :, 2] = img[:, :, 2] / b

    # normalize again to change dynamic range of each color channel
    img = cv2.normalize(img, None, img_min, img_max, cv2.NORM_MINMAX).astype(np.uint8)

    return img


def gray_world(img):
    r_ave, g_ave, b_ave = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()
    img = img.astype(float)
    img_min, img_max = img.min(), img.max()

    img[:, :, 0] = img[:, :, 0] / r_ave
    img[:, :, 1] = img[:, :, 1] / g_ave
    img[:, :, 2] = img[:, :, 2] / b_ave

    # normalize again to change dynamic range of each color channel
    img = cv2.normalize(img, None, img_min, img_max, cv2.NORM_MINMAX).astype(np.uint8)

    return img


def brightest_pixel(img):
    img = img.astype(float)
    img_min, img_max = img.min(), img.max()

    img[:, :, 0] = img[:, :, 0] / img_max
    img[:, :, 1] = img[:, :, 1] / img_max
    img[:, :, 2] = img[:, :, 2] / img_max

    # normalize again to change dynamic range of each color channel
    img = cv2.normalize(img, None, img_min, img_max, cv2.NORM_MINMAX).astype(np.uint8)

    return img


if __name__ == '__main__':
    # Read image
    img = read_rgb(r"Images/white_balance.png")
    orig_img = img.copy()

    ''' 0 for "Von Kries"
        1 for "Gray World Assumption"
    '''
    using = 1

    if using == 0:  # Von Kries method
        img = von_kries(img)
    elif using == 1:  # Gray World Assumption
        img = gray_world(img)
    elif using == 2:  # Brightest Pixel Assumption (this work on non-saturated image)
        img = brightest_pixel(img)

    # Plot
    compare_before_after(orig_img, img)
