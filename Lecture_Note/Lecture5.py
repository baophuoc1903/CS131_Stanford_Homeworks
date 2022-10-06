import cv2
import numpy as np
import matplotlib.pyplot as plt


def harris_corner(img, show_plot=True):
    out = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    out[dst > 0.1 * dst.max()] = [0, 0, 255][::-1]

    if show_plot:
        plt.imshow(img)
        plt.show()

    dst[dst > 0.1 * dst.max()] = 1.0
    return out, cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)


if __name__ == '__main__':
    # Read image
    path = r"Images/hough_1.png"
    img = cv2.imread(path)

    # Harris corner detection
    new_img, dst = harris_corner(img, show_plot=False)
    plt.imshow(new_img)
    plt.show()
