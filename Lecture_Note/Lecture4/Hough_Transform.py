import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from Canny import canny_edge_detection, compare_images


def hough_transform(img, theta_step=1):
    """
    Transform image from x,y space into polar r,theta space (y=a*x+b => r=y*sin(theta)+x*cos(theta))

    :param img: original image
    :param theta_step: parameter that used to quantize theta (from -90 degree to 90 degree with theta_step size)
    :return: hough_img: 2D matrix - line image in r,theta space
             thetas: 1D array - theta quantization (length equal number of column in hough_img)
             dist: 1D array - radius quantization (length equal number of rows in hough_img)
    """
    # Angle quantization
    thetas = np.deg2rad(np.arange(-90, 90, theta_step))

    n_cols = len(thetas)

    # Radius quantization
    max_dist = int(np.around(np.hypot(img.shape[0], img.shape[1])))
    n_rows = 2 * max_dist + 1

    hough_img = np.zeros((n_rows, n_cols), dtype=int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 0:
                for k in range(n_cols):
                    r_quantize = int(np.around(i*np.sin(thetas[k])+j*np.cos(thetas[k])))
                    hough_img[r_quantize+max_dist, k] += 1

    dist = np.linspace(-max_dist, max_dist, 2*max_dist+1)
    return hough_img, thetas, dist


def draw_line(img, list_theta, list_p, thickness=1):
    """
    Drawing a list of lines given the p,theta parameter corresponding in polar space

    :param img: image that need to draw lines
    :param list_theta: list of angles of lines
    :param list_p: list of radius of lines corresponding to list_theta
    :param thickness: thickness of line that is drawn
    :return: img after draw all lines
    """
    out = img.copy()

    for theta, p in zip(list_theta, list_p):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * p
        y0 = b * p
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(out, pt1, pt2, (255, 0, 0), thickness, cv2.LINE_AA)

    return out


if __name__ == '__main__':
    # Run to visualize each step of Canny edge detection
    img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), r"Images/hough_2.png")
    orig_img = cv2.imread(img_path, 0)

    canny_img, _ = canny_edge_detection(orig_img)
    hough_img, thetas, ps = hough_transform(canny_img, theta_step=1)

    threshold = 160
    y_idx, x_idx = np.where(hough_img > threshold)

    line_detection = draw_line(cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB), thetas[x_idx], ps[y_idx])

    compare_images(images=[orig_img, canny_img, hough_img, line_detection],
                   labels=["Original", "Canny", "Hough transform", f"{len(y_idx)} Line Detected"])
