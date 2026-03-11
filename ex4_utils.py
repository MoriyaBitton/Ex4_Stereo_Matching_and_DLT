import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    # disparity range
    min_offset, max_offset = disp_range[0], disp_range[1]
    # disparity map
    disparity_map = np.zeros((img_l.shape[0], img_l.shape[1], max_offset))
    # calculate average value of our image kernel and normalize
    norm_l = img_l - filters.uniform_filter(img_l, k_size)
    norm_r = img_r - filters.uniform_filter(img_r, k_size)

    for offset in range(min_offset, max_offset):
        # move the left img to the right
        steps = offset + min_offset
        norm_l_to_r = np.roll(norm_l, -steps)
        # normalize
        sigma_r = filters.uniform_filter(np.square(norm_r), k_size)
        sigma_l = filters.uniform_filter(np.square(norm_l_to_r), k_size)
        left = norm_l_to_r / sigma_l
        right = norm_r / sigma_r
        sigma_ssd = filters.uniform_filter(np.square(left - right), k_size)

        # update disparity_map with SSD score
        disparity_map[:, :, offset] = np.square(sigma_ssd)

    # for each pixel choose maximum depth value
    return np.argmin(disparity_map, axis=2)


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    # disparity range
    min_offset, max_offset = disp_range[0], disp_range[1]
    # disparity map
    disparity_map = np.zeros((img_l.shape[0], img_l.shape[1], max_offset))
    # calculate average value of our image kernel and normalize
    norm_l = img_l - filters.uniform_filter(img_l, k_size)
    norm_r = img_r - filters.uniform_filter(img_r, k_size)

    for offset in range(min_offset, max_offset):
        # move left img
        steps = offset + min_offset
        norm_l_to_r = np.roll(norm_l, -steps)
        # normalize
        sigma = filters.uniform_filter(norm_l_to_r * norm_r, k_size)
        sigma_l = filters.uniform_filter(np.square(norm_l_to_r), k_size)
        sigma_r = filters.uniform_filter(np.square(norm_r), k_size)

        # update disparity_map with NC score
        disparity_map[:, :, offset] = sigma / np.sqrt(sigma_l * sigma_r)

    # for each pixel choose maximum depth value
    return np.argmax(disparity_map, axis=2)


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
        Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
        returns the homography and the error between the transformed points to their
        destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

        src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
        dst_pnt: 4+ keypoints locations (x,y) on the destination image. Shape:[4+,2]

        return: (Homography matrix shape:[3,3],
                Homography error)
    """
    A = []
    for i in range(src_pnt.shape[0]):
        # init src vector
        x_s, y_s = src_pnt[i][0], src_pnt[i][1]
        # init dest vector
        x_d, y_d = dst_pnt[i][0], dst_pnt[i][1]
        # init A matrix
        A.append([x_s, y_s, 1, 0, 0, 0, -x_d * x_s, -x_d * y_s, -x_d])
        A.append([0, 0, 0, x_s, y_s, 1, -y_d * x_s, -y_d * y_s, -y_d])

    _, _, Vh = np.linalg.svd(np.asarray(A))
    # Isolate H matrix and normalize
    H = (Vh[-1, :] / Vh[-1, -1]).reshape(3, 3)

    # Checking my result VS OpenCV
    # H_cv, _ = cv2.findHomography(src_pnt, dst_pnt)
    # print('Homography Matrix OpenCV\n', H_cv)

    # calculate errors
    Error = 0.
    for i in range(src_pnt.shape[0]):
        src = np.append(src_pnt[i], 1)
        dst = np.append(dst_pnt[i], 1)
        Error = np.sqrt(sum(H.dot(src) / H.dot(src)[-1] - dst) ** 2)

    return H, Error


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
       Displays both images, and lets the user mark 4 or more points on each image. Then calculates the homography and transforms the source image on to the destination image. Then transforms the source image onto the destination image and displays the result.

       src_img: The image that will be 'pasted' onto the destination image.
       dst_img: The image that the source image will be 'pasted' on.

       output:
        None.
    """
    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    ##### Your Code Here ######
    src_p = []
    fig2 = plt.figure()

    def onclick_2(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        src_p.append([x, y])

        if len(src_p) == 4:
            plt.close()
        plt.show()

    # display image 2
    cid = fig2.canvas.mpl_connect('button_press_event', onclick_2)
    plt.imshow(src_img)
    plt.show()
    src_p = np.array(src_p)

    src_p = np.array(src_p, dtype=np.float32)
    dst_p = np.array(dst_p, dtype=np.float32)

    print("src_p:", src_p, src_p.shape)
    print("dst_p:", dst_p, dst_p.shape)

    H_cv, mask = cv2.findHomography(src_p, dst_p)

    warp = np.zeros(dst_img.shape)
    for i in range(src_img.shape[0]):
        for j in range(src_img.shape[1]):
            xy = np.array([j, i, 1]).T
            new_p = H_cv @ xy
            y_ = int(new_p[0]/new_p[-1])
            x_ = int(new_p[1]/new_p[-1])
            if 0 <= x_ < src_img.shape[0] and 0 <= y_ < src_img.shape[1]:
                warp[x_, y_] = src_img[i, j]
            else:
                warp[x_, y_] = src_img[i, j]
    mask = warp == 0
    canvas = dst_img * mask + (1 - mask) * warp
    plt.imshow(canvas)
    plt.savefig("results/homography.png")
    plt.show()
