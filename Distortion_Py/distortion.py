import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import glob


from numpy.core.fromnumeric import resize

def correct_distortion(img, strength, zoom):
    x, y = img.shape[:2]
    half_x = x / 2
    half_y = y / 2
    correction_radius = ((half_x ** 2 + half_y ** 2) ** 0.5) / (strength + 0.00001)

    destination = np.zeros((x, y, 3), np.uint8)

    for i in range(x):
        for j in range(y):
            newI = i - half_x
            newJ = j - half_y

            dist = (newI ** 2 + newJ ** 2) ** 0.5
            r = dist / correction_radius

            theta = 1
            if r != 0:
                theta = np.arctan(r) / r
            
            sourceX = int(half_x + theta * newI * zoom)
            sourceY = int(half_y + theta * newJ * zoom)

            if (sourceX < x and sourceY < y):
                destination[i, j] = img[sourceX, sourceY]
    return destination

def calibrate(show):
    CHECKERBOARD = (6,8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = []

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    images = glob.glob('./cali_s3/*.png')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            imgpoints.append(corners2)

            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        if show == True:
            img = cv2.resize(img, (960, 640))
            cv2.imshow('img',img)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs
    
def correct_perspective(original, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    gray = clahe.apply(gray)
    blurred = cv2.blur(gray, (9, 9))
    (thresh_val, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(thresh_val)
    threshs = cv2.resize(thresh, (960, 540))
    cv2.imshow("Thresh", threshs)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations = 5)
    closed = cv2.dilate(closed, None, iterations = 5)
    closeds = cv2.resize(closed, (960, 540))
    cv2.imshow("Closed", closeds)
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

    max = 0    
    max_index = 0

    print(len(cnts))
    for i in range(len(cnts)):
        c = cnts[i]
        cv2.drawContours(img, c, -1, (255, 0, 0), 3)
        area = cv2.contourArea(c, False)
        if area > max:
            max = area
            max_index = i

    approx = cv2.approxPolyDP(cnts[max_index], 2, True)
    bound = cv2.boundingRect(cnts[max_index])
    print(approx)

    max_xpy = 0
    max_xpy_index = []
    min_xpy = 100000
    min_xpy_index = []
    max_xmy = -1000000
    max_xmy_index = []
    min_xmy = 100000
    min_xmy_index = []

    for i in approx:
        pt = i[0]
        xpy = pt[0] + pt[1]
        xmy = pt[0] - pt[1]
        if xpy > max_xpy:
            max_xpy = xpy
            max_xpy_index = pt
        if xpy < min_xpy:
            min_xpy = xpy
            min_xpy_index = pt
        if xmy > max_xmy:
            max_xmy = xmy
            max_xmy_index = pt
        if xmy < min_xmy:
            min_xmy = xmy
            min_xmy_index = pt
        cv2.circle(img, i[0], 5, [0, 0, 255], -1)

    cv2.circle(img, max_xpy_index, 10, [0, 255, 255], -1)
    cv2.circle(img, min_xpy_index, 10, [0, 255, 255], -1)
    cv2.circle(img, max_xmy_index, 10, [0, 255, 255], -1)
    cv2.circle(img, min_xmy_index, 10, [0, 255, 255], -1)


    # start = np.float32([(approx[0][0][0], approx[0][0][1]),
    #                     (approx[3][0][0], approx[3][0][1]), 
    #                     (approx[2][0][0], approx[2][0][1]), 
    #                     (approx[1][0][0], approx[1][0][1])])
    start = np.float32([min_xpy_index,
                        max_xmy_index, 
                        max_xpy_index, 
                        min_xmy_index])
    dest = np.float32([(bound[0], bound[1]),
                        (bound[0] + bound[2], bound[1]), 
                        (bound[0] + bound[2], bound[1] + bound[3]), 
                        (bound[0], bound[1] + bound[3])])


    w, h = img.shape[:2]
    M = cv2.getPerspectiveTransform(start, dest)
    warped = cv2.warpPerspective(original, M, (h, w))
    plt.show()
    start = start.reshape((-1,1,2))
    dest = dest.reshape((-1,1,2))
    cv2.polylines(img, np.int32([start]), True, (0,255,0))
    cv2.polylines(img, np.int32([dest]), True, (0,0,255))
    cv2.imshow('Contours', img)
    cv2.imshow('Warped', warped)
    cv2.waitKey(0)

def gamma(img, v):
	inv = 1.0 / v
	lut = np.array([((i / 255.0) ** inv) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(img, lut)


if __name__ == '__main__':
    image = cv2.imread(str(sys.argv[1]))
    image_s = cv2.resize(image, (960, 640))

    # gray = cv2.cvtColor(image_s, cv2.COLOR_BGR2GRAY)
    # (thresh_val, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    # cv2.CHAIN_APPROX_SIMPLE)

    # max = 0    
    # max_index = 0

    # print(len(cnts))
    # for i in range(len(cnts)):
    #     c = cnts[i]
    #     print(c)
    #     cv2.drawContours(thresh, c, -1, (255, 0, 0), 3)
    #     area = cv2.contourArea(c, False)
    #     if area > max:
    #         max = area
    #         max_index = i

    # closeds = cv2.resize(thresh, (960, 540))
    # approx = cv2.approxPolyDP(cnts[max_index], 2, True)
    # bound = cv2.boundingRect(cnts[max_index])
    # blurred = cv2.blur(thresh, (9, 9))

    # thresh_s = cv2.resize(blurred, (960, 640))
    

    # cv2.imshow("Closed", thresh_s)
    # cv2.waitKey(0)
    



    # ret, mtx, dist, rvecs, tvecs = calibrate(bool(int(sys.argv[2])))
    # print(f"error: {ret}")
    # print(mtx)
    # print(dist)
    
    # h, w = image.shape[:2]
    # w1, h1 = 1 * w, 1 * h

    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w1, h1))
    # mapx, mapy=cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w1, h1), 5)
    # dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    # x, y, w, h = roi
    # # dst = dst[y : y + h, x : x + w]

    # dst_s = cv2.resize(dst, (960, 640))    
    # cv2.imshow("original", image_s)
    # cv2.imshow("undistorted image", dst_s )
    # cv2.waitKey(0)






    result = gamma(image_s, 0.25)

    correct_perspective(image_s, result)

    result = correct_distortion(image, float(sys.argv[2]), float(sys.argv[3]))
    cv2.imshow("result", result)
    cv2.waitKey(0)
