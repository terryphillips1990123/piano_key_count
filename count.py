import cv2
import numpy as np
from skimage import measure
from scipy import ndimage
from sklearn.cluster import KMeans
import cv2
import numpy as np
np.random.seed(42)

def preprocess(image):
    h, w, _ = image.shape

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, thresh = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh,50,150,apertureSize=3)


    kernel = np.ones((2,2),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations = 2)


    lines = cv2.HoughLinesP(
                edges, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=50, # Min number of votes for valid line
                minLineLength=100, # Min allowed length of line
                maxLineGap=10 # Max allowed gap between line for joining them
                )   
    # Iterate over points
    if lines is not None:
        lines_list =[]
        for points in lines:
            # Extracted points nested in the list
            x1,y1,x2,y2=points[0]
            length = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            angle = np.rad2deg(np.arccos(abs(x2-x1) / length))
            if angle > 45:
                continue
            lines_list.append([(x1,y1),(x2,y2), length])

        lines_list.sort(key = lambda x: x[0][1])
        l1 = lines_list[0]
        l2 = lines_list[-1]
        # if max(l1[0][1], l1[1][1]) < image.shape[0]/4 and min(l2[0][1], l2[1][1]) > image.shape[0]*3/4:
        a1 = (l1[0][1] - l1[1][1]) / (l1[0][0] - l1[1][0] + 1e-6)
        b1 = l1[0][1] - a1* l1[0][0]
        p11 = (0, int(b1))
        p21 = (image.shape[1], int(a1*image.shape[1]+b1))

        a2 = (l2[0][1] - l2[1][1]) / (l2[0][0] - l2[1][0] + 1e-6)
        b2 = l2[0][1] - a2* l2[0][0]
        p12 = (0, int(b2))
        p22 = (image.shape[1], int(a2*image.shape[1]+b2))

        
        src = np.array([p11, p21, p22, p12]).astype(np.float32)
        dst = np.array([
                [0, 0],
                [image.shape[1] - 1, 0],
                [image.shape[1] - 1, image.shape[0] - 1],
                [0, image.shape[0] - 1]], dtype = "float32")
        matrix = cv2.getPerspectiveTransform(src, dst)
        image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)                    
    # draw contours on the original image
    image_copy = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (0, 255, 0)
        (x,y,w,h) = cv2.boundingRect(contours[i])
        if w < thresh.shape[1]//2 or h < thresh.shape[0]//2:
            continue
        boundary = hull_list[i].squeeze()
        cv2.drawContours(image_copy, hull_list, i, color)

    gray = cv2.cvtColor(image_copy,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    kernel = np.ones((2,2),np.uint8)
    thresh = cv2.dilate(thresh,kernel,iterations = 2)


    lines = cv2.HoughLinesP(
                thresh, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=50, # Min number of votes for valid line
                minLineLength=40, # Min allowed length of line
                maxLineGap=10 # Max allowed gap between line for joining them
                )

    # Iterate over points
    if lines is not None:
        lines_list =[]
        for points in lines:
            # Extracted points nested in the list
            x1,y1,x2,y2=points[0]
            length = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            angle = np.rad2deg(np.arccos(abs(x2-x1) / length))
            if angle < 45:
                continue
            lines_list.append([(x1,y1),(x2,y2), length])

        lines_list.sort(key = lambda x: x[0][0])
        l1 = lines_list[0]
        l2 = lines_list[-1]
        if max(l1[0][0], l1[1][0]) < image.shape[1]/4 and min(l2[0][0], l2[1][0]) > image.shape[1]*3/4:
            a1 = (l1[0][1] - l1[1][1]) / (l1[0][0] - l1[1][0] + 1e-6)
            b1 = l1[0][1] - a1* l1[0][0]
            p11 = (int(-b1/a1), 0)
            p21 = (int((image.shape[0] - b1)/a1), image.shape[0])

            a2 = (l2[0][1] - l2[1][1]) / (l2[0][0] - l2[1][0] + 1e-6)
            b2 = l2[0][1] - a2* l2[0][0]
            p12 = (int(-b2/a2), 0)
            p22 = (int((image.shape[0] - b2)/a2), image.shape[0])
        else:
            p11, p12, p22, p21 = np.array([
                [0, 0],
                [image.shape[1] - 1, 0],
                [image.shape[1] - 1, image.shape[0] - 1],
                [0, image.shape[0] - 1]], dtype = "float32")

        src = np.array([p11, p12, p22, p21]).astype(np.float32)
        dst = np.array([
                [0, 0],
                [image.shape[1] - 1, 0],
                [image.shape[1] - 1, image.shape[0] - 1],
                [0, image.shape[0] - 1]], dtype = "float32")
        matrix = cv2.getPerspectiveTransform(src, dst)
        image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))
    return image

def count(img):
    h, w, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    tmp = np.full_like(gray, 0)
    gray = gray[h//5:4*h//5, :]

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, thresh = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,19,2)

    thresh = cv2.bitwise_not(thresh)

    tmp = np.sum(thresh, axis=1)
    tmp = (tmp - np.min(tmp)) / np.max(tmp)
    search = np.where(tmp > 0.5)[0]
    assert len(search) != 0
    first_line = search[0]
    last_line = search[-1]
    middle_line = (first_line + last_line) // 2

    kernel = np.ones((2,2),np.uint8)
    thresh = cv2.erode(thresh,kernel,iterations = 2)

    label_thresh = measure.label(thresh, connectivity = 2)
    outlier = np.unique(label_thresh[np.where(label_thresh[:, 0] != 0)[0], 0])
    outlier2 = np.unique(label_thresh[np.where(label_thresh[:, -1] != 0)[0], -1])
    
    for o in outlier:
        thresh[label_thresh==o] = 0
    for o in outlier2:
        thresh[label_thresh==o] = 0
    outlier = np.unique(label_thresh[-1, np.where(label_thresh[-1, :] != 0)[0]])
    for o in outlier:
        thresh[label_thresh==o] = 0

    kernel = np.ones((2,2),np.uint8)
    morpho = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    morpho = cv2.morphologyEx(morpho, cv2.MORPH_CLOSE, kernel)
    black_key = morpho[first_line:last_line, :]

    label = measure.label(black_key, connectivity = 2)

    unique_label = np.unique(label[black_key.shape[0]//2, :])[1:]

    numBlackKey = len(unique_label)

    check = None
    idx = -1
    dis = []
    for i, pixel in enumerate(black_key[black_key.shape[0]//2]):
        if pixel == 255:
            if check == -1:
                check = 1
            if check == 0:
                dis.append(i-idx-1)
                check = 1
            idx = i
        else:
            if check is None:
                check = -1
            if check == 1:
                check = 0
    new_dis = np.array(dis)
    new_dis.sort()
    
    new_dis = new_dis.reshape((-1, 1))
    kmeans = KMeans(n_clusters=2, random_state=42).fit(new_dis)
    label = kmeans.labels_
    new_dis = new_dis.reshape((1, -1))[0]
    upper = []
    lower = []
    cluster0 = new_dis[label==0]
    cluster1 = new_dis[label==1]
    if np.mean(cluster0) > np.mean(cluster1):
        upper = cluster0
        lower = cluster1
    else:
        upper = cluster1
        lower = cluster0

    numWhileKey = len(upper)*2 + len(lower)
    # left
    a = 0
    for i, v in enumerate(dis):
        if v in upper:
            if i == 0:
                a+=1
            break
        else:
            a+=1
    if np.count_nonzero(black_key[:, 0]==0) != 0:
        numWhileKey += a

    # right
    a = 0
    for i, v in enumerate(dis[::-1]):
        if v in upper:
            if i == 0:
                a+=1
            break
        else:
            a+=1
    if np.count_nonzero(black_key[:, -1]==0) != 0:
        numWhileKey += a

    # cv2.imshow("image", gray)
    # cv2.imshow("image2", morpho)
    # cv2.imshow("image4", black_key)
    # cv2.imshow("image1", thresh)
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return numBlackKey, numWhileKey