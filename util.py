from collections import Counter
import numpy as np
import cv2

def find_items(img, stations=True):
    """Finds stations or trains in an iamge

    Args:
        img (RGB-image): screenshot of the game
        stations (bool, optional): if True, finds stations. if False, finds trains.

    Returns:
        list: list of trains or stations
        int: number of stations found
    """
    n_stations = 0
    items = []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    (height, width, _) = img.shape
    size = height*width
    _, thresh_blurred = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    contours,_ = cv2.findContours(thresh_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_b = np.zeros(img.shape[:2], dtype='uint8')
    cv2.fillPoly(mask_b, contours, 255)
    mask_w = np.ones(img.shape, dtype='uint8')*255
    cv2.fillPoly(mask_w, contours, 0)
    masked = cv2.bitwise_and(img, img, mask = mask_b)
    masked_w = masked+mask_w

    for contour in contours:
        if cv2.contourArea(contour)/size < 0.001:
            continue
        if cv2.contourArea(contour)/size < 0.005:
            ctype = 'train'
            padding = int(width/20)
        else:
            ctype = 'station'
            padding = int(width/15)
            n_stations += 1
        if stations and ctype == 'train':
            continue
        if not stations and ctype == 'station':
            continue

        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            continue
        c_x = int(moments["m10"] / moments["m00"])
        c_y = int(moments["m01"] / moments["m00"])
        cropped = masked_w[c_y-padding:c_y+padding, c_x-padding:c_x+padding]
        color_name = find_color(cropped)

        if ctype == 'train' and color_name == 'white':
            edges = cv2.Canny(cropped ,50, 150)
            lines = cv2.HoughLinesP(edges, rho = 1,theta = 1*np.pi/180, threshold = 40, minLineLength = 30, maxLineGap = 0)
            if lines is not None:
                continue

        if ctype == 'station':
            items.append({'x': c_x, 'y': c_y, 'type': 'station', 'color': color_name})
        else:
            items.append({'x': c_x, 'y': c_y, 'color': color_name})
    return items, n_stations


def find_color(obj):
    """Returns the color name of the train or station

    Args:
        obj (BGR-image): image of the train or station

    Returns:
        string: color name
    """

    hsv = cv2.cvtColor(np.array(obj), cv2.COLOR_BGR2HSV)
    hsv = hsv.reshape(obj.shape[0]*obj.shape[1],3).astype('float')
    hsv[(hsv[:,0]==0)&(hsv[:,1]==0)&(hsv[:,2]==255)] = np.nan

    cnt_red = len(np.where((hsv[:,0]>160)&(hsv[:,0]<170))[0])
    cnt_yellow = len(np.where((hsv[:,0]>=20)&(hsv[:,0]<30))[0])
    cnt_green = len(np.where((hsv[:,0]>=55)&(hsv[:,0]<65))[0])
    cnt_violet = len(np.where((hsv[:,0]>=140)&(hsv[:,0]<150))[0])
    cnt_blue = len(np.where((hsv[:,0]>=95)&(hsv[:,0]<105))[0])

    colors = {
        'red': cnt_red,
        'green': cnt_green,
        'blue': cnt_blue,
        'yellow': cnt_yellow,
        'violet': cnt_violet
    }

    if np.nanmean(hsv[:,1])<50:
        if np.nanmean(hsv[:,2])>200:
            return 'white'
        return 'black'
    if np.nanmean(hsv[:,1])<150 and np.nanstd(hsv[:,1])>95 and np.nanstd(hsv[:,2])<40:
        return 'red-white'

    k = Counter(colors)
    main_colors = k.most_common(2)

    if main_colors[1][1] != 0 and main_colors[0][1]/main_colors[1][1] < 4:
        color1 = main_colors[0][0]+'-'+main_colors[1][0]
        color2 = main_colors[1][0]+'-'+main_colors[0][0]
        if color1 in ['yellow-violet', 'red-blue', 'green-violet']:
            return color1
        if color2 in ['yellow-violet', 'red-blue', 'green-violet']:
            return color2

    if main_colors[0][0] in ['blue', 'green', 'yellow']:
        if np.nanstd(hsv[:,2])>55:
            return main_colors[0][0]+'-black'

    return main_colors[0][0]
