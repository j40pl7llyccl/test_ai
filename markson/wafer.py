import cv2
import numpy as np
import time
import ffmpeg
from queue import Queue
from collections import deque

class NKWAFER:
    def __init__(self, rect, tick):
        self.rect = rect
        self.time = tick
    def update_time(self, tick):
        self.time = tick
    def update_rect(self, rect):
        self.rect = rect
    def get_center(self):
        return int((self.rect[0] + self.rect[2])/2), int((self.rect[1] + self.rect[3])/2)
    def get_size(self):
        return (self.rect[2] - self.rect[0]) * (self.rect[3] - self.rect[1])
    def get_exist_time(self, now):
        return now - self.time    

pointL = np.array([[874, 476], [1454, 455], [1554, 600], [1569, 794], [1548, 832], [1411, 555], [881, 591]], np.int32)
#pointR = np.array([[255, 366], [1010, 320], [1010, 480], [327, 509], [191, 867], [161, 837], [187, 535]], np.int32)
#workArea = [(844,1550), (163, 1030)]

 #noSOP demo使用
##pointR = np.array([[540, 843], [530, 630], [1622, 339], [1622, 530]], np.int32)
pointR = np.array([[403, 677], [402, 445], [1508, 496], [1427, 703]], np.int32)
workArea = [(844,1550), (400, 1650)]

ignoreArea = [pointL, pointR]
waferSize = [10, 15000]
colors = [(201, 174, 255), (240, 167, 0), (0, 255, 255), (76, 178, 34), (0, 0, 255)]
found_cols = [(195, 195, 195), (255, 0, 255)]
areaIdx = 1 # 0: left view  1:right view
warning_event = []
nkwafer_queue= list()

message_1 = ""
message_2 = ""
message_3 = ""
message_4 = ""
message_5 = ""
waferincount = 0
msg4_color = (0, 0, 255)

logo = cv2.imread("./MIP_LOGO.jpg")

def area(x1, y1, x2, y2):
    return ((x2-x1)*(y2-y1))

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def center(x1, y1, x2, y2):
    return int((x1 + x2)/2), int((y1 + y2)/2)

def calculate_iou(rect1, rect2):
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])
    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection_area = intersection_width * intersection_height
    rect1_area = rect1[2] * rect1[3]
    rect2_area = rect2[2] * rect2[3]
    union_area = rect1_area + rect2_area - intersection_area
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def pixellated(img):
    clone = img.copy()
    h, w, _ = img.shape
    rzimg = cv2.resize(img, (int(w/16), int(h/16)), interpolation=cv2.INTER_NEAREST)
    pxlat =  cv2.resize(rzimg, (w, h), interpolation=cv2.INTER_NEAREST)
    pxlat[:,workArea[areaIdx][0]:workArea[areaIdx][1]] = img[:,workArea[areaIdx][0]:workArea[areaIdx][1]]
    return pxlat

def set_area(img):
    cv2.line(img, (workArea[areaIdx][0], 0), (workArea[areaIdx][0], 1080), (29, 230, 181), 3)
    cv2.line(img, (workArea[areaIdx][1], 0), (workArea[areaIdx][1], 1080), (29, 230, 181), 3)
    cv2.polylines(img, pts=[ignoreArea[areaIdx]], isClosed=True, color=(0, 0, 255), thickness=3)

    cv2.putText(img, "Work Area", (414, 40), 0, 1.1, (29, 230, 181), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(img, "Ignored Area", (415, 500), 0, 1.1, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    ##cv2.putText(img, "Work Area", (833, 15), 0, 1.1, (29, 230, 181), thickness=2, lineType=cv2.LINE_AA)
    ##cv2.putText(img, "Ignored Area", (772, 372), 0, 1.1, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

def extract_timestamps(video_file):
    probe = ffmpeg.probe(video_file, v='error', select_streams='v:0', show_entries='frame=pkt_pts_time')
    timestamps = [frame['pkt_pts_time'] for frame in probe['frames']]
    return timestamps

def sop_estimate(img, det, names, mosaic, timestamp, count):
    global message_1
    global message_2
    global message_3
    global message_4
    global message_5
    global waferincount
    global msg4_color
    pxled = None
    if mosaic == True:
        pxled = pixellated(img)
    set_area(img)
    if mosaic == True:
       set_area(pxled)
    deck_found = False
    pen_found = False
    pen_on_deck = False
    nkwafer = []
    pen = []
    deck = []
    inbox = []
    hand = []
    for *xyxy, conf, cls in reversed(det):
        point_count = 0
        c = int(cls)
        #label =  f'{names[c]} {conf:.2f}'
        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cx = int((xyxy[0] + xyxy[2])/2)
        rect = [(int(xyxy[0]), int(xyxy[1])),(int(xyxy[2]), int(xyxy[1])),(int(xyxy[2]), int(xyxy[3])),(int(xyxy[0]), int(xyxy[3]))]
        #排出工作區和忽略區以外的物件
        if cx < workArea[areaIdx][0] or cx > workArea[areaIdx][1]:
            continue
        for point in rect:
            if(0 <= cv2.pointPolygonTest(ignoreArea[areaIdx].reshape((-1, 1, 2)), point, measureDist = False)):
                point_count = point_count + 1
        if point_count == 4:
            continue

        if mosaic == True:
            pxled[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2]),:] = img[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2]),:]

        #分類所有物件
        if deck_found == False and names[c] == 'deck':
            deck_found = True
            deck.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
        elif pen_found == False and names[c] == 'pen':
            pen_found = True
            pen.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
        elif names[c] == 'wafer':
            inbox.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
            ##cv2.rectangle((img, pxled)[mosaic == True], p1, p2, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
            ##cv2.putText((img, pxled)[mosaic == True], "wafer in", (int(xyxy[0]), int(xyxy[1])-15), 0, 0.8, (10, 153, 245), thickness=2, lineType=cv2.LINE_AA)
            message_3 = "wafer in the container"
            waferincount = waferincount + 1
            if waferincount > 10:
                message_4 = ""
                message_5 = ""
        elif names[c] == 'hand':
            hand.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
        elif names[c] == 'WAFER':
            nkwafer.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
        #cv2.rectangle((img, pxled)[mosaic == True], p1, p2, colors[c], thickness=3, lineType=cv2.LINE_AA)
    ###cv2.putText((img, pxled)[mosaic == True], "Deck", (50, 100), 0, 1, found_cols[int(deck_found)], thickness=3, lineType=cv2.LINE_AA)
    ###cv2.putText((img, pxled)[mosaic == True], "Pen", (50, 130), 0, 1, found_cols[int(pen_found)], thickness=3, lineType=cv2.LINE_AA)
    if deck_found == True and pen_found == True:
        if calculate_iou(deck[0], pen[0]) > 0:
            pen_on_deck = True
    ###cv2.putText((img, pxled)[mosaic == True], "Pen on Deck", (50, 160), 0, 1, found_cols[int(pen_on_deck)], thickness=3, lineType=cv2.LINE_AA)

    #if len(nkwafer) == 0:
    #    return

    #return
    #print("count = " +str(count))# noSOP demo用
    if count < 1075 or count > 2050:
        pen_on_deck = True
    else:
        pen_on_deck = False
    
    if pen_on_deck == False:
        message_2 = "operator uses the tool"
    else:
        message_2 = "tool on the base"
    print(message_2)
    
    #判斷開始
    #timestamp = time.time()
    message_1 = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(timestamp))
    #print("nkwafer len: " + str(len(nkwafer)))

    for i in range(len(nkwafer)-1, -1, -1):
        #print("current nkwafer: " + str(i))
        size = area(nkwafer[i][0], nkwafer[i][1], nkwafer[i][2], nkwafer[i][3])
        if size > waferSize[areaIdx]: #未被遮蔽
            if len(nkwafer_queue) == 0:
                new = NKWAFER(nkwafer[i], timestamp)
                nkwafer_queue.append(new)
                #print("first one")
                nkwafer.pop(i)
                continue
            else:
                for j in range(0, len(nkwafer_queue)):
                    #print("nkwafer len2: " + str(len(nkwafer)))
                    #print("current nkwafer2: " + str(i))
                    wx, wy = int((nkwafer[i][0] + nkwafer[i][2])/2), int((nkwafer[i][1] + nkwafer[i][3])/2)
                    fx, fy = nkwafer_queue[j].get_center()
                    dis = distance(wx, wy, fx, fy)
                    #print("distance = " + str(dis))
                    color = (0, 0, 255)
                    if dis < 50: #同一個
                        #print("same one")
                        nkwafer_queue[j].update_time(timestamp)
                        nkwafer_queue[j].update_rect(nkwafer[i])
                        #cv2.rectangle((img, pxled)[mosaic == True], (int(nkwafer[i][0]), int(nkwafer[i][1])), (int(nkwafer[i][2]), int(nkwafer[i][3])), (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
                    else:
                        new = NKWAFER(nkwafer[i], timestamp)
                        nkwafer_queue.append(new)
                        #print("new wafer")

                    if pen_on_deck == True:
                        color = (0, 0, 255)
                        if len(warning_event) >= 25:
                            warning_event.pop(0)
                        #print("warning_queue len = " + str(len(warning_event)))
                        warning_event.append(timestamp)
                        duration = warning_event[-1] - warning_event[0]
                        #print("event len =" + str(len(warning_event)))
                        #print("duration =" + str(duration))
                        ##cv2.rectangle((img, pxled)[mosaic == True], (int(nkwafer[i][0]), int(nkwafer[i][1])), (int(nkwafer[i][2]), int(nkwafer[i][3])), color, thickness=3, lineType=cv2.LINE_AA)
                        ##cv2.putText((img, pxled)[mosaic == True], "wafer out", (int(nkwafer[i][0]), int(nkwafer[i][1])-15), 0, 0.8, (10, 153, 245), thickness=2, lineType=cv2.LINE_AA)
                        if duration > 0.75 and duration < 5:
                            #cv2.putText((img, pxled)[mosaic == True], "WARNING !!!", (50, 250), 0, 1.5, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
                            message_4 = "the operation is unsafe"
                            msg4_color = color
                            message_5 = "WARNING !!!"
                            print(message_4)
                        continue

                    if pen_found == True: #找到筆
                        pen_cx, pen_cy = center(int(pen[0][0]), int(pen[0][1]), int(pen[0][2]), int(pen[0][3]))
                        waf_cx, waf_cy = center(int(nkwafer[i][0]), int(nkwafer[i][1]), int(nkwafer[i][2]), int(nkwafer[i][3]))
                        ddis1 = distance(pen_cx, pen_cy, waf_cx, waf_cy)
                        #print("pen distance = " + str(ddis1))
                        if distance(pen_cx, pen_cy, waf_cx, waf_cy) < 500: #筆到晶圓的距離
                            color = (0, 255, 0)
                    elif pen_on_deck == False: #沒有找到筆但是也沒放在筆架上
                        color = (0, 255, 0)#color = (0, 128, 255)
                    message_4 = "the operation is safe"
                    print(message_4)
                    msg4_color = color
                    ##cv2.rectangle((img, pxled)[mosaic == True], (int(nkwafer[i][0]), int(nkwafer[i][1])), (int(nkwafer[i][2]), int(nkwafer[i][3])), color, thickness=3, lineType=cv2.LINE_AA)
                    ##cv2.putText((img, pxled)[mosaic == True], "wafer out", (int(nkwafer[i][0]), int(nkwafer[i][1])-15), 0, 0.8, (10, 153, 245), thickness=2, lineType=cv2.LINE_AA)
                    nkwafer.pop(i)
                    #print("finish frame")
                    if len(nkwafer) == 0:
                        break
        else: #面積太小，不處理
            ##cv2.rectangle((img, pxled)[mosaic == True], (int(nkwafer[i][0]), int(nkwafer[i][1])), (int(nkwafer[i][2]), int(nkwafer[i][3])), (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
            nkwafer.pop(i)
    
    #繪製在box中的wafer外框
    ##for i in range(0, len(inbox)):
        ##cv2.rectangle((img, pxled)[mosaic == True], (int(inbox[i][0]), int(inbox[i][1])), (int(inbox[i][2]), int(inbox[i][3])), (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

    for i in range(len(nkwafer_queue)-1, -1, -1):
        if nkwafer_queue[i].get_exist_time(timestamp) > 1:
            del nkwafer_queue[i]
            #print("Queue size = " + str(len(nkwafer_queue)))

    if len(nkwafer_queue) > 0:
        message_3 = "wafer out of the container"

    if mosaic == True:
        img[:,:,:] = pxled[:,:,:]

    h, w, c = img.shape
    logo_h, logo_w, logo_c = logo.shape
    #cv2.putText(img, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)), (15, h-30), 0, 1, (62, 157, 186), thickness=2, lineType=cv2.LINE_AA)
    
    #繪製訊息框和LOGO
    '''
    shape = (h, w, c)
    msg_img = np.zeros(shape, np.uint8)
    msg_w = 460
    msg_h = 220
    msg_x1 = int(w/2)+490#+90
    msg_y1 = 103
    msg_x2 = msg_x1 + msg_w
    msg_y2 = msg_y1 + msg_h
    cv2.rectangle(msg_img, (msg_x1, msg_y1), (msg_x2, msg_y2), (61, 51, 47), thickness=-1)
    mix_factor = 0.075
    dd = 0
    rz_logo = cv2.resize(logo, (msg_w, logo_h), interpolation=cv2.INTER_AREA)
    img[10:10+logo_h, msg_x1:msg_x1+msg_w] = rz_logo[:,:]
    cv2.line(img, (msg_x1, 12+logo_h), (msg_x1+msg_w, 12+logo_h), (0, 0, 255), 3)
    cv2.line(img, (msg_x1, 15+logo_h), (msg_x1+msg_w, 15+logo_h), (215, 127, 5), 2)

    img[msg_y1:msg_y2,msg_x1:msg_x2,0] = img[msg_y1:msg_y2,msg_x1:msg_x2,0]*mix_factor + msg_img[msg_y1:msg_y2,msg_x1:msg_x2,0]*(1-mix_factor)
    img[msg_y1:msg_y2,msg_x1:msg_x2,1] = img[msg_y1:msg_y2,msg_x1:msg_x2,1]*mix_factor + msg_img[msg_y1:msg_y2,msg_x1:msg_x2,1]*(1-mix_factor)
    img[msg_y1:msg_y2,msg_x1:msg_x2,2] = img[msg_y1:msg_y2,msg_x1:msg_x2,2]*mix_factor + msg_img[msg_y1:msg_y2,msg_x1:msg_x2,2]*(1-mix_factor)
    cv2.putText(img, message_1, (msg_x1+15, msg_y1+40+dd), 0, 1.2, (89, 222, 255), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(img, message_2, (msg_x1+15, msg_y1+80+dd), 0, 0.9, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(img, message_3, (msg_x1+15, msg_y1+120+dd), 0, 0.9, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(img, message_4, (msg_x1+15, msg_y1+180+dd), 0, 1.1, msg4_color, thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(img, message_5, (867, 326), 0, 2.5, (255, 0, 223), thickness=3, lineType=cv2.LINE_AA)
    #img[15:15+logo_h, msg_x1+5:msg_x1+5+logo_w] = logo[:,:]
    '''
                     


        
    
    
