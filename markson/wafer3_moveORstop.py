import cv2
import numpy as np
import time
from queue import Queue

class NKWAFER:
    def __init__(self, rect, tick):
        self.rect = rect
        self.time = Queue()
        self.time.put(tick)
    def update_time(self, tick):
        self.time.put(tick)
    def update_rect(self, rect):
        self.rect = rect
    def get_center(self):
        return int((self.rect[0] + self.rect[2])/2), int((self.rect[1] + self.rect[3])/2)
    def get_size(self):
        return (self.rect[2] - self.rect[0]) * (self.rect[3] - self.rect[1])
    

pointL = np.array([[874, 476], [1454, 455], [1554, 600], [1569, 794], [1548, 832], [1411, 555], [881, 591]], np.int32)
pointR = np.array([[255, 366], [1010, 320], [1010, 480], [327, 509], [191, 867], [161, 837], [187, 535]], np.int32)
workArea = [(844,1550), (163, 1030)]
ignoreArea = [pointL, pointR]
waferSize = [10, 15000]
colors = [(201, 174, 255), (240, 167, 0), (0, 255, 255), (76, 178, 34), (0, 0, 255)]
found_cols = [(195, 195, 195), (255, 0, 255)]
areaIdx = 1 # 0: left view  1:right view
warning_event = Queue()
nkwafer_queue= Queue()

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

def set_area(img):
    cv2.line(img, (workArea[areaIdx][0], 0), (workArea[areaIdx][0], 1080), (29, 230, 181), 3)
    cv2.line(img, (workArea[areaIdx][1], 0), (workArea[areaIdx][1], 1080), (29, 230, 181), 3)
    cv2.polylines(img, pts=[ignoreArea[areaIdx]], isClosed=True, color=(0, 0, 255), thickness=3)

def sop_estimate(img, det, names):
    set_area(img)
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

        #分類所有物件
        if deck_found == False and names[c] == 'deck':
            deck_found = True
            deck.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
        elif pen_found == False and names[c] == 'pen':
            pen_found = True
            pen.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
        elif names[c] == 'wafer':
            inbox.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
        elif names[c] == 'hand':
            hand.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
        elif names[c] == 'WAFER':
            nkwafer.append((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
        cv2.rectangle(img, p1, p2, colors[c], thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(img, "Deck", (50, 100), 0, 1, found_cols[int(deck_found)], thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(img, "Pen", (50, 130), 0, 1, found_cols[int(pen_found)], thickness=3, lineType=cv2.LINE_AA)
    if deck_found == True and pen_found == True:
        if calculate_iou(deck[0], pen[0]) > 0:
            pen_on_deck = True
    cv2.putText(img, "Pen on Deck", (50, 160), 0, 1, found_cols[int(pen_on_deck)], thickness=3, lineType=cv2.LINE_AA)
    if len(nkwafer) == 0:
        return

    return
    
    #判斷開始
    timestamp = time.time()
    for i in range(len(nkwafer)-1, -1, -1):
        size = area(nkwafer[i][0], nkwafer[i][1], nkwafer[i][2], nkwafer[i][3])
        if size > waferSize[areaIdx]: #未被遮蔽
            if(nkwafer_queue.qsize() == 0):
                new = NKWAFER(nkwafer[i], timestamp)
                nkwafer_queue.put(new)
                print("first one")
                nkwafer.pop(i)
                continue
            else:
                for _ in range(nkwafer_queue.qsize()):
                    wx, wy = int((nkwafer[i][0] + nkwafer[i][2])/2), int((nkwafer[i][1] + nkwafer[i][3])/2)
                    cell = wafer_queue.get()
                    fx, fy = cell.get_center()
                    dis = distance(wx, wy, fx, fy)
                    #print("distance = " + str(dis))
                    if dis < 10: #同一個
                        print("same one")
                        cell.update_time(timestamp)
                        cell.update_rect(nkwafer[i])
                        cv2.rectangle(img, (int(nkwafer[i][0]), int(nkwafer[i][1])), (int(nkwafer[i][2]), int(nkwafer[i][3])), (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
                        nkwafer.pop(i)
                        break
                    elif dis >= 10 and dis <=30: #移動中
                        print("moving...")
                        cell.update_time(timestamp)
                        cell.update_rect(nkwafer[i])

                        color = (0, 0, 255)
                        if pen_found == True: #找到筆
                            pen_cx, pen_cy = center(int(pen[0][0]), int(pen[0][1]), int(pen[0][2]), int(pen[0][3]))
                            waf_cx, waf_cy = center(int(nkwafer[i][0]), int(nkwafer[i][1]), int(nkwafer[i][2]), int(nkwafer[i][3]))
                            ddis1 = distance(pen_cx, pen_cy, waf_cx, waf_cy)
                            print("pen distance = " + str(ddis1))
                            if distance(pen_cx, pen_cy, waf_cx, waf_cy) < 220: #筆到晶圓的距離
                                color = (0, 255, 0)
                        elif pen_on_deck == False: #沒有找到筆但是也沒放在筆架上
                            color = (0, 255, 0)
                        cv2.rectangle(img, (int(nkwafer[i][0]), int(nkwafer[i][1])), (int(nkwafer[i][2]), int(nkwafer[i][3])), color, thickness=3, lineType=cv2.LINE_AA)
                    else:
                        new = NKWAFER(nkwafer[i], timestamp)
                        nkwafer_queue.put(new)
                        print("new wafer")
                        nkwafer.pop(i)
        else: #面積太小，不處理
            cv2.rectangle(img, (int(nkwafer[i][0]), int(nkwafer[i][1])), (int(nkwafer[i][2]), int(nkwafer[i][3])), (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
            nkwafer.pop(i)
    
    #繪製在box中的wafer外框
    for i in range(0, len(inbox)):
        cv2.rectangle(img, (int(inbox[i][0]), int(inbox[i][1])), (int(inbox[i][2]), int(inbox[i][3])), (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

    #for _ in range(nkwafer_queue.qsize()):
    #    each = wafer_queue.get()
    #    time_to_die = each.
    
                     


        
    
    
