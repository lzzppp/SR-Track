#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = ["vis"]


def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=15): 
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5 
    pts= [] 
    for i in np.arange(0,dist,gap): 
        r=i/dist 
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5) 
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5) 
        p = (x,y) 
        pts.append(p) 

    if style=='dotted': 
        for p in pts: 
            cv2.circle(img,p,thickness,color,-1) 
    else: 
        s=pts[0] 
        e=pts[0] 
        i=0 
        for p in pts: 
            s=e 
            e=p 
        if i%2==1: 
            cv2.line(img,s,e,color,thickness) 
        i+=1
            
def drawpoly(img,pts,color,thickness=1,style='dotted',): 
    s=pts[0] 
    e=pts[0] 
    pts.append(pts.pop(0)) 
    for p in pts: 
        s=e 
        e=p 
        drawline(img,s,e,color,thickness,style) 

def drawrect(img,pt1,pt2,color,thickness=1,style='dotted'): 
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
    drawpoly(img,pts,color,thickness,style) 

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, predict_tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    # im_h, im_w = im.shape[:2]
    
    text_scale = 20
    text_thickness = 4
    line_thickness = 4

    highlight_object_ids = [31] # [1, 31, 35, 37]
    # highlight_object_ids = [30] # [1, 30, 35, 36, 37, 43, 46]
    
    colors = {obj_id: get_color(abs(obj_id)) for obj_id in obj_ids}
    
    colors[1] = (255, 106, 106) # "red"
    colors[5] = (255, 127, 36)
    colors[31] = (255, 255, 0) # "yellow"
    colors[35] = (50, 205, 50) # "lime"
    colors[37] = (255, 127, 36) # "magenta"
    colors[36] = (255, 255, 255) #
    if 43 in colors:
        colors[43] = (colors[43][0] + 10, colors[43][1] + 10, colors[43][2] + 10)
    colors[46] = (255, 222, 173) #

    # if frame_id <= 10:
    # for i, tlwh in enumerate(predict_tlwhs):
    #     x1, y1, w, h = tlwh
    #     intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
    #     obj_id = int(obj_ids[i])

    #     # if obj_id not in highlight_object_ids:
    #     #     continue
        
    #     color = colors[abs(obj_id)]
    #     r, g, b = color
    #     drawrect(im, (intbox[0], intbox[1]), (intbox[2], intbox[3]), (b,g,r), thickness=line_thickness - 1, style='dotted')

    im = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)) 
    draw = ImageDraw.Draw(im)
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        # if obj_id not in highlight_object_ids:
        #     continue
        # if int(obj_id) == 30:
        #     id_text = '{}'.format(int(obj_id) + 1)
        # else:
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        # color = get_color(abs(obj_id))
        color = colors[abs(obj_id)]
        if abs(obj_id) == 30 and frame_id > 4:
            color = (255, 0, 0)
        
        fonttype = ImageFont.truetype('./pretrained/Gemelli.ttf', text_scale)
        label_size = draw.textsize(id_text, font=fonttype)
        text_origin = np.array([int(x1), int(y1 + 0.1 * label_size[1])])
        
        draw.rectangle(intbox,outline=color,width=line_thickness)
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color)
        draw.text(tuple(text_origin), id_text, fill=(0, 0, 0), font=fonttype)
    
    img = cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
        
    return img


def plot_detection(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, x2, y2 = tlwh
        intbox = tuple(map(int, (x1, y1, x2, y2)))
        obj_id = int(obj_ids[i])
        score = round(float(scores[i]), 3)
        id_text = '{}'.format(int(obj_id))
        score_text = '{}'.format(score)
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, score_text, ((intbox[0] + intbox[2])//2, (intbox[1] + intbox[3])//2), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
