import argparse
import copy
import os
import time
from collections import Counter, deque
from datetime import datetime

import cv2
import face_alignment
import matplotlib as mpl
import matplotlib.collections as collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from imutils import face_utils
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import distance as dist
from scipy.special import softmax
from scipy.stats import entropy
from torchvision import transforms, utils

import util
from configs.image_cfg import _C as cfg
from dataset import make_data_loader
from dataset.transforms import build_transforms
from model.graph_net import Graph_Net
from model.overall_net import Net


def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # train_loader, test_loader, num_classes, label_template = make_data_loader(cfg)
    # emotion class
    label_template = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    pos_template   = ['Happy', 'Surprise']
    # neg_template   = ['Angry', 'Disgust', 'Fear', 'Sad']
    neg_template   = [ 'Fear', 'Sad']
    neutral_class  = 'Neutral'

    num_classes    = len(label_template)
    # define and load model
    # model_path = cfg.MODEL.SAVE_WEIGHT_PATH+'.pth'
    model_path = cfg.MODEL.SAVE_WEIGHT_PATH+'_cat.pth'
    model = Net(cfg, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # face detector
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

    (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
     
    f_transform = build_transforms(cfg, is_train=False, is_face=True)
    c_transform = build_transforms(cfg, is_train=False, is_face=False)

    # video path
    video_path = args.file_path
    cap = cv2.VideoCapture(video_path)
    f_stack = deque([])
    c_stack = deque([])
    seq_pred = deque([])
    points   = deque([])
    seq_arousal_val = deque([])

    long_seq_pred = deque([])
    long_long_seq_pred = deque([])
    long_seq_prob = deque([])
    long_seq_size = 48
    long_long_seq_size = 160
    window_size = 16
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    global_step = 0


    disorder_1  = []
    disorder_2  = []
    disorder_3  = []
    huge_change = []
    emotion_pred = []
    arousal_pred = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
                
            frame = cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
            # run model
            face, context, landmark = detFace(fa, frame)

            if landmark is None:
                continue
            collectBatch(face, f_transform, f_stack, window_size)
            collectBatch(context, c_transform, c_stack, window_size)

            xr_start = frame.shape[0] - 10
            xl_start = 10

            yr_start = 30
            yl_start = 30
            
            # define color
            warning_color = (0, 0, 255)    # red
            remind_color  = (81, 118, 237) # orange
            text_color    = (255, 128, 0)  # blue 

            cv2.putText(frame, "real-time emotion", (xl_start, yl_start), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 4)
            cv2.putText(frame, "state of disorder behavior", (xr_start, yr_start), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            xr_start += 160
            yl_start += 40
            yr_start += 40

            leyebrow = landmark[lBegin:lEnd]
            reyebrow = landmark[rBegin:rEnd]
            distq = eye_brow_distance(points,window_size,leyebrow[-1],reyebrow[0])
            seq_arousal_val.append(normalize_values(points,distq))
            
            if len(seq_arousal_val) > window_size:
                seq_arousal_val.popleft()

            if len(f_stack) == window_size:
                
                # prediction
                with torch.no_grad():
                    f_tensor = torch.stack(list(copy.deepcopy(f_stack))).to(device)
                    c_tensor = torch.stack(list(copy.deepcopy(c_stack))).to(device)

                    # output = model(f_tensor, c_tensor).mean(0)
                    # ema temporal update
                    output    = model(f_tensor, c_tensor)
                    tmp_step  = torch.arange(0, window_size).float().cuda() + 1
                    tmp_alpha = 1 - 1 / (tmp_step + 1)
                    output    = (tmp_alpha.unsqueeze(-1).expand_as(output) * output).mean(dim=0)
                    
                    pred_class = output.argmax()
                    pred_label = label_template[pred_class]
                    emotion_pred.append(pred_label)
                    # append value
                    arousal_value = np.mean(seq_arousal_val)
                    arousal_pred.append(arousal_value)
                    # constraint value
                    long_seq_pred.append(pred_label)
                    long_seq_prob.append(output.cpu().numpy())
                    if (len(long_seq_pred) > long_long_seq_size) and (len(long_seq_prob) > long_long_seq_size):
                        long_seq_pred.popleft()
                        long_seq_prob.popleft()

                    is_moody = detMoody(global_step, long_seq_pred, long_seq_prob, long_seq_size, frame)
                    if is_moody:
                        disorder_1.append(global_step)
                        cv2.putText(frame, "moody", (xr_start, yr_start), cv2.FONT_HERSHEY_SIMPLEX, 1, warning_color, 4)
                        yr_start += 40

                    if arousal_value > 0.75:
                        huge_change.append(global_step)
                        cv2.putText(frame, "high arousal", (xr_start, yr_start), cv2.FONT_HERSHEY_SIMPLEX, 1, remind_color, 4)
                        yr_start += 40

                    is_neg, info = detNeg(global_step, neg_template, neutral_class, long_seq_pred, long_seq_prob, long_long_seq_size, frame)
                    if is_neg:
                        if 'neutral' not in info:
                            disorder_2.append(global_step)
                        else:
                            disorder_3.append(global_step)
                        cv2.putText(frame, info, (xr_start, yr_start), cv2.FONT_HERSHEY_SIMPLEX, 1, remind_color, 4)
                        yr_start += 40

                        
                    cv2.putText(frame, 'emotion: ' + pred_label, (xl_start, yl_start), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 4)
                    yl_start += 40
                    cv2.putText(frame, 'arousal: {:.4f}'.format(arousal_value), (xl_start, yl_start), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 4)


                    # show frame
                    cv2.imshow('frame', frame)
                    global_step += 1
        else: 
            break

    cap.release()
    cv2.destroyAllWindows()
    print('show result')

    category_names = ['moody', 'neg', 'flat', 'arousal']
    results = {
        'moody':   build_plot('moody',  disorder_1,  global_step, long_seq_size, emotion_pred, 'red'),
        'neg':     build_plot('neg',    disorder_2,  global_step, long_long_seq_size , emotion_pred, 'coral'),
        'flat':    build_plot('flat',   disorder_3,  global_step, long_long_seq_size , emotion_pred, 'coral'),
        'arousal': build_plot('arousal',huge_change, global_step, window_size, arousal_pred, 'yellow'),
    }

def build_plot(title, disorder, global_step, size, emo_pred, color):
    print(disorder, global_step, size)
    
    idx    = []
    result = []
    for i in range(global_step):
        idx.append(i)
        if i in disorder:
            result.append(1)
            ## extend window
            # s = max(i - size, 0)
            # for j in range(s, i):
            #     result[j] = 1
        else:
            result.append(0)

    idx = np.array(idx)
    result = np.array(result)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)
    ax1.set_title("state")
    ax1.plot(idx, emo_pred, color='blue')

    ax2.set_title(title)
    ax2.plot(idx, result, color='black')
    ax2.axhline(0, color='black', lw=1)

    collection = collections.BrokenBarHCollection.span_where(
        idx, ymin=0, ymax=1, where=result > 0, facecolor=color, alpha=0.5)
    ax2.add_collection(collection)

    plt.savefig("./" + title + '.png')
    plt.show()

def detNeg(global_step, neg_template, neutral_class, seq_pred, seq_prob, min_seq_size, frame):

    # count prediction
    pred_num = len(seq_pred)
    seq_prob = np.array(copy.deepcopy(seq_prob))
    if pred_num == min_seq_size:


        pred_prob = np.mean(seq_prob, 0)
        pred_prob = softmax(pred_prob)

        count_pred_list, most_pred = label_counter(seq_pred)
        count_pred_list = np.array(count_pred_list, dtype=np.float) / sum(count_pred_list)
        
        # print(most_pred, entropy(count_pred_list), entropy(pred_prob))
        if most_pred[1] > (pred_num // 3) and entropy(count_pred_list) < 1. and entropy(pred_prob) < 1.:
            if most_pred[0] in neg_template: 
                print("stay in negative emotion (陷入絕望) =>", most_pred, 'entropy of count: {:.4f}'.format(entropy(count_pred_list)), 'entropy of prob: {:.4f}'.format(entropy(pred_prob)), global_step)
                return True, "stay in negative emotion"
            elif most_pred[0] == neutral_class: 
                print("flat affect (缺乏情緒反應) =>", most_pred, 'entropy of count: {:.4f}'.format(entropy(count_pred_list)), 'entropy of prob: {:.4f}'.format(entropy(pred_prob)), global_step)
                return True, "flat affect"
        return False, ''
    else:
        return False, ''
    
def detMoody(global_step, seq_pred, seq_prob, min_seq_size, frame):

    # emotion disorder 
    # short prediction => moody
    # idx = [i for i in range(160)]
    seq_prob = np.array(copy.deepcopy(seq_prob))
    seq_num = len(seq_pred)
    if seq_num >= min_seq_size:
        st = int(seq_num - min_seq_size)
        ed = int(st + min_seq_size)

        # select_prob = seq_prob[i for i in range(st, ed)]
        select_prob = []
        select_pred = []
        for i in range(st, ed):
            select_prob.append(seq_prob[i, :])
            select_pred.append(seq_pred[i])

        pred_prob = np.mean(select_prob, 0)
        pred_prob = softmax(pred_prob)

        # count prediction
        count_pred_list, most_pred = label_counter(select_pred)
        count_pred_list = np.array(count_pred_list, dtype=np.float) / sum(count_pred_list)

        if entropy(count_pred_list) > 1. and entropy(pred_prob) > 1. and len(count_pred_list) > 1 and most_pred[-1] <= min_seq_size // 3:
            print("moody (喜怒無常) =>", most_pred, 'entropy of count: {:.4f}'.format(entropy(count_pred_list)), 'entropy of prob: {:.4f}'.format(entropy(pred_prob)), global_step)
            return True
        else:
            return False

def paint_chinese_opencv(im,chinese,pos,color):
    img_PIL = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('simsun.ttc',25)
    fillColor = color #(255,0,0)
    position = pos #(100,100)
    # if not isinstance(chinese,unicode):
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position,chinese,font=font,fill=fillColor)
 
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    return img

def label_counter(pred, thres = 2):
    count_window_pred = Counter(pred)
    count_window_pred_list = []
    for key in count_window_pred:
        val = count_window_pred[key]
        # remove noise
        if val > thres:
            count_window_pred_list.append(val)
    return count_window_pred_list, count_window_pred.most_common(1)[0]


def prob(log_output):
    return np.exp(log_output)/sum(np.exp(log_output))

def eye_brow_distance(points, size, leye, reye):
    distq = dist.euclidean(leye,reye)
    points.append(int(distq))
    if len(points) > size:
        points.popleft()
    return distq

def normalize_values(points, disp):
    # print(points)
    normalized_value = abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
    arousal_value = np.exp(-(normalized_value))
    return arousal_value

def collectBatch(image, transform, stack, size):
    tensor = transform(Image.fromarray(image))
    stack.append(tensor)
    if len(stack) > size:
        stack.popleft()

def detFace(model, image):
    process_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    w, h, c = process_image.shape
    preds = model.get_landmarks(process_image)
    if preds != None and len(preds) > 0:
        max_face_area = 0
        max_face_info = [0, 0, 0, 0]
        max_landmark = None
        try:
            for i, landmark in enumerate(preds):
                        
                x1 = np.min(landmark[:, 0])
                y1 = np.min(landmark[:, 1])
                x2 = np.max(landmark[:, 0])
                y2 = np.max(landmark[:, 1])

                box_len = int((float(max(x2 - x1, y2 - y1)) / 2.) * 1.3)
                center_x = int(float(x1 + x2) / 2.)
                center_y = int(float(y1 + y2) / 2.)
                
                if box_len > max_face_area:
                    max_face_area = box_len
                    max_face_info = [center_x, center_y]
                    max_landmark = landmark

            sx = max_face_info[0] - max_face_area
            ex = max_face_info[0] + max_face_area
            sy = max_face_info[1] - max_face_area
            ey = max_face_info[1] + max_face_area
            face = process_image[sx:ex, sy:ey, :]
            wf, hf, cf = face.shape
            result = np.full((max_face_area * 2,max_face_area * 2, c), (0,0,0), dtype=np.uint8)
            xx = (max_face_area * 2 - wf) // 2
            yy = (max_face_area * 2 - hf) // 2
            result[xx:xx+wf, yy:yy+hf] = process_image[sx:ex, sy:ey, :]
            
            return result, process_image, landmark
        except Exception as e: 
            print(e)
            mid_w = w // 2
            mid_h = h // 2
            box_len = min(mid_h, mid_w) // 2
            return process_image[(mid_w - box_len):(mid_w + box_len), (mid_h - box_len):(mid_h + box_len), :], process_image, None
    else:
        return None, None, None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='scz inference')
    parser.add_argument('--file_path', default='./test.mp4', help='file path')
    args = parser.parse_args()
    main(args)