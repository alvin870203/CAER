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
from configs.AVEC_cfg import _C as avec_cfg
from configs.image_cfg import _C as caer_cfg
from dataset.transforms import build_transforms
from model.graph_net import Graph_Net
from model.overall_net import Net


def main(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = caer_cfg.MODEL.DEVICE_ID
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # train_loader, test_loader, num_classes, label_template = make_data_loader(cfg)
    # emotion class
    label_template = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    pos_template   = ['Happy', 'Surprise']
    neg_template   = ['Disgust', 'Fear', 'Sad']
    # neg_template   = [ 'Fear', 'Sad']
    neutral_class  = 'Neutral'
    depression_level = 45

    num_classes    = len(label_template)
    # define and load model
    # model_path = cfg.MODEL.SAVE_WEIGHT_PATH+'.pth'
    dmodel_path = caer_cfg.MODEL.SAVE_WEIGHT_PATH+'_dmodel.pth'
    emodel_path = caer_cfg.MODEL.SAVE_WEIGHT_PATH+'_emodel.pth'
    is_face = True
    is_context = True
    graph_mode = 'b'
    # define DNN 
    e_model  = Net(caer_cfg, len(label_template), is_face=is_face, is_context=is_context, mode=graph_mode).to(device)
    d_model  = Net(avec_cfg, depression_level+1, True, is_face=is_face, is_context=is_context, mode=graph_mode).to(device)
    
    e_model.load_state_dict(torch.load(emodel_path))
    d_model.load_state_dict(torch.load(dmodel_path))
    e_model.eval()
    d_model.eval()


    # face detector
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

    (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
     
    f_transform = build_transforms(caer_cfg, is_train=False, is_face=True)
    c_transform = build_transforms(caer_cfg, is_train=False, is_face=False)
    # video path
    video_path = args.file_path
    cap = cv2.VideoCapture(video_path)
    f_stack = deque([])
    c_stack = deque([])
    seq_pred = deque([])
    points   = deque([])
    seq_arousal_val = deque([])

    long_seq_pred = deque([])
    long_seq_d_pred = deque([])
    long_long_seq_pred = deque([])
    long_long_seq_d_pred = deque([])
    long_seq_prob = deque([])
    long_seq_size = 18
    long_long_seq_size = 36
    window_size = 6
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    
    global_step = 0


    disorder_1  = []
    disorder_2  = []
    disorder_3  = []
    disorder_4  = []
    huge_change = []
    emotion_pred = []
    arousal_pred = []
    depression_pred = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if global_step < 32:
            global_step += 1
            continue
            
        
        print('start detect', global_step)
        
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
            # xr_start += 160
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
                    e_output    = e_model(f_tensor, c_tensor)
                    d_output    = d_model(f_tensor, c_tensor, is_depression=True)

                    tmp_step  = torch.arange(0, window_size).float().cuda() + 1
                    tmp_alpha = 1 - 1 / (tmp_step + 1)
                    e_output  = (tmp_alpha.unsqueeze(-1).expand_as(e_output) * e_output).mean(dim=0)
                    d_output  = (tmp_alpha.unsqueeze(-1).expand_as(d_output) * d_output).mean(dim=0)

                    # print(d_output.shape, e_output.shape)
                    d_mse_pred = torch.sigmoid(d_output[-1]) * (depression_level)
                    d_dis_pred = torch.sigmoid(d_output[:-1]).argmax().clamp(min=0, max=depression_level)


                    pred_level = (d_mse_pred + d_dis_pred) / 2.
                    if pred_level > 20:
                        pred_class = e_output.topk(k=2)[-1][-1]
                        pred_label = label_template[pred_class]
                        if pred_label not in neg_template:
                            pred_class = e_output.argmax()
                            pred_label = label_template[pred_class]
                    else:
                        pred_class = e_output.argmax()
                        pred_label = label_template[pred_class]


                    # print(pred_level, pred_class, pred_label)
                    emotion_pred.append(pred_label)
                    depression_pred.append(pred_level.cpu().numpy())
                    # append value
                    arousal_value = np.mean(seq_arousal_val)
                    arousal_pred.append(arousal_value)
                    # constraint value
                    long_seq_pred.append(pred_label)
                    long_seq_prob.append(e_output.cpu().numpy())
                    long_seq_d_pred.append(pred_level.cpu().numpy())
                    if (len(long_seq_pred) > long_long_seq_size) and (len(long_seq_prob) > long_long_seq_size):
                        long_seq_pred.popleft()
                        long_seq_prob.popleft()
                        long_seq_d_pred.popleft()

                    is_moody = detMoody(global_step, long_seq_pred, long_seq_prob, long_seq_size, frame)
                    if is_moody:
                        disorder_1.append(global_step)
                        cv2.putText(frame, "moody (mania disorder)", (xr_start, yr_start), cv2.FONT_HERSHEY_SIMPLEX, 1, warning_color, 4)
                        yr_start += 40

                    # if arousal_value > 0.75:
                    #     huge_change.append(global_step)
                    #     cv2.putText(frame, "high arousal", (xr_start, yr_start), cv2.FONT_HERSHEY_SIMPLEX, 1, remind_color, 4)
                    #     yr_start += 40

                    # depression detect
                    avg_level = np.mean(np.array(long_seq_d_pred))

                    is_neg, info = detNeg(global_step, neg_template, neutral_class, long_seq_pred, long_seq_prob, long_long_seq_size, frame)
                    if is_neg:
                        if 'flat' not in info:
                            disorder_2.append(global_step)
                        else:
                            if avg_level > 20:
                                disorder_3.append(global_step)
                        cv2.putText(frame, info, (xr_start, yr_start), cv2.FONT_HERSHEY_SIMPLEX, 1, remind_color, 4)
                        yr_start += 40
                    
                    if avg_level > 20:
                        disorder_4.append(global_step)
                        if avg_level > 28:
                            cv2.putText(frame, 'Severe depression', (xr_start, yr_start), cv2.FONT_HERSHEY_SIMPLEX, 1, remind_color, 4)
                        else:
                            cv2.putText(frame, 'Moderate depression', (xr_start, yr_start), cv2.FONT_HERSHEY_SIMPLEX, 1, remind_color, 4)
                    yr_start += 40

                    cv2.putText(frame, 'emotion: ' + pred_label, (xl_start, yl_start), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 4)
                    yl_start += 40
                    cv2.putText(frame, 'depression: {:.4f}'.format(pred_level), (xl_start, yl_start), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 4)
                    # yl_start += 40
                    # cv2.putText(frame, 'arousal: {:.4f}'.format(arousal_value), (xl_start, yl_start), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 4)

                    # show frame
                    cv2.imshow('frame', frame)
                    global_step += 1

                    if global_step == 2000:
                        break
        else: 
            break

    cap.release()
    cv2.destroyAllWindows()
    print('show result')


    save_image_folder = './results/' + video_path.split('/')[-1].replace('.mp4','') + '/'
    if not os.path.isdir(save_image_folder):
        os.mkdir(save_image_folder)

    category_names = ['moody', 'neg', 'flat', 'arousal']
    results = {
        'moody':   build_plot(save_image_folder, 'moody (mania disorder)',  disorder_1,  global_step, long_seq_size, emotion_pred, depression_pred, 'red'),
        'neg':     build_plot(save_image_folder, 'stay in negative emotion (depression disorder)',    disorder_2,  global_step, long_long_seq_size , emotion_pred, depression_pred, 'green'),
        'flat':    build_plot(save_image_folder, 'flat affect (depression disorder)',   disorder_3,  global_step, long_long_seq_size , emotion_pred, depression_pred, 'coral'),
        'depression':    build_plot(save_image_folder, 'high depressive level (depression disorder)',   disorder_4,  global_step, long_long_seq_size , emotion_pred, depression_pred, 'blue'),
        # 'arousal': build_plot('arousal',huge_change, global_step, window_size, arousal_pred, 'yellow'),
    }

def build_plot(save_image_folder, title, disorder, global_step, size, emo_pred, dep_pred, color):
    print(disorder, global_step, size)
    
    idx    = []
    result = []
    np_len_emo = len(emo_pred)
    np_len_dep = len(dep_pred)
    append_emo = []
    append_dep = []
    for i in range(global_step):
        idx.append(i)

        if len(append_emo) < (global_step - len(emo_pred)):
            append_emo.append('N/A')
        if len(append_dep) < (global_step - len(dep_pred)):
            append_dep.append(0)
        
        if i in disorder:
            result.append(1)
            ## extend window
            # s = max(i - size, 0)
            # for j in range(s, i):
            #     result[j] = 1
        else:
            result.append(0)
    

    emo_pred = append_emo + emo_pred
    dep_pred = append_dep + dep_pred

    idx = np.array(idx)
    result = np.array(result)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=0.5)
    ax1.set_title("emotional state")
    ax1.plot(idx, emo_pred, color='blue', marker='o')

    ax2.set_title("depressive level")
    ax2.plot(idx, dep_pred, color='blue')

    ax3.set_title(title)
    ax3.plot(idx, result, color='black')
    ax3.axhline(0, color='black', lw=1)

    collection = collections.BrokenBarHCollection.span_where(
        idx, ymin=0, ymax=1, where=result > 0, facecolor=color, alpha=0.5)
    ax3.add_collection(collection)

    plt.savefig(save_image_folder + title + '.png')
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
        # if most_pred[1] > (pred_num // 3) and entropy(count_pred_list) < 1. and entropy(pred_prob) < 1.:
        if most_pred[1] > (pred_num // 5) and entropy(count_pred_list) < 0.9 and entropy(pred_prob) < 0.9:
            if most_pred[0] in neg_template: 
                # print("stay in negative emotion (陷入絕望) =>", most_pred, 'entropy of count: {:.4f}'.format(entropy(count_pred_list)), 'entropy of prob: {:.4f}'.format(entropy(pred_prob)), global_step)
                return True, "stay in negative emotion (depression disorder)"
            elif most_pred[0] == neutral_class: 
                # print("flat affect (缺乏情緒反應) =>", most_pred, 'entropy of count: {:.4f}'.format(entropy(count_pred_list)), 'entropy of prob: {:.4f}'.format(entropy(pred_prob)), global_step)
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
            # print("moody (喜怒無常) =>", most_pred, 'entropy of count: {:.4f}'.format(entropy(count_pred_list)), 'entropy of prob: {:.4f}'.format(entropy(pred_prob)), global_step)
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
    parser.add_argument('--file_path', default='./demo_video/test_2.mp4', help='file path')
    args = parser.parse_args()
    main(args)