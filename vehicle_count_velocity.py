import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
# 
import moviepy.editor as mpy
import traceback

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')


# tt3B
calcLine1 = [ [(488, 536), (644, 536)], [(357, 634), (524, 634)] ]  #Down [(from(x,y), to(x,y)]
calcLine2 = [ [(645, 536), (778, 536)], [(525, 634), (704, 634)] ]  #Down [(from(x,y), to(x,y)]
calcLine3 = [ [(779, 536), (915, 536)], [(705, 634), (886, 634)] ]  #Down [(from(x,y), to(x,y)] 886 -> 900
calcLine4 = [ [(1144, 622), (1321, 622)], [(1100, 517), (1233, 517)] ] #Up [(from(x,y), to(x,y)]
calcLine5 = [ [(1322, 622), (1514, 622)], [(1234, 517), (1364, 516)] ] #Up [(from(x,y), to(x,y)]
calcLine6 = [ [(1515, 622), (1697, 622)], [(1365, 516), (1508, 516)] ] #Up [(from(x,y), to(x,y)]
calcRange_y = 50   # length of Y (for up or down of the line)

def cross_CalculateLine(i, t, Lane2Lv, ID_to_T1T2, track_id, now_LABEL, last_CENTROID, now_CENTROID, count_Car, count_Truck, count_Bus, count_Others):
    UP, UP2, UP3, DOWN, DOWN2, DOWN3 = False,False,False,False,False,False
    calculateLine1 = calcLine1[i]
    calculateLine2 = calcLine2[i]
    calculateLine3 = calcLine3[i]
    calculateLine4 = calcLine4[i]
    calculateLine5 = calcLine5[i]
    calculateLine6 = calcLine6[i]

    # print("id , now_LABEL , last_CENTROID, now_CENTROID:{}".format([ track_id, now_LABEL , last_CENTROID, now_CENTROID] ) )
    # print("DOWN", DOWN, DOWN2, DOWN3, "UP", UP, UP2, UP3)
    if (last_CENTROID[1]>calculateLine4[0][1]) and (now_CENTROID[1]<=calculateLine4[0][1]) and (now_CENTROID[0]>=calculateLine4[0][0]) and (now_CENTROID[0]<=calculateLine4[1][0]):
        UP = True
        if(now_LABEL=="truck"):
            count_Truck[4] += 1
        elif(now_LABEL=="car"):
            count_Car[4] += 1
        elif(now_LABEL=="bus"):
            count_Bus[4] += 1
        else:
            count_Others[4] += 1
    if (last_CENTROID[1]>calculateLine5[0][1]) and (now_CENTROID[1]<=calculateLine5[0][1]) and (now_CENTROID[0]>=calculateLine5[0][0]) and (now_CENTROID[0]<=calculateLine5[1][0]):
        UP2 = True
        if(now_LABEL=="truck"):
            count_Truck[5] += 1
        elif(now_LABEL=="car"):
            count_Car[5] += 1
        elif(now_LABEL=="bus"):
            count_Bus[5] += 1
        else:
            count_Others[5] += 1
    if (last_CENTROID[1]>calculateLine6[0][1]) and (now_CENTROID[1]<=calculateLine6[0][1]) and (now_CENTROID[0]>=calculateLine6[0][0]) and (now_CENTROID[0]<=calculateLine6[1][0]):
        UP3 = True
        if(now_LABEL=="truck"):
            count_Truck[6] += 1
        elif(now_LABEL=="car"):
            count_Car[6] += 1
        elif(now_LABEL=="bus"):
            count_Bus[6] += 1
        else:
            count_Others[6] += 1
    if (last_CENTROID[1]<calculateLine1[0][1]) and (now_CENTROID[1]>=calculateLine1[0][1]) and (now_CENTROID[0]>=calculateLine1[0][0]) and (now_CENTROID[0]<=calculateLine1[1][0]):
        #print("last_CENTROIDS[id][1], calculateLine1[0][1], now_CENTROIDS[now_id][1]:{}".format((last_CENTROIDS[id], calculateLine1[0][1], now_CENTROIDS[now_id])))
        DOWN = True
        if(now_LABEL=="truck"):
            count_Truck[1] += 1
        elif(now_LABEL=="car"):
            count_Car[1] += 1
        elif(now_LABEL=="bus"):
            count_Bus[1] += 1
        else:
            count_Others[1] += 1
    if (last_CENTROID[1]<calculateLine2[0][1]) and (now_CENTROID[1]>=calculateLine2[0][1]) and (now_CENTROID[0]>=calculateLine2[0][0]) and (now_CENTROID[0]<=calculateLine2[1][0]):
        #print("last_CENTROIDS[id][1], calculateLine2[0][1], now_CENTROIDS[now_id][1]:{}".format((last_CENTROIDS[id], calculateLine2[0][1], now_CENTROIDS[now_id])))
        DOWN2 = True
        if(now_LABEL=="truck"):
            count_Truck[2] += 1
        elif(now_LABEL=="car"):
            count_Car[2] += 1
        elif(now_LABEL=="bus"):
            count_Bus[2] += 1
        else:
            count_Others[2] += 1
    if (last_CENTROID[1]<calculateLine3[0][1]) and (now_CENTROID[1]>=calculateLine3[0][1]) and (now_CENTROID[0]>=calculateLine3[0][0]) and (now_CENTROID[0]<=calculateLine3[1][0]):
        #print("last_CENTROIDS[id][1], calculateLine3[0][1], now_CENTROIDS[now_id][1]:{}".format((last_CENTROIDS[id], calculateLine3[0][1], now_CENTROIDS[now_id])))
        DOWN3 = True
        if(now_LABEL=="truck"):
            count_Truck[3] += 1
        elif(now_LABEL=="car"):
            count_Car[3] += 1
        elif(now_LABEL=="bus"):
            count_Bus[3] += 1
        else:
            count_Others[3] += 1
    if DOWN or DOWN2 or DOWN3 or UP or UP2 or UP3:
        # print([track_id, i, t])
        if i == 0:
            ID_to_T1T2[track_id] = [t, -1, -1]
        elif i == 1:
            if track_id in ID_to_T1T2:
                ID_to_T1T2[track_id][1] = t
                t1 = ID_to_T1T2[track_id][0]
                ds = t - t1
                vms = 30/ds
                vkmh = vms * 3.6
                ID_to_T1T2[track_id][2] = vkmh
                if(vkmh<20):
                    level = 1
                elif (vkmh>=20) and (vkmh<40):
                    level = 2
                elif (vkmh>=40) and (vkmh<60):
                    level = 3
                elif (vkmh>=60) and (vkmh<80):
                    level = 4
                else:
                    level = 5

                if UP  == True:
                    Lane2Lv[4] = level
                elif UP2   == True:
                    Lane2Lv[5] = level
                elif UP3   == True:
                    Lane2Lv[6] = level
                elif DOWN  == True:
                    Lane2Lv[1] = level
                elif DOWN2 == True:
                    Lane2Lv[2] = level
                elif DOWN3 == True:
                    Lane2Lv[3] = level
                else:
                    print("error")

                print([track_id, "t2 - t1", t - t1, "m/s", vms, "km/h", vkmh, "level:", level])
        else:
            ID_to_T1T2[track_id] = [-1, -1, -1]

def draw_CalculateLine(frame, i, Lane2Lv):
    calculateLine1 = calcLine1[i]
    calculateLine2 = calcLine2[i]
    calculateLine3 = calcLine3[i]
    calculateLine4 = calcLine4[i]
    calculateLine5 = calcLine5[i]
    calculateLine6 = calcLine6[i]
    color = []
    for k in range(7):
        color.append( (0, 0, 0) )
    color[1] = (0, 0, 155)
    color[2] = (0, 0, 205)
    color[3] = (0, 0, 255)
    color[4] = (255, 0, 0)
    color[5] = (205, 0, 0)
    color[6] = (155, 0, 0)

    level2color = [(0,0,0), (200, 0, 200), (255, 0, 0), (240, 120, 0) , (255, 255, 0), (0, 200, 0)]
    for k in range(1,7):
        if Lane2Lv[k] > 0:
            color[k] = level2color[Lane2Lv[k]]

    cv2.line(frame, (calculateLine1[0][0],calculateLine1[0][1]), (calculateLine1[1][0],calculateLine1[1][1]), color[1], 2)
    cv2.line(frame, (calculateLine2[0][0],calculateLine2[0][1]), (calculateLine2[1][0],calculateLine2[1][1]), color[2], 2)
    cv2.line(frame, (calculateLine3[0][0],calculateLine3[0][1]), (calculateLine3[1][0],calculateLine3[1][1]), color[3], 2)
    cv2.line(frame, (calculateLine4[0][0],calculateLine4[0][1]), (calculateLine4[1][0],calculateLine4[1][1]), color[4], 2)
    cv2.line(frame, (calculateLine5[0][0],calculateLine5[0][1]), (calculateLine5[1][0],calculateLine5[1][1]), color[5], 2)
    cv2.line(frame, (calculateLine6[0][0],calculateLine6[0][1]), (calculateLine6[1][0],calculateLine6[1][1]), color[6], 2)

    return frame

def bbox2Centroid(bbox):
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    return (int(x+(w/2)), int(y+(h/2)))

def printText(img, count_Car, count_Truck, count_Bus, count_Others):
    y1 = 30
    y2 = 55
    y3 = 80
    y4 = 105

    cv2.putText(img, "Car:", (250, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 155), 1)
    cv2.putText(img, "Truck:", (250, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 155), 1)
    cv2.putText(img, "Bus:", (250, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 155), 1)
    cv2.putText(img, "Others:", (250, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 155), 1)
    cv2.putText(img, str(count_Car[1]), (360, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Truck[1]), (360, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Bus[1]), (360, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Others[1]), (360, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(img, "Car:", (500, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 205), 1)
    cv2.putText(img, "Truck:", (500, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 205), 1)
    cv2.putText(img, "Bus:", (500, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 205), 1)
    cv2.putText(img, "Others:", (500, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 205), 1)
    cv2.putText(img, str(count_Car[2]), (610, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Truck[2]), (610, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Bus[2]), (610, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Others[2]), (610, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(img, "Car:", (750, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(img, "Truck:", (750, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(img, "Bus:", (750, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(img, "Others:", (750, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(img, str(count_Car[3]), (860, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Truck[3]), (860, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Bus[3]), (860, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Others[3]), (860, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(img, "Car:", (1000, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(img, "Truck:", (1000, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(img, "Bus:", (1000, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(img, "Others:", (1000, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    cv2.putText(img, str(count_Car[4]), (1110, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Truck[4]), (1110, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Bus[4]), (1110, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Others[4]), (1110, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(img, "Car:", (1250, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (205, 0, 0), 1)
    cv2.putText(img, "Truck:", (1250, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (205, 0, 0), 1)
    cv2.putText(img, "Bus:", (1250, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (205, 0, 0), 1)
    cv2.putText(img, "Others:", (1250, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (205, 0, 0), 1)
    cv2.putText(img, str(count_Car[5]), (1360, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Truck[5]), (1360, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Bus[5]), (1360, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Others[5]), (1360, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(img, "Car:", (1500, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (155, 0, 0), 1)
    cv2.putText(img, "Truck:", (1500, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (155, 0, 0), 1)
    cv2.putText(img, "Bus:", (1500, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (155, 0, 0), 1)
    cv2.putText(img, "Others:", (1500, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (155, 0, 0), 1)
    cv2.putText(img, str(count_Car[6]), (1610, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Truck[6]), (1610, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Bus[6]), (1610, y3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, str(count_Others[6]), (1610, y4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    return img


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    # tt3B
    count_Car = [0,0,0,0,0,0,0]
    count_Truck = [0,0,0,0,0,0,0]
    count_Bus = [0,0,0,0,0,0,0]
    count_Others = [0,0,0,0,0,0,0]
    count_T2Car = [0,0,0,0,0,0,0]
    count_T2Truck = [0,0,0,0,0,0,0]
    count_T2Bus = [0,0,0,0,0,0,0]
    count_T2Others = [0,0,0,0,0,0,0]
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    # mpy
    ID_to_T1T2 = {}
    try:
        vid = mpy.VideoFileClip(video_path)
        frame_num = 0
        Lane2Lv = []
        for k in range(7):
            Lane2Lv.append(0)
        for timestamp, raw_img in vid.iter_frames(with_times=True):
            frame_num += 1;
            frame = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(raw_img)
            # print([timestamp, frame_num])

            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )

            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]
            
            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)
            
            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())
            
            # custom allowed classes (uncomment line below to customize tracker for only people)
            #allowed_classes = ['person']
            
            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)
            if FLAGS.count:
                cv2.putText(frame, "#Obj: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                # print("#Obj: {}".format(count))

            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)
            
            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
            
            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
            
            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       
            
            stepa = {}
            for track in tracker.tracks:
                # print("(pre-upd)Tracker.track_id: {},.to_tlwh: {}, Class: {}".format(str(track.track_id), str(track.to_tlwh()), class_name))
                # print("stepa[]:{}".format([track.track_id, track.to_tlwh()]))
                stepa[track.track_id] = track.to_tlwh()
            # print("stepa:{}".format(str(stepa)))
            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            # update tracks
            draw_CalculateLine(frame, 0, Lane2Lv)
            draw_CalculateLine(frame, 1, Lane2Lv)
            for track in tracker.tracks:
                # tt3B:
                if track.track_id in stepa:
                    #print("track existed in prev fr:{}".format([track.track_id,bbox2Centroid(stepa.get(track.track_id)) ] ) )
                    #print("cur fr:{}".format([ track.track_id , bbox2Centroid( track.to_tlbr() ) ] ) )
                    last_CENTROID = bbox2Centroid(stepa.get(track.track_id))
                    now_CENTROID = bbox2Centroid( track.to_tlwh() )
                    now_LABEL = track.get_class()
                    t = timestamp
                    # i, now_LABEL, last_CENTROID, now_CENTROID, count_Car, count_Truck, count_Bus, count_Others
                    cross_CalculateLine(0, t, Lane2Lv, ID_to_T1T2, track.track_id, now_LABEL, last_CENTROID, now_CENTROID, count_Car, count_Truck, count_Bus, count_Others)
                    cross_CalculateLine(1, t, Lane2Lv, ID_to_T1T2, track.track_id, now_LABEL, last_CENTROID, now_CENTROID, count_T2Car, count_T2Truck, count_T2Bus, count_T2Others)

                    # print("DOWN aft", DOWN, DOWN2, DOWN3, "UP", UP, UP2, UP3)
                # print("(aft-upd)Tracker.track_id: {},.to_tlwh: {}, Class: {}".format(str(track.track_id), str(track.to_tlwh()), class_name))
                # tt3B
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()

            # calc velocity
                v = "n/a"
                if track.track_id in ID_to_T1T2:
                    if ID_to_T1T2[track.track_id][1] > 0 :
                        v = str("%.2f" % round(ID_to_T1T2[track.track_id][2],2) )
            # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id)+ "-" +v,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            
            # if enable info flag then print details about each track
                if FLAGS.info:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            
            frame = printText(frame, count_Car, count_Truck, count_Bus, count_Others)
            cv2.putText(frame, "fr#: {}".format(frame_num), (5, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)

            # calculate frames per second of running detections
            fps = 1.0 / (time.time() - start_time)
            # print("FPS: %.2f" % fps)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if not FLAGS.dont_show:
                cv2.imshow("Output Video", result)
            
            # if output flag is set, save video file
            if FLAGS.output:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except Exception as e:
        print('Video has ended or failed, try a different video format!')
        print(e)
        traceback.print_exc()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
