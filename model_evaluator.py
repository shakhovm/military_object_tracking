import json
import os
import sys
import time

from ui.tracker_wrapper import TrackerWrapper
from ltr.dataset.my_dataset import MyDataset
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

from utils.kalman_filter import KalmanBoxTracker
from utils.phase_corr_smoothing import smooth_bbox


def wh2xy(bbox):
    return [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]


def xy2wh(bbox):
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


def int_wh2xy(bbox):
    return list(map(int, wh2xy(bbox)))


def int_xy2wh(bbox):
    return list(map(int, xy2wh(bbox)))


def iou(boxA: np.ndarray, boxB: np.ndarray, format="xyxy"):
    if format == "xyxy":
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = np.maximum(boxA[:, 0], boxB[:, 0])
        yA = np.maximum(boxA[:, 1], boxB[:, 1])
        xB = np.minimum(boxA[:, 2], boxB[:, 2])
        yB = np.minimum(boxA[:, 3], boxB[:, 3])
        # compute the area of intersection rectangle
        interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[:, 2] - boxA[:, 0] + 1) * (boxA[:, 3] - boxA[:, 1] + 1)
        boxBArea = (boxB[:, 2] - boxB[:, 0] + 1) * (boxB[:, 3] - boxB[:, 1] + 1)
    else:
        new_box_a = boxA.copy()
        new_box_b = boxB.copy()
        new_box_a[:, 2] += new_box_a[:, 0]
        new_box_a[:, 3] += new_box_a[:, 1]
        new_box_b[:, 2] += new_box_b[:, 0]
        new_box_b[:, 3] += new_box_b[:, 1]
        xA = np.maximum(new_box_a[:, 0], new_box_b[:, 0])
        yA = np.maximum(new_box_a[:, 1], new_box_b[:, 1])
        xB = np.minimum(new_box_a[:, 2], new_box_b[:, 2])
        yB = np.minimum(new_box_a[:, 3], new_box_b[:, 3])
        interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
        boxAArea = (boxA[:, 2] + 1) * (boxA[:, 3] + 1)
        boxBArea = (boxB[:, 2] + 1) * (boxB[:, 3] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def extract_rect(img, bbox):
    x, y, w, h = bbox
    return img[y: y + h, x: x + w]


def evaluate_data(sample_name, dataset, tracker, visualize=False):
    init = True
    seq_id = dataset.sequence_list.index(sample_name)
    info = dataset.get_sequence_info(seq_id)
    bboxes = info['bbox'].numpy()
    out_of_view = info['visible'].numpy()
    # bboxes[:, 2] += bboxes[:, 0]
    # bboxes[:, 3] += bboxes[:, 1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    init_bbox = bboxes[0]
    prev_frame_time = 0
    seq_path = dataset._get_sequence_path(seq_id)
    frames = range(len(os.listdir(seq_path)))
    fpses = []
    smoothed_boxes_phase = []
    smoothed_boxes_kalman = []
    predicted_boxes = []
    for frame_id, bbox in zip(frames, bboxes):

        img_path = os.path.join(seq_path, '{:08}.jpg'.format(frame_id + 1))
        frame = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if init:
            init_state = [int(init_bbox[0]), int(init_bbox[1]), int(init_bbox[2]), int(init_bbox[3])]

            new_bbox = [init_state[0], init_state[1],
                        init_state[2] + init_state[0],
                        init_state[3] + init_state[1]]
            kf = KalmanBoxTracker(new_bbox)
            prev_rect = frame[init_state[1]:init_state[1] + init_state[3],
                        init_state[0]: init_state[0] + init_state[2]]
            smoothed_bbox = init_state
            smoothed_boxes_phase.append(smoothed_bbox)
            smoothed_boxes_kalman.append(init_state)
            predicted_boxes.append(init_state)

            init = False
        else:
            init_state = None
        frame_disp = tracker.track(frame, init_state)
        if not init:
            new_bbox = kf.predict()[0]
            new_bbox = list(map(int, new_bbox))
            predicted_box = tracker.main_params.output_boxes[1][-1]
            transformed_box = [int(predicted_box[0]), int(predicted_box[1]),
                               int(predicted_box[0] + predicted_box[2]),
                               int(predicted_box[1] + predicted_box[3])]

            current_rect = frame[predicted_box[1]:predicted_box[1] + predicted_box[3],
                           predicted_box[0]: predicted_box[0] + predicted_box[2]]
            try:
                smoothed_bbox = smooth_bbox(
                    current_rect_with_noise=current_rect, prev_rect=prev_rect,
                    pred_bbox=predicted_box
                )
                smoothed_bbox = smoothed_bbox[0]
            except:
                smoothed_bbox = predicted_box
            smoothed_boxes_phase.append(smoothed_bbox)
            kf.update(transformed_box)
            smoothed_boxes_kalman.append(xy2wh(new_bbox))
            predicted_boxes.append(predicted_box)
            # tracker.main_params.output_boxes[1][-1] =
            # print(smoothed_bbox)
            prev_rect = extract_rect(frame, smoothed_bbox)
            # tracker.main_params.output_boxes[1][-1] = smoothed_bbox
            # cv2.imshow("Frame1", prev_rect)
            # prev_rect = current_rect
            smoothed_bbox = wh2xy(smoothed_bbox)



        # frame_disp = frame.copy()
        # print(bbox)
        # cv2.rectangle(frame_disp, tuple(bbox[:2]),
        #               tuple(bbox[2:]),  (255, 0, 0), 5)
        # time.sleep(0.1)
        # Display the resulting frame
        if visualize:
            new_frame_time = time.time()

            iou_score = float(iou(np.expand_dims(bbox, axis=0),
                                  np.array([tracker.main_params.output_boxes[1][-1]]), format="xywh"
                                  )[0]
                              )
            iou_score = round(iou_score, 4)
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fpses.append(fps)
            fps = str(fps)
            cv2.putText(frame_disp, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

            cv2.putText(frame_disp, str(iou_score), (7, 170), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame_disp, str(frame_id), (7, 250), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame_disp, "KALMAN", (570, 70), font, 3, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(frame_disp, "Actual", (570, 170), font,
                        3, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame_disp, "PHASE", (570, 270), font,
                        3, (255, 255, 0), 3, cv2.LINE_AA)

            cur_bbox = list(map(int, [bbox[0], bbox[1], bbox[2], bbox[3]]))
            cur_bbox[3] += cur_bbox[1]
            cur_bbox[2] += cur_bbox[0]
            cv2.rectangle(frame_disp, tuple(cur_bbox[:2]),
                          tuple(cur_bbox[2:]), (255, 0, 0), 5)

            cv2.rectangle(frame_disp, tuple(smoothed_bbox[:2]),
                          tuple(smoothed_bbox[2:]), (255, 255, 0), 5)

            cv2.rectangle(frame_disp, tuple(new_bbox[:2]),
                          tuple(new_bbox[2:]), (0, 0, 255), 5)

            cv2.imshow('frame', cv2.cvtColor(frame_disp, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if visualize:
        print(f"mean fps: {np.array(fpses).mean()}")
        cv2.destroyAllWindows()
    return np.array(tracker.main_params.output_boxes[1], dtype=np.float32), bboxes, {
        "kalman": smoothed_boxes_kalman,
        "phase": smoothed_boxes_phase,
        "predicted": predicted_boxes
    }


if __name__ == '__main__':
    video_path = sys.argv[1]
    vizdata = bool(int(sys.argv[2]))
    tracker_name = sys.argv[3]
    dataset = MyDataset()
    tracker = TrackerWrapper(tracker_name)
    # tracker = None
    # pred_boxes, target_bboxes = evaluate_data("006", dataset, tracker, visualize=vizdata)
    video_path = Path(video_path)
    pred_boxes, target_bboxes, dct = evaluate_data(video_path.stem, dataset, tracker, visualize=vizdata)
    pred_boxes = pred_boxes[1:]
    pred_boxes[:, 2] += pred_boxes[:, 0]
    pred_boxes[:, 3] += pred_boxes[:, 1]

    target_bboxes[:, 2] += target_bboxes[:, 0]
    target_bboxes[:, 3] += target_bboxes[:, 1]
    df = pd.DataFrame.from_dict({
        "pred_xl": pred_boxes[:, 0],
        "pred_yt": pred_boxes[:, 1],
        "pred_xr": pred_boxes[:, 2],
        "pred_yb": pred_boxes[:, 3],
        "target_xl": target_bboxes[:, 0],
        "target_yt": target_bboxes[:, 1],
        "target_xr": target_bboxes[:, 2],
        "target_yb": target_bboxes[:, 3],

    })
    df.to_csv(f"artifacts/trained-{tracker_name}-{video_path.stem}.csv")
    with open(f"artifacts/trained-{tracker_name}-{video_path.stem}.json", 'w') as f:
        json.dump(dct, f)
    print(f"-" * 100)
    print(f"mIoU: {iou(pred_boxes, target_bboxes).mean()}")

    print(f"-" * 100)
    # vid = cv2.VideoCapture(0)
