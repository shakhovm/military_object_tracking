import json
import cv2
import time
from helper_functions import Helper


def bbox_displayer(json_path, video_path, frames, save_path=None):
    with open(json_path, 'r', encoding='utf-8') as f:
        if json_path.endswith("json"):
            results = json.load(f)
        else:
            results = []
            for line in f.readlines():
                line = list(map(lambda x: int(float(x)), line.strip().split(',')))
                results.append([[line[0], line[1], line[0] + line[2], line[1] + line[3]]])


    cap = cv2.VideoCapture(video_path)
    if save_path:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), 35., (w, h))
    helper = Helper()
    i = 0
    new_results = []
    # areas = []
    while True:
        if i >= frames:
            break
        if not save_path:
            time.sleep(0.01)
        ret, frame = cap.read()
        if not ret:
            break
        try:
            if isinstance(results, dict):
                if str(i) in results:
                    bbox = results[str(i)]
                    helper.display_bboxes(frame, bbox)
            else:
                if len(results[i]) > 0:
                    bbox = results[i]
                    helper.display_bboxes(frame, bbox)
        except Exception as e:
            print(e)
            pass
        cv2.putText(frame, str(i), (100, 100), 2, 2, (0, 255, 255))

        if save_path:
            writer.write(frame)
        else:
            cv2.imshow("Frame", helper.resize(frame, 0.5))
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        i += 1

    if save_path:
        writer.release()

if __name__ == '__main__':
    import sys
   
    #file_name = "2021-02-18-4"
    video_path = sys.argv[1]
    frames = sys.maxsize if len(sys.argv) == 3 else int(sys.argv[3])
    bbox_displayer(sys.argv[2], video_path, frames, None)
