import sys
import time

from ui.tracker_wrapper import TrackerWrapper
from ltr.dataset.my_dataset import MyDataset
from pathlib import Path
import cv2


def run_video(video_path, dataset, tracker):
    vid = cv2.VideoCapture(video_path)
    pathlib_video_path = Path(video_path)
    init = True
    info = dataset.get_sequence_info(dataset.sequence_list.index(pathlib_video_path.stem))
    bboxes = info['bbox'].numpy()
    prev_frame_time = 0
    while True:
        ret, frame = vid.read()
        if init:
            r = cv2.selectROI('frame', frame)

            init = False
            init_state = [r[0], r[1], r[2], r[3]]
            # init_state = [int(init_bbox[0]), int(init_bbox[1]), int(init_bbox[2]), int(init_bbox[3])]
        else:
            init_state = None
        # Capture the video frame
        # by frame

        if not ret:
            break
        frame_disp = tracker.track(frame, init_state)
        font = cv2.FONT_HERSHEY_SIMPLEX
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(frame_disp, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    assert len(sys.argv) == 3
    video_path = sys.argv[1]
    tracker_name = sys.argv[2]

    dataset = MyDataset()
    tracker = TrackerWrapper(tracker_name)
    # tracker = None
    run_video(video_path, dataset, tracker)

