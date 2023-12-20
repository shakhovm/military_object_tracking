from argparse import Namespace
from collections import OrderedDict

from pytracking.evaluation import Tracker
from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper
import cv2


class TrackerWrapper:
    def __init__(self, tracker_name):
        TRACKERS = {
            "atom": ["atom", "default"],
            "dimp": ["dimp", "dimp50"],
            "tomp": ["tomp", "tomp50"]
        }
        tracker_param = TRACKERS[tracker_name]
        self.init_tracker = Tracker(tracker_param[0], tracker_param[1])
        # self.init_tracker = Tracker('atom', 'default')
        # self.init_tracker = Tracker('tomp', 'tomp50')
        self.tracker = self._init_tracker()
        self.main_params, self.info = self._init_params()
        self._tracker_disp_colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0),
                                4: (255, 255, 255), 5: (0, 0, 0), 6: (0, 255, 128),
                                7: (123, 123, 123), 8: (255, 128, 0), 9: (128, 0, 255)}

    def _init_tracker(self):
        params = self.init_tracker.get_parameters()

        debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.init_tracker.name
        params.param_name = self.init_tracker.parameter_name
        self.init_tracker._init_visdom(None, debug_)
        return MultiObjectWrapper(self.init_tracker.tracker_class, params,
                                  self.init_tracker.visdom, fast_load=True)

    def _init_params(self):
        class MainParams:
            def __init__(self):
                self.next_object_id = 1
                self.sequence_object_ids = []
                self.prev_output = OrderedDict()
                self.output_boxes = OrderedDict()
                self.frame_number = 0

        info = {'object_ids': [], 'init_object_ids': [], 'init_bbox': OrderedDict()}
        return MainParams(), info

    def track(self, frame, init_position=None):

        self.main_params.frame_number += 1
        frame_disp = frame.copy()

        info = OrderedDict()
        info['previous_output'] = self.main_params.prev_output

        if init_position:
            # r = cv2.selectROI(display_name, frame)
            r = init_position
            init_state = [r[0], r[1], r[2], r[3]]

            print(init_state)
            info['init_object_ids'] = [self.main_params.next_object_id, ]
            info['init_bbox'] = OrderedDict({1: init_state})
            self.main_params.sequence_object_ids.append(1)
            self.main_params.output_boxes[1] = [init_state, ]
            self.main_params.next_object_id += 1

        # if len(self.main_params.sequence_object_ids) > 0:
        info['sequence_object_ids'] = self.main_params.sequence_object_ids
        out = self.tracker.track(frame, info)
        self.main_params.prev_output = OrderedDict(out)

        if 'target_bbox' in out:
            for obj_id, state in out['target_bbox'].items():
                state = [int(s) for s in state]
                cv2.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                              self._tracker_disp_colors[obj_id], 5)
                # if save_results:
                self.main_params.output_boxes[obj_id].append(state)
        else:
            self.main_params.output_boxes[1].append([0, 0, 0, 0,])

            p_frame = frame
            # Put text
            font_color = (255, 255, 255)
            # msg = "Select target(s). Press 'r' to reset or 'q' to quit."
            # cv2.rectangle(frame_disp, (5, 5), (630, 40), (50, 50, 50), -1)
            # cv2.putText(frame_disp, msg, (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 2)
        return frame_disp

