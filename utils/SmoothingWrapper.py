from utils.kalman_filter import KalmanBoxTracker


class SmoothingWrapper:
    def __init__(self):
        ...


class KalmanWrapper(SmoothingWrapper):
    def __init__(self, init_x):
        super().__init__()
        self.kf = KalmanBoxTracker(init_x)