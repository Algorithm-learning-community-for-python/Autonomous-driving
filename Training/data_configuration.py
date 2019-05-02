
from keras.optimizers import Adam


class TrainConf:
    def __init__(self, **entries):
        self.loss = None
        self.batch_size = None
        self.lr = 0
        self.optimizer = None
        self.metrics = None
        self.epochs = 0

        self.__dict__.update(entries)

class Config:
    def __init__(self):

        self.available_columns = [
            "Throttle",
            "Reverse",
            "at_TL",
            "frame",
            "Manual",
            "Hand brake",
            "Steer",
            "Direction",
            "Gear",
            "TL_state",
            "speed_limit",
            "TL",
            "Brake",
            "Speed"
        ]

        self.input_data = {
            "Direction": True,
            "Speed": False,
            "speed_limit": False,
            "TL_state": False,
            "Throttle": False,
            "Reverse": False,
            "at_TL": False,
            "frame": False,
            "Manual": False,
            "Hand brake": False,
            "Steer": False,
            "Gear": False,
            "TL": False,
            "Brake": False,
        }

        self.direction_categories = [
            "RoadOption.VOID",
            "RoadOption.LEFT",
            "RoadOption.RIGHT",
            "RoadOption.STRAIGHT",
            "RoadOption.LANEFOLLOW",
            "RoadOption.CHANGELANELEFT",
            "RoadOption.CHANGELANERIGHT"
        ]

        self.tl_categories = [
            "Green",
            "Yellow",
            "Red"
        ]

        self.input_size_data = {
            "Image": [140, 320, 3],
            "Direction": [7],
            "Speed": [1],
            "speed_limit": [1],
            "TL_state": [3],
            "Output": 1,
        }

        lr = 0.0001
        args = {
            "loss": "mse",
            "optimizer": Adam(lr),
            "lr": lr,
            "metrics": None,
            "epochs": 2,
            "batch_size": 32,
        }
        self.train_conf = TrainConf(**args)
        self.train_valid_split = 0.2
        self.bottom_crop = 115
        self.top_crop = 70