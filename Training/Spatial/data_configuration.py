
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

class Config(object):
    """ Contains settings used for Training and configuration """
    def __init__(self):
        lr = 0.0001
        args = {
            "loss": "mse",
            "optimizer": Adam(lr),
            "lr": lr,
            "metrics": None,
            "epochs": 30,
            "batch_size": 32,
        }
        self.train_conf = TrainConf(**args)
        self.train_valid_split = 0.2
        self.bottom_crop = 0 #115
        self.top_crop = 100
        self.filter_input = True
        self.filtering_degree = 0.5  # 0 = remove none, 1 = remove all
        self.recordings_path = "/Measurments/modified_recording.csv"
        self.folder_index = -1


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
            "Speed": True,
            "speed_limit": True,
            "TL_state": True,
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
        self.output_data = {
            "Direction": False,
            "Speed": False,
            "speed_limit": False,
            "TL_state": False,
            "Throttle": True,
            "Reverse": False,
            "at_TL": False,
            "frame": False,
            "Manual": False,
            "Hand brake": False,
            "Steer": True,
            "Gear": False,
            "TL": False,
            "Brake": True,
        }
        self.input_size_data = {
            "Image": [240-(self.top_crop+self.bottom_crop), 320, 3],
            "Direction": [7],
            "Speed": [1],
            "speed_limit": [1],
            "TL_state": [3],
            "Output": 1,
            "Sequence_length": 5,
        }
        self.output_size_data = {
            "Throttle": 1,
            "Brake": 1,
            "Steer": 1,
        }
        self.loss_functions = {
            "output_Throttle": "mse", #Might be better with binary_crossentropy
            "output_Brake": "mse",
            "output_Steer": "mse",
        }
        self.activation_functions = {
            "output_Throttle": None, #Might be better with binary_crossentropy
            "output_Brake": None,
            "output_Steer": None,
        }
        self.Sequence_length = 5
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

        