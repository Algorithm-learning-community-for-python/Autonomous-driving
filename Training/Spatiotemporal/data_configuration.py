
from keras.optimizers import Adam, SGD

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
        lr = 0.00012
        args = {
            "loss": "mse",
            "optimizer": Adam(lr),
            "lr": lr,
            "metrics": None,
            "epochs": 20,
            "batch_size": 16,
        }
        self.train_conf = TrainConf(**args)
        self.model_type = "Spatiotemporal"
        self.images_path = "/Updated_images/"

        self.train_valid_split = 0.2
        self.bottom_crop = 0
        self.top_crop = 165
        self.filter_input = True
        self.filtering_degree = 0.8  # 0 = remove none, 1 = remove all
        self.filter_threshold = 0.1

        self.filtering_degree_speed = 0.8
        self.filter_threshold_speed = 0.001
        self.recordings_path = "/Measurments/modified_recording.csv"
        self.folder_index = -1
        self.add_noise = False
        self.skip_steps = 1
        self.step_size_training = 3
        """
        Step size testing is dependent on:
        - sensor_tick during recording (defined in the recorder class)
        - step_size during training
        - fps during testing
        - sensor_tick(0.1) #(or average_fps_trainging(2.43))
        and it can be calculated by the following formula:
        if sensor_tick:
            step_size_testing = fps * sensor_tick * step_size_training
        if average_fps:
            step_size_testing = average_fps_trainging * fps
        example step_size_training=3, sensor_tick = 0.1, fps=60  ==> step_size_testing=12
        """ 
        self.step_size_testing = 9
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
            "ohe_speed_limit",
            "TL",
            "Brake",
            "Speed"
        ]

        self.input_data = {
            "Direction": True,
            "Speed": True,
            "speed_limit": False,
            "ohe_speed_limit": True,
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
            "ohe_speed_limit": False,
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
            "Image": [66, 200, 3], #"Image": [345-(self.top_crop+self.bottom_crop), 460, 3],
            "Direction": [7],
            "Speed": [1],
            "speed_limit": [1], 
            "ohe_speed_limit": [11],
            "TL_state": [3],
            "Output": 1,
            "Sequence_length": 4,
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
        self.loss_weights={
            'output_Throttle': 1.,
            'output_Brake': 1.,
            'output_Steer': 2.
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
        self.sl_categories = [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1,
        ]

        