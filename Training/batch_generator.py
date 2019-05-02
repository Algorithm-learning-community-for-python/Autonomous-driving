import os
import pandas as pd

# TODO: FINISH

class KerasBatchGenerator(object):

    def __init__(self, num_steps, batch_size, skip_step=0):
        self.num_steps = num_steps
        self.batch_size = batch_size
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        self.current_recording = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step
        self.data_paths = []
        self.data_as_recordings = []
        self.folders = []

        self.fetch_measurements()

    def fetch_folders(self):
        for folder in os.listdir('../Training_data'):
            if folder == ".DS_Store" or folder == "store.h5":
                continue
            self.data_paths.append("../Training_data/" + folder)
            self.folders.append(folder)

        store = pd.HDFStore("../Training_data/store.h5")
        for folder in folders:
            df = store[folder]
            self.data_as_recordings.append(df)
        store.close()

    def get_new_recording(self):


    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                    self.get_new_recording()
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y