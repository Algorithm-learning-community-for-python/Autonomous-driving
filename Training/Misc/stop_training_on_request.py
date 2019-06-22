import sys, select
from keras.callbacks import EarlyStopping

class StopTrainingOnInput(EarlyStopping):
    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()

                
        print('press "y" then enter within 5 seconds to stop training')
        i, o, e = select.select( [sys.stdin], [], [], 5 )

        if (i):
            answer =sys.stdin.readline().strip()
            print("You entered ", answer )
            if answer == "y":
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)
