import matplotlib.pyplot as plt

# TODO: Move methods from spatial
class Visualizer:

    @staticmethod
    def plot_training_results(self, trainer):
        # summarize history for accuracy
        plt.plot(trainer.history.history['acc'])
        plt.plot(trainer.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(trainer.history.history['loss'])
        plt.plot(trainer.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
