import numpy as np

class Preprocessor():
        
    def normalize(self, x, max_value=255, scew=0.5):
        return np.divide(x, max_value)-scew
            
