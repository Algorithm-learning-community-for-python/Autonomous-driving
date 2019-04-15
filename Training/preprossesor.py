import numpy as np
import cv2

class Preprocessor():
    def normalize(self, x, max_value=255, scew=0.5):
        return np.divide(x, max_value)-scew

    def crop_image(self, img):
        bottom_crop = 115
        top_crop = 510
        img = img[bottom_crop:top_crop, :, :]
        return img

    def brighten_image(self, image):
        brigthen_coff = np.random.uniform() + 0.5
        # Convert to HLS
        hls_img = np.array(cv2.cvtColor(image,cv2.COLOR_BGR2HSV), dtype = np.float64)
        ## scale pixel values up or down for channel 1(Lightness)
        ## The scaling is in the range +/- 20% of original value
        hls_img[:,:,1] = hls_img[:,:,1]*brigthen_coff 
        ##Set all values above 255 to 255
        hls_img[:,:,1][hls_img[:,:,1]>255]  = 255 
        # Convert back to BGR
        bgr_img = cv2.cvtColor(np.array(hls_img, dtype = np.uint8), cv2.COLOR_HSV2BGR) ## Conversion to BGR
        return bgr_img

    def remove_low_steering(self, files, steerings):
        filtered_files = []
        filtered_steerings = []

        for idx, steering in enumerate(steerings):
            if abs(steering) > 0.1:
                filtered_files.append(files[idx])
                filtered_steerings.append(steering)
            else:
                rand = np.random.randint(10)
                
                if rand < 7:
                    filtered_files.append(files[idx])
                    filtered_steerings.append(steering)

        print("Dropped {} rows with low steering".format(len(files) - len(filtered_files)))
        return filtered_files, filtered_steerings

    def add_shadow(self, img):
        top_x = 0
        bottom_x = len(img)
        top_y = int(320*np.random.uniform())
        bottom_y = int(320*np.random.uniform())

        hls = np.array(cv2.cvtColor(img,cv2.COLOR_BGR2HSV), dtype=np.float64) 

        mask = 0*hls[:,:,1]

        mesh = np.mgrid[0:len(img), 0:len(img[0])]
        x_mesh = mesh[0]
        y_mesh = mesh[1]

        mask[((x_mesh-top_x)*(bottom_y-top_y) - (bottom_x-top_x)*(y_mesh-top_y) >= 0)] = 1

        darken_coff = 0.5
        cond1 = mask==1
        cond2 = mask==0

        if np.random.uniform() >0.5:
            hls[:,:,1][cond1] = hls[:,:,1][cond1]*darken_coff
        else:
            hls[:,:,1][cond1] = hls[:,:,1][cond1]*darken_coff

        bgr_img = cv2.cvtColor(np.array(hls, dtype = np.uint8), cv2.COLOR_HSV2BGR) ## Conversion to BGR

        return bgr_img

    def shift_image(self, img,steering):
        num_rows, num_cols = img.shape[:2]
        transelation_range = 100
        translation_x = transelation_range * np.random.uniform() - transelation_range/2
        translation_y = transelation_range * np.random.uniform() / 2 - transelation_range/4
        new_steering = steering + translation_x / transelation_range*0.4
        translation_matrix = np.float32([ [1,0,translation_x], [0,1,translation_y] ])
        img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
        return img_translation, new_steering