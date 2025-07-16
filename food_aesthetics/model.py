import tensorflow as tf
import numpy as np
from PIL import Image
import cv2 as cv
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.mobilenet import MobileNet
import h5py


class FoodAesthetics:
    def __init__(self):

        super(FoodAesthetics, self).__init__()

        self.__batch_size = 1
        self.temperature = 1.536936640739441
        self.model = NimaMobileNet(training=False)
        self.model.build((self.__batch_size, 224, 224, 3))
        self.__home_path = Path(__file__).parent.resolve()
        self._load_weights_manually()

    def _load_weights_manually(self):
        """
        Load weights manually to handle layer count mismatch.
        """
        import h5py
        
        # First, build the model by calling it with dummy data
        dummy_input = tf.random.normal((1, 224, 224, 3))
        self.model(dummy_input)
        
        weights_path = self.__home_path/'trained_weights.h5'
        
        with h5py.File(weights_path, 'r') as f:
            # Load base model weights (MobileNet)
            base_model_group = f['mobilenet_1.00_None']
            for layer in self.model.base_model.layers:
                if layer.name in base_model_group:
                    layer_group = base_model_group[layer.name]
                    weights = []
                    for weight_name in layer.weights:
                        weight_key = weight_name.name.split('/')[-1]
                        if weight_key in layer_group:
                            weights.append(layer_group[weight_key][:])
                    if weights:
                        layer.set_weights(weights)
            
            # Load dense layer weights
            if 'model' in f and 'dense' in f['model']:
                dense_group = f['model']['dense']
                dense_weights = []
                if 'kernel:0' in dense_group:
                    dense_weights.append(dense_group['kernel:0'][:])
                if 'bias:0' in dense_group:
                    dense_weights.append(dense_group['bias:0'][:])
                if dense_weights:
                    self.model.dense1.set_weights(dense_weights)
            
            # Load final dense layer weights
            if 'dense_1' in f and 'dense_1' in f['dense_1']:
                final_dense_group = f['dense_1']['dense_1']
                final_weights = []
                if 'kernel:0' in final_dense_group:
                    final_weights.append(final_dense_group['kernel:0'][:])
                if 'bias:0' in final_dense_group:
                    final_weights.append(final_dense_group['bias:0'][:])
                if final_weights:
                    self.model.dense2.set_weights(final_weights)

    def aesthetic_score(self, path):
        """
        Compute aestehtic score of an image.

        Input: path of the image.
        Output: aesthetic score in range from 0 to 1.
        """
        
        photo = np.array(self._load_image(path))
        photo = tf.image.random_crop(tf.convert_to_tensor(photo / 255, dtype=tf.float16), (224, 224, 3))
        
        #photo = np.array(self._load_and_center_crop(path))
        #photo = tf.convert_to_tensor(photo / 255, dtype=tf.float16)

        logits = self.model(tf.expand_dims(photo, axis = 0))
        logits_scaled = tf.math.divide(logits, self.temperature)
        score = tf.nn.softmax(logits_scaled).numpy()[:, 1].item()
        return score


    def _load_image(self, path):
        """"
        Open and center crop picture mantaining aspect ratio.

        Input: path of image.
        Output: resized image. Shortest side: 224 pixels.
        """
        pic = Image.open(path)
        #pic = io.imread(path)
        width, height = pic.size
        s = max(224/width, 224/height)

        if width < height:
            pic_res = pic.resize((224, round(s*height)))
        else:
            pic_res = pic.resize((round(s*width), 224))

        return pic_res

    
    def _load_and_center_crop(self, path):
        """
        Load, Resize, and Center Crop image.
        
        Input: image path.
        Output: center cropped image.
        """
        pic = Image.open(path)
        width, height = pic.size

        # 1. resize image: shortest side is 224 pixels 
        s = max(224/width, 224/height)

        if width < height:
            pic_res = pic.resize((224, round(s*height)))
        else:
            pic_res = pic.resize((round(s*width), 224))

        #print(np.array(pic_res).shape)    

        # 2. center crop image
        left = (width - 224)/2
        top = (height - 224)/2
        right = (width + 224)/2
        bottom = (height + 224)/2

        cropped = pic_res.crop((left, top, right, bottom))
        #print(np.array(cropped).shape)

        return cropped



    def _image2hsv(self, pic):
        """"
        Convert image from RGB to HSV.

        Input: image.
        Output: HSV matrix w/ shape H x W x C, C = 3 for H, S, and V respectively.
        """
        return cv.cvtColor(np.float32(self._load_image(pic)), cv.COLOR_RGB2HSV)


    def brightness(self, path):
        """
        Cross Pixel Average of VALUE (2) dimension.

        Input: H x W x C HSV image.
        Output: brightness value range [0, 255].
        """
        image = self._image2hsv(path)
        return image[:,:,2].mean()

    def saturation(self, path):
        """
        Cross Pixel Average of SATURATION (1) dimension.

        Input: H x W x C HSV image.
        Output: saturation value range [0, 255].
        """
        image = self._image2hsv(path)
        return image[:, :, 1].mean()

    def contrast(self, path):
        """
        Cross Pixel Standard Deviation of VALUE(2) dimension.

        Input: H x W x C HSV image.
        Output: contrast value range [0, n].
        """
        image = self._image2hsv(path)
        return image[:, :, 2].std()

    def clarity(self, path, thresold = 0.7, scaler = 255):
        """
        Proportion of Normalized VALUE (2) pixels that exceed the thresold (0.7).

        Input: H x W x C HSV image.
        Output: clarity value range [0, 1].
        """
        image = self._image2hsv(path)
        h, w, c = image.shape
        return np.sum(image[:,:,2] / scaler > thresold) / (h * w)

    def warm(self, path):
        """"
        Proporion of Warm Hue (<30, >110) pixels.

        Input: H x W x C HSV image.
        Output: warm value range [0, 1].
        """
        image = self._image2hsv(path)
        h, w, c = image.shape
        return np.sum((image[:, :, 0] < 30) | (image[:, :, 0] > 110)) / (h * w)

    def colourfulness(self, path):
        """
        Follow Hasler and Suesstrunk (2003) to compute colourfoulness.

        Input: H x W x C RGB image.
        Output: colourfoluness score.
        """
        image = np.array(self._load_image(path))
        R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
        rg = R - G
        yb = 0.5 * (R+G) - B
        sigma = np.sqrt(np.square(np.std(rg)) + np.square(np.std(yb)))
        mu = np.sqrt(np.square(np.mean(rg)) + np.square(np.mean(yb)))
        c = sigma + 0.3 * mu # colourfoulness
        return c

    def _grabcut(self, path, n_iter = 10):
        """"
        Performs GrabCut according to:
        https://docs.opencv.org/master/d8/d83/tutorial_py_grabcut.html

        Input: image path.
        Output: a binary FOREGROUND mask of dimension H x W
                such that 1 = foreground, 0 = background.
        """
        image = np.array(self._load_image(path))
        mask = np.zeros(image.shape[:2],np.uint8) # mask of shape of the image
        bgdModel = np.zeros((1,65),np.float64) # take as given by docs
        fgdModel = np.zeros((1,65),np.float64) # take as given by docs

        # see description later
        rect = (1,1,image.shape[0]-1, image.shape[1]-1)
        # starts at coordinates 0,0 and init a rectangle of size -1 w/ respect both to
        # width and to height
        cv.grabCut(image ,mask,rect, bgdModel, fgdModel, n_iter, cv.GC_INIT_WITH_RECT)
        return np.where((mask==2)|(mask==0),0,1).astype('uint8')


    def size_difference(self, path):
        """
        Computes Size Difference between Foreground and Background.

        Input: image path.
        Output: normalized size difference score.
        """

        mask = self._grabcut(path)
        h, w = mask.shape
        counts = np.unique(mask, return_counts=True)

        #{counts[0][0]:counts[1][0], counts[0][1]:counts[1][1]}
        # 1 foreground, 0 background
        n_for = counts[1][1]
        n_back = counts[1][0]
        return (n_for - n_back) / (h * w)


    def color_difference(self, path):
        """
        Computes the Euclidean distance between each RGB channel of Foreground
        compared to Background.

        Input: image path.
        Output: Normed Euclidean distance score across each channel.
        """

        image = image = np.array(self._load_image(path))
        mask = self._grabcut(path)

        # compute front
        front = image*mask[:,:,np.newaxis]

        # compute back
        back_mask = np.where(mask == 1, 0, 1) # jsut flip the original mask to detect the front
        back = image * back_mask[:,:, np.newaxis]

        # R, G, B
        r_front, g_front, b_front = front[:,:,0].flatten(), front[:,:,1].flatten(), front[:,:,2].flatten()
        r_back, g_back, b_back = back[:,:,0].flatten(), back[:,:,1].flatten(), back[:,:,2].flatten()

        # create vectors (front and back) containing the average value for each channel - exclude the 0s from both front and back
        avg_front = np.array([r_front[r_front != 0].mean(), g_front[g_front != 0].mean(), b_front[b_front != 0].mean()])
        avg_back = np.array([r_back[r_back != 0].mean(), g_back[g_back != 0].mean(), b_back[b_back != 0].mean()])

        return np.linalg.norm(avg_front - avg_back)


    def texture_difference(self, path):
        """
        Compute texture differences between front edges and back edges.

        Input: image path.
        Output: absolute value of normalized front and back edge texture difference.
        """

        # input
        image = image = np.array(self._load_image(path))
        mask = self._grabcut(path)

        # compute front
        front = image*mask[:,:,np.newaxis]
        front_flattened = front.flatten()
        front_shape = front_flattened[front_flattened != 0].shape[0] # number of pixels of the front
        #print(front_shape)

        # compute back
        back_mask = np.where(mask == 1, 0, 1) # jsut flip the original mask to detect the front
        back = image * back_mask[:,:, np.newaxis].astype(np.uint8)
        back_flattened = back.flatten()
        back_shape = back_flattened[back_flattened != 0].shape[0] # number of pixels of the back
        #print(back_shape)

        # edges front and back
        edges_front = cv.Canny(front, front.shape[0], front.shape[1])
        edges_back = cv.Canny(back, back.shape[0], back.shape[1])

        return np.abs((np.sum(edges_front) / front_shape) - (np.sum(edges_back) / back_shape))


    def segment_colorfulness(self, path):
        # split the image into its respective RGB components, then mask
        # each of the individual RGB channels so we can compute
        # statistics only for the masked region
        """

        """

        # input
        image = image = np.array(self._load_image(path))
        mask = self._grabcut(path)

        (B, G, R) = cv.split(image.astype("float"))
        R = np.ma.masked_array(R, mask=mask)
        G = np.ma.masked_array(G, mask=mask)
        B = np.ma.masked_array(B, mask=mask)
        # compute rg = R - G
        rg = np.absolute(R - G)
        # compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)
        # compute the mean and standard deviation of both `rg` and `yb`,
        # then combine them
        stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
        meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))
        # derive the "colorfulness" metric and return it
        return stdRoot + (0.3 * meanRoot)


# model 
class NimaMobileNet(tf.keras.Model):
    def __init__(self, training = True):

        super(NimaMobileNet, self).__init__()
        self.training = training
        self.base_model = MobileNet((None, None, 3), alpha=1, include_top=False,
            pooling='avg', weights=None)
        self.dense1 = Dense(10, activation='relu')
        self.dense2 = Dense(2)

    def call(self, x):
        x = self.base_model(x)
        x = self.dense1(x)
        return self.dense2(x)


if __name__ == '__main__':
    print('import ok')
    #print(Path(__file__).parent.resolve()/'/weights/trained_weights.h5')
    aes = FoodAesthetics()
    print('class inited')
    img = './images/image2.jpeg'
    print(aes.aesthetic_score(img))
    #print(aes.brightness(img))
    #print(aes.saturation(img))
    #print(aes.contrast(img))
    #print(aes.clarity(img))
    #print(aes.warm(img))
    #print(aes.colourfulness(img))
    #print(aes.size_difference(img))
    #print(aes.color_difference(img))
    #print(aes.texture_difference(img))
    #print(aes.segment_colorfulness(img))
