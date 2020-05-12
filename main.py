from HRNet.c1_decoder import get_decoder
from HRNet.hrnet import get_encoder
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import scipy.io
import matplotlib.pylab as plt


def transform(image, size):
    resize = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    return resize(Image.fromarray(np.uint8(image))).unsqueeze(0)


def show(output):
    final = output[0].detach().numpy()
    classes = np.argmax(final, axis=0)
    print(classes.shape)
    mask = colors[classes]
    plt.imshow(mask)
    plt.show()


if __name__ == '__main__':
    encoder = get_encoder('encoder_epoch_30.pth')
    decoder = get_decoder(150, 'decoder_epoch_30.pth')
    image = cv2.imread('test.png')
    size = (560, 800)
    image = transform(image, size)
    encoding_img = encoder(image)
    decoding_img = decoder(encoding_img, segSize=size)
    colors = scipy.io.loadmat('color150.mat')['colors']
    show(decoding_img)
