import OpenEXR
import Imath
import numpy as np
import cv2


def write_exr(img, filename):
    header = OpenEXR.Header(img.shape[1], img.shape[0])
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    header['channels'] = dict([(c, half_chan) for c in "RGB"])

    out = OpenEXR.OutputFile(filename, header)
    r_channel = (img[:, :, 2]).astype(np.float16).tostring()
    g_channel = (img[:, :, 1]).astype(np.float16).tostring()
    b_channel = (img[:, :, 0]).astype(np.float16).tostring()
    out.writePixels({'R': r_channel, 'G': g_channel, 'B': b_channel})
    out.close()


def scale(img):
    image = ((img - np.min(np.ravel(img)))/(np.max(np.ravel(img)) - np.min(np.ravel(img)))) * 255.0
    image[image < 1e-5] = 1e-5
    return image


def scale_fixed(img, img_min, img_max):
    image = img.copy()
    image[image < img_min] = img_min
    image[image > img_max] = img_max
    image = ((image - img_min)/(img_max - img_min)) * 255.0
    return image


def scale_and_replace(image, image_min, image_max, image_min_fixed, image_max_fixed):
    image = ((image - image_min) / (image_max - image_min)) * 255.0
    image_fixed = ((image - image_min_fixed) / (image_max_fixed - image_min_fixed)) * 255.0
    #image[image < 1e-5] = 1e-5
    #image_fixed[image_fixed < 1e-5] = 1e-5
    return image, image_fixed


def calculate_min_max(min_exp, max_exp):
    exposure_times = np.array([min_exp, max_exp], dtype=np.float32)

    img_list = [
        np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8),
        np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8),
    ]

    merge = cv2.createMergeDebevec()
    hdr = merge.process(img_list, times=exposure_times)

    return np.nanmin(hdr), np.nanmax(hdr)

