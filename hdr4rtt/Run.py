import os
from datetime import datetime
import win32file
import win32pipe
import struct
import time

from hdr4rtt.Classifier import Classifier
from hdr4rtt.Helpers import *
import argparse


def main():
    PrettyBigValue = 100000000.0

    parser = argparse.ArgumentParser(description="HDR4RTT Classifier")

    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--dataset_type", default="voc", type=str,
                        help='Specify dataset type. Currently support voc and coco.')
    parser.add_argument("--score_threshold", type=float, default=0.7)

    parser.add_argument("--image_properties", type=str, default="1280;720;3;4",
                        help="Image properties in the format: width;height;channels;bytedepth")

    parser.add_argument("--input_dir", default="", type=str, help="Path to dir with binary HDR images")
    parser.add_argument("--output_dir", default="", type=str, help="Path to dir to output classified HDR images")

    parser.add_argument('-outfiles', action='store_true', help="Set to output results to disk")
    parser.add_argument('-outshow', action='store_true', help="Set to output results to screen")
    parser.add_argument('-outtonemap', action='store_true', help="Set to tonemap the image on screen")
    parser.add_argument('-outbreakfirst', action='store_true', help="Set to break after first image")

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()
    print(args)

    classifier = Classifier(
        cfg_file=args.config_file,
        cfg_opt=args.opts,
        ckpt=args.ckpt,
        dataset_type=args.dataset_type,
        score_threshold=args.score_threshold
    )

    img_params = args.image_properties.split(';')
    width = int(img_params[0])
    height = int(img_params[1])
    channels = int(img_params[2])
    byte_depth = int(img_params[3])
    image_bytes = width * height * channels * byte_depth

    tonemaper = cv2.createTonemap(gamma=2.2)
    images = 0

    print('Ready to process {}x{}x{} images with {} bytes per channel'.format(width, height, channels, byte_depth))

    if args.input_dir != '':
        print("Running in process directory mode")
        src_dir = args.input_dir
        out_dir = ''

        if args.outfiles:
            if args.output_dir == '':
                if args.input_dir != '':
                    src_dir = args.input_dir
                    out_dir = os.path.join(src_dir, 'output')
                    try:
                        print('Using "{}" as output path'.format(out_dir))
                        os.mkdir(out_dir)
                    except Exception as e:
                        print(e)
                else:
                    print("Cannot output files because no --output-dir set")
                    exit(-1)
            else:
                out_dir = args.output_dir

        files = os.listdir(src_dir)

        # all_max = 0
        # for filename in files:
        #     print('Processing ' + filename)
        #     img = np.fromfile(os.path.join(src_dir, filename), dtype=np.float32)
        #     max_img = np.nanmax(img)
        #     if(max_img > all_max):
        #         all_max = max_img
        #
        # print('The max is: ' + str(all_max))
        # exit(0)

        for filename in files:
            print('Processing ' + filename)
            images += 1

            img = np.fromfile(os.path.join(src_dir, filename), dtype=np.float32) \
                .reshape(height, width, channels)

            evs = np.log2(np.nanmax(img.ravel()) / np.nanmin(img.ravel()))
            max = np.nanmax(img)
            min = np.nanmin(img)

            print('Received image n=' + str(images) + ' evs=' + str(evs) + ' max=' + str(max) + ' min=' + str(min))

            image, image_fixed = scale_and_replace(img, min, max, 0, PrettyBigValue)

            img_classified = classifier.process_image(image, width, height, image_fixed)

            if args.outfiles:
                filename = os.path.join(out_dir, filename + '_processed.exr')
                write_exr(img_classified, filename)

            if args.outshow:
                if args.outtonemap:
                    cv2.imshow('IMG', tonemaper.process(img_classified.copy()))
                else:
                    cv2.imshow('IMG', img_classified)
                cv2.waitKey(1)

            if args.outbreakfirst:
                break

    else:
        print("Running in pipe mode")
        out_dir = ''
        if args.outfiles:
            if args.output_dir == '':
                print("Cannot output files because no --output-dir set")
                exit(-1)
            else:
                out_dir = args.output_dir
        try:
            print('Trying to connect to pipe')

            handle = win32file.CreateFile(
                r'\\.\pipe\hdr4rtt',
                win32pipe.PIPE_ACCESS_INBOUND,
                0,
                None,
                win32file.OPEN_EXISTING,
                0,
                None
            )

            print('Connected')

            while True:
                rcv_image = win32file.ReadFile(handle, image_bytes)[1]

                if len(rcv_image) == 0:
                    print('Nothing received')
                    continue

                images += 1

                # continue

                img = np.frombuffer(rcv_image, np.float32).reshape(height, width, channels)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")

                evs = np.log2(np.nanmax(img.ravel()) / np.nanmin(img.ravel()))
                max = np.nanmax(img)
                min = np.nanmin(img)

                print('Received image n=' + str(images) + ' evs=' + str(evs) + ' max=' + str(max) + ' min=' + str(min))

                image, image_fixed = scale_and_replace(img, min, max, 0, PrettyBigValue)

                img_classified = classifier.process_image(image, width, height, image_fixed)

                # continue

                if args.outfiles:
                    filename = os.path.join(out_dir, timestamp + '_processed.exr')
                    write_exr(img_classified, filename)

                if args.outshow:
                    if args.outtonemap:
                        cv2.imshow('IMG', tonemaper.process(img_classified.copy()))
                    else:
                        cv2.imshow('IMG', img_classified)

                    if args.outbreakfirst:
                        cv2.waitKey(0)
                    else:
                        cv2.waitKey(1)

                if args.outbreakfirst:
                    break

        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
