import time
import cv2

import torch

from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset
from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils.checkpoint import CheckPointer


class Classifier:
    @torch.no_grad()
    def __init__(self, cfg_file, cfg_opt, ckpt, dataset_type, score_threshold):
        cfg.merge_from_file(cfg_file)
        cfg.merge_from_list(cfg_opt)
        cfg.freeze()

        print("Loaded configuration file {}".format(cfg_file))
        with open(cfg_file, "r") as cf:
            config_str = "\n" + cf.read()
            print(config_str)
        print("Running with config:\n{}".format(cfg))

        if dataset_type == "voc":
            self.class_names = VOCDataset.class_names
        elif dataset_type == 'coco':
            self.class_names = COCODataset.class_names
        else:
            raise NotImplementedError('Not implemented now.')
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.model = build_detection_model(cfg)
        self.model = self.model.to(self.device)
        self.checkpointer = CheckPointer(self.model, save_dir=cfg.OUTPUT_DIR)
        self.checkpointer.load(ckpt, use_latest=ckpt is None)
        self.weight_file = ckpt if ckpt else self.checkpointer.get_checkpoint_file()
        print('Loaded weights from {}'.format(self.weight_file))

        self.cpu_device = torch.device("cpu")
        self.transforms = build_transforms(cfg, is_train=False)
        self.model.eval()

        self.score_threshold = score_threshold

    def annotate_image(self, src_img, boxes, labels, scores):
        for box in range(len(boxes)):
            cv2.rectangle(src_img, (boxes[box][0], boxes[box][1]), (boxes[box][2], boxes[box][3]), (0, 0, 0), 10)
            cv2.rectangle(src_img, (boxes[box][0], boxes[box][1]), (boxes[box][2], boxes[box][3]), (0, 0, 255), 2)
            cv2.putText(src_img, self.class_names[labels[box]] + str(': ') + str(scores[box]),
                        (boxes[box][0], boxes[box][3]),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 5)
            cv2.putText(src_img, self.class_names[labels[box]] + str(': ') + str(scores[box]),
                        (boxes[box][0], boxes[box][3]),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

        return src_img

    def process_image(self, img, width, height, annot_img):
        start = time.time()

        images = self.transforms(img)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        result = self.model(images.to(self.device))[0]
        inference_time = time.time() - start

        result = result.resize((width, height)).to(self.cpu_device)

        scores = result['scores'].detach().numpy()
        boxes = result['boxes'].detach().numpy()
        labels = result['labels'].detach().numpy()

        indices = scores > self.score_threshold
        boxes = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]
        meters = ' | '.join(
            [
                'objects {:02d}'.format(len(boxes)),
                'load {:03d}ms'.format(round(load_time * 1000)),
                'inference {:03d}ms'.format(round(inference_time * 1000)),
                'FPS {}'.format(round(1.0 / inference_time))
            ]
        )
        print(meters)

        return self.annotate_image(annot_img, boxes, labels, scores)
