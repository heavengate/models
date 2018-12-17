import os
import time
import numpy as np
import paddle
import paddle.fluid as fluid
import box_utils
import reader
from utility import print_arguments, parse_args
import models
# from coco_reader import load_label_names
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from config.config import cfg

np.set_printoptions(threshold='nan')
np.set_printoptions(suppress=True)

def infer():

    if not os.path.exists('output'):
        os.mkdir('output')

    model = models.YOLOv3(cfg.model_cfg_path, is_train=False)
    model.build_model()
    outputs = model.get_pred()
    input_size = model.get_input_size()
    yolo_anchors = model.get_yolo_anchors()
    yolo_classes = model.get_yolo_classes()
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if cfg.pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
        fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)
    # yapf: enable
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())
    fetch_list = outputs
    image_names = []
    if cfg.image_name is not None:
        image_names.append(cfg.image_name)
    else:
        for image_name in os.listdir(cfg.image_path):
            if image_name.split('.')[-1] in ['jpg', 'png']:
                image_names.append(image_name)
    for image_name in image_names:
        infer_reader = reader.infer(input_size, os.path.join(cfg.image_path, image_name))
        label_names, _ = reader.get_label_infos()
        data = next(infer_reader())
        im_shape = data[0][2]
        outputs = exe.run(
            fetch_list=[v.name for v in fetch_list],
            feed=feeder.feed(data),
            return_numpy=True)

        pred_boxes, pred_scores, pred_labels = box_utils.get_all_yolo_pred(outputs, yolo_anchors,
                                                            yolo_classes, (input_size, input_size))
        boxes, scores, labels = box_utils.calc_nms_box_new(pred_boxes, pred_scores, pred_labels, 
                                                       cfg.valid_thresh, cfg.nms_thresh)
        boxes = box_utils.rescale_box_in_input_image(boxes, im_shape, input_size)
        path = os.path.join(cfg.image_path, image_name)
        box_utils.draw_boxes_on_image(path, boxes, scores, labels, label_names, cfg.conf_thresh)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    infer()
