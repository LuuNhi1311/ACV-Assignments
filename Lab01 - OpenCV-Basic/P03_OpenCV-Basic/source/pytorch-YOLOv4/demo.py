

from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import os
import time
import torch
import argparse

use_cuda = True

def load_model(cfgfile, weightfile):
    m = Darknet(cfgfile, inference=True)
    if weightfile.endswith('.pt') or weightfile.endswith('.pth'):
        state = torch.load(weightfile, map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        m.load_state_dict(state)
    else:
        m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    if use_cuda:
        m.cuda()
    return m


def list_images(imgfile):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    if os.path.isdir(imgfile):
        return sorted(os.path.join(imgfile, f) for f in os.listdir(imgfile)
                      if f.lower().endswith(exts))
    return [imgfile]


def detect_image(m, img, class_names, conf=0.5, nms=0.6, cuda=None):
    import cv2
    cuda = use_cuda if cuda is None else cuda
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    boxes = do_detect(m, sized, conf, nms, cuda)
    drawn = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)
    return drawn, len(boxes[0])


def detect_cv2(cfgfile, weightfile, imgfile, namesfile=None, outdir='visualizations', conf=0.4, nms=0.6):
    import cv2
    m = load_model(cfgfile, weightfile)

    num_classes = m.num_classes
    if not namesfile:
        namesfile = {20: 'data/voc.names', 80: 'data/coco.names'}.get(num_classes, 'data/x.names')
    class_names = load_class_names(namesfile)

    out_dir = os.path.join(outdir, 'detections')
    os.makedirs(out_dir, exist_ok=True)
    images = list_images(imgfile)
    print('Detecting %d image(s), saving to %s ...' % (len(images), out_dir))

    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue
        start = time.time()
        drawn, nb = detect_image(m, img, class_names, conf, nms)
        print('%s: %d boxes in %.3fs' % (os.path.basename(path), nb, time.time() - start))
        cv2.imwrite(os.path.join(out_dir, os.path.basename(path)), drawn)


def detect_cv2_camera(cfgfile, weightfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    if args.torch:
        m.load_state_dict(torch.load(weightfile))
    else:
        m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(0)

    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    while True:
        ret, img = cap.read()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)

        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)

    cap.release()


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/best.pt',
                        help='path of trained model (.pt/.pth or darknet .weights).', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default='./data/mscoco2017/train2017/190109_180343_00154162.jpg',
                        help='image file or a folder of images.', dest='imgfile')
    parser.add_argument('-names', type=str, default=None,
                        help='class names file (one per line).', dest='namesfile')
    parser.add_argument('-outdir', type=str, default='visualizations',
                        help='output dir; boxes saved under <outdir>/detections.', dest='outdir')
    parser.add_argument('-conf', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('-nms', type=float, default=0.6, help='NMS IoU threshold')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if args.imgfile:
        detect_cv2(args.cfgfile, args.weightfile, args.imgfile,
                   namesfile=args.namesfile, outdir=args.outdir, conf=args.conf, nms=args.nms)
    else:
        detect_cv2_camera(args.cfgfile, args.weightfile)
