import multiprocessing
import matplotlib.image as mpimg
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import pypylon
from matplotlib.pyplot import pause, figure, draw

# x = 100
# y = 100
# h = 50
# w = 10
# delta = 20
img_flag = 0
demo_net = 'vgg16' 
gpu_id = 0

CLASSES = ('__background__', 'apple', 'person')

NETS = {'vgg16': ('VGG16',
                  'vgg16_faster_rcnn_iter_4000_clean.caffemodel'),
        'zf': ('ZF',
                  'zf_faster_rcnn_iter_8000.caffemodel')}



def capture_image(cam):
    cap_images = cam.grab_images(1)
    cap_img = [o for o in cap_images][0]
    out_img = np.zeros((cap_img.shape[0], cap_img.shape[1] / 3, 3), dtype=np.uint8)
    for m in range(cap_img.shape[1]):
        c = int(m / 3)
        out_img[:, c, 2 - (m % 3)] = cap_img[:, m]
    return out_img




def frame_detect_fasterRCNN(arrays, bbox_queue):
    """Detect object classes in an image using faster RCNN with pre-computed net models."""

    thresh = 0.8
    # load caffe net
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        
    prototxt = os.path.join(cfg.MODELS_DIR, NETS[demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'imagenet_models',
                              NETS[demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))


    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    cfg.GPU_ID = gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    
    while True:

        try:
            im = arrays[1]
            arrays[0].release()
            # Detect all object classes and regress object bounds
            timer = Timer()
            timer.tic()
            scores, boxes = im_detect(net, im)
            timer.toc()
            print ('Detection took {:.3f}s for '
                  '{:d} object proposals').format(timer.total_time, boxes.shape[0])

            # Visualize detections for each class
            # CONF_THRESH = 0.8
            NMS_THRESH = 0.3
            for cls_ind, cls in enumerate(CLASSES[1:]):
                if cls == 'apple':

                    cls_ind += 1  # because we skipped background
                    cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                    cls_scores = scores[:, cls_ind]
                    dets = np.hstack((cls_boxes,
                                      cls_scores[:, np.newaxis])).astype(np.float32)
                    keep = nms(dets, NMS_THRESH)
                    dets = dets[keep, :]

                    # filter and keep all the bboxes with scores larger than threshold
                    inds = np.where(dets[:, -1] >= thresh)[0]
                    if len(inds) == 0:
                        return
                    for i in inds:
                        final_bbox.append(dets[i, :4])
                break
            bbox_queue.put(final_bbox)
        except BaseException, e:
            pass


def show_result(img_flag, bbox_queue, cam):



    while(1):

        frame = capture_image(cam)
        img_mparray.acquire()
        img_ndarray.fill(frame)

        # try:
        #     bbox_get = bbox_queue.get(block=False)
        #     for i in bbox_get:
        #         bbox = bbox_get[i, :4]
        #         score = bbox_get[i, -1]
        #         cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        #         img_flag = 0
        # except BaseException, e:
        #     if not img_flag:
        #         #img.put(frame, block=False)
        #         img_arrays[0].acquire()
        #         img_arrays[1] = frame
        #         img_flag += 1
        #         bbox = [0, 0, 0, 0]
        #     elif bbox:
        #         cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        #     else:
        #         pass

        cv2.waitKey(10)
        cv2.imshow("image", frame)
        # ax = figure().gca()
        # ax.imshow(frame)
        # draw()
        pause(0.01)
        #return img_arrays
    cam.close()


if __name__ =='__main__':

    img_flag = 0
    bbox = []


    # define a queue to communicate between 2 processes
    bbox_queue = multiprocessing.Queue()

    array_dim = (480, 640)
    img_mparray = multiprocessing.Array('I', int(np.prod(array_dim)), lock = multiprocessing.Lock())
    img_ndarray = np.frombuffer(img_mparray.get_obj(), dtype='I').reshape(array_dim)
    arrays = [img_mparray, img_ndarray]

    devices = pypylon.factory.find_devices()
    cam = pypylon.factory.create_device(devices[0])
    cam.open()
    cam.properties['PixelFormat'] = 'RGB8'

    detector = multiprocessing.Process(target=frame_detect_fasterRCNN, name="detect bbox", args=(arrays, bbox_queue,))
    detector.start()

    while True:
        frame = capture_image(cam)

        if not img_flag:
            img_mparray.acquire()
            #img_ndarray.fill(frame)
            img_ndarray = frame
            img_flag = 0

        img_flag += 1
        try:
            bbox = bbox_queue.get()
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        except BaseException, e:
            pass
        cv2.waitKey(10)
        cv2.imshow("capture images", frame)
        pause(0.01)
    cam.close()


    # to test the frame_detect_fasterRCNN function
    # bbox_queue = multiprocessing.Queue()
    # array_dim = (480, 640)
    # img_mparray = multiprocessing.Array('I', int(np.prod(array_dim)), lock = multiprocessing.Lock())
    # img_ndarray = np.frombuffer(img_mparray.get_obj(), dtype='I').reshape(array_dim)
    # img_arrays = [img_mparray, img_ndarray]
    #
    # devices = pypylon.factory.find_devices()
    # cam = pypylon.factory.create_device(devices[0])
    # cam.open()
    # cam.properties['PixelFormat'] = 'RGB8'
    #
    # show_result(img_flag, bbox_queue, cam)
    # cam.close()

    print("DONE")







