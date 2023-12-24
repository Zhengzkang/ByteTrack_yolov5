import pathlib
import sys
import json
from demo_track import get_image_list, Predictor
from models.common import DetectMultiBackend
from tracker.byte_tracker import BYTETracker
from tracking_utils.timer import Timer
from loguru import logger
import argparse
import os



# ####### 参数设置
conf = 0.20
nms = 0.40
#######
tsize = (800, 800)
imgsz = 800
weights = "yolov5s.pt"
num_classes = 1
trt_file = "/project/train/models/train/exp/weights/best.engine"
fp16 = False
trt = False


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./datasets/mot/train/MOT17-02-FRCNN/img1", help="path to images or video"
        # "--path", default="./videos/16h-17h.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="test",
        help="save name for results txt/video",
    )

    parser.add_argument("-c", "--ckpt", default="yolov5s.pt", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--num_classes", type=int, default=1, help="number of classes")
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=(608, 1088), type=tuple, help="test image size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    return parser


def image_demo(predictor, path, test_size):
    args = make_parser().parse_args()
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort(key=lambda x: int(x.split('/')[-1][:-4]))
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    for image_name in files:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        outputs, img_info = predictor.inference(image_name, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            timer.toc()
            # online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
            #                           fps=1. / timer.average_time)
        else:
            timer.toc()
            # online_im = img_info['raw_img']

    return results


def init():
    # Initialize
    # os.system('python /project/train/src_repo/yolov5/export.py --weights /project/train/models/train/exp/weights/best.pt --half --img 800')
    # weights= '/project/train/models/train/exp/weights/best.engine'
    ckpt_file = weights
    model = DetectMultiBackend(ckpt_file, device='cuda')
    if fp16:
        model.model.half()
    model.eval()

    return model


def process_video(handle=None, input_video=None, args=None, **kwargs):
    # args = json.loads(args)

    frames_dict = {}
    for frame in pathlib.Path(input_video).glob('*.png'):
        frame_id = int(frame.with_suffix('').name)
    frames_dict[frame_id] = frame.as_posix()

    frames = list(frames_dict.items())  # frames[¨] = (frame_id, frame_file)

    # if trt:
    #     handle.head.decode_in_inference = False
    #     logger.info("Using TensorRT to inference")
    # else:
    #     trt_file = None
    # if trt:
    #     device = "gpu"

    predictor = Predictor(handle, num_classes, conf,
                          nms, tsize, trt_file, device='gpu', fp16=False)

    output_tracker_file = 'tracker.txt'
    # output_tracker_file = args['output_tracker_file']
    for frame, p in frames:
        result = image_demo(predictor, p, tsize)
        with open(output_tracker_file, 'w') as tracker_file:
            pred_tracker_data = result
            tracker_file.write(pred_tracker_data)

    return json.dumps({
        "model_data": {
            "objects": []
        },
        "status": "success"})


if __name__ == '__main__':
    from glob import glob
    # Test API
    image_names = 'test'
    predictor = init()
    process_video(predictor, image_names)

