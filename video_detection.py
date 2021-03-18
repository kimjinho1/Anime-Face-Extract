import os
import cv2
import sys
import glob
import json
import time
import random
import hashlib
import imutils
import argparse
import numpy as np
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict

from core.detector import LFFDDetector


# 모든 데이터(영상)의 주소를 딕셔너리 형태로 리턴
def get_all_videos():
    d = {}
    root_path = glob("./data/*")
    for i in range(len(root_path)):
        root_path[i] = root_path[i].replace('\\', '/')
    for path in root_path:
        li = []
        for file_path in glob(f"{path}/*"):
            t = file_path.replace('\\', '/')
            if t not in li:
                li.append(t)
        li.sort()
        d[path+'/'] = li
    cnt = 0
    for file_path, file_names in d.items():
        print(f"{file_path}: {file_names}")
        cnt += len(file_names)
    print(f"\n{cnt} Videos found\n")

    return d


def face_detection(df, video_path, output_path, detector, size=None, confidence_threshold=None, nms_threshold=None, roi=(), frame_count=None, run_per_x_frames=120, fps=None):
    cap = cv2.VideoCapture(video_path)
    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = fps or cap.get(cv2.CAP_PROP_FPS) / run_per_x_frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if roi:
        xmin, ymin, xmax, ymax = roi
        frame_width, frame_height = xmax - xmin, ymax - ymin
    else:
        xmin = 0
        ymin = 0
        xmax = frame_width
        ymax = frame_height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fc = frame_count or fc
    frame_count = fc
    bar = tqdm(total=int(frame_count))

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cnt = len(glob(f"{output_path}*"))-1
    if cnt < 0:
        cnt += 1
    while True:
        ret = cap.grab()
        frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if not ret or frame_idx > frame_count:
            break
        if int(frame_idx) % run_per_x_frames != 0:
            bar.update()
            continue

        _, frame = cap.retrieve()
        boxes = detector.detect(frame, size=size, confidence_threshold=confidence_threshold, nms_threshold=nms_threshold)
        if boxes:
            s = ""
            for i in range(len(boxes)):
                xmin, ymin, xmax, ymax = boxes[i]["xmin"], boxes[i]["ymin"], boxes[i]["xmax"], boxes[i]["ymax"]
                s += f"{ymin}|{xmin}|{ymax}|{xmax}/"
            df.loc[len(df)] = [f"{output_path}{cnt}.jpg", s]
            cv2.imwrite(f"{output_path}/{cnt}.jpg", frame)
            cnt += 1
        bar.update()
    bar.close()
    cap.release()
    df.to_csv( f"{output_path}ROI.csv", encoding='euc-kr')
    
    return cnt


def main():
    parser = argparse.ArgumentParser(description="Run detector on the video.")
    parser.add_argument("-i", "--input-path", help="Input images directory.", default=None)
    parser.add_argument("-o", "--output-path", help="Output video path.", default=None)
    parser.add_argument("-d", "--detector-path", help="Detector JSON configuration path.", default="configs/anime.json")
    parser.add_argument("--processing_all", default=False, type=bool)
    parser.add_argument("--use-gpu", help="Use GPU or not (default=%(default)s).", type=int, default=0)
    parser.add_argument("--size", help="Image size (default=%(default)s).", type=float, default=None)
    parser.add_argument("--confidence-threshold", help="Confidence threshold (default=%(default)s).", type=float, default=None)
    parser.add_argument("--nms-threshold", help="NMS threshold (default=%(default)s).", type=float, default=None)
    parser.add_argument("--fps", help="Video output FPS (default=%(default)s).", type=float, default=None)
    parser.add_argument("--run-per-x-frames", help="Run per X frames (default=%(default)s).", type=float, default=120)
    parser.add_argument("--frame-count", help="Number of video frames to process (including skipped ones) (default=%(default)s).", type=float, default=None)
    
    args = parser.parse_args()

    with open(args.detector_path, "r") as f:
        config = json.load(f)
    if args.use_gpu > 0:
        use_gpu = True
    else:
        use_gpu = False
    detector = LFFDDetector(config, use_gpu=use_gpu)
    
    # 입력이 잘못된 경우
    if (args.input_path == None or args.output_path == None) and \
        args.processing_all == False:
        print("Wrong input! try again")
        return ;
    
    # 모든 데이터를 다 처리
    if args.processing_all == True:
        d = get_all_videos()
        for file_path, file_names in d.items():
            anime_name = file_path.split('/')[-2]
            output_path = f"result/{anime_name}/"
            print(f"{anime_name} Start")

            # 확인용 카운트 변수
            a = 0

            df = pd.DataFrame(columns=("image_path","ROI"))
            for input_path in file_names:
                print("file_path:", file_path)
                print("file name:", input_path)
                print("output_path:", output_path, '\n')
                cnt = face_detection(df, input_path, output_path, detector, size=args.size,
                    confidence_threshold=args.confidence_threshold, nms_threshold=args.nms_threshold,
                    frame_count=args.frame_count, run_per_x_frames=args.run_per_x_frames, fps=args.fps
                )
                print(f"{anime_name} 총 {cnt}개 저장됨.\n")

    # 지정한 데이터만 처리
    else:
        df = pd.DataFrame(columns=("image_path","ROI"))
        cnt = face_detection(df,
            args.input_path, 
            args.output_path, 
            detector, 
            size=args.size,
            confidence_threshold=args.confidence_threshold,
            nms_threshold=args.nms_threshold,
            frame_count=args.frame_count, 
            run_per_x_frames=args.run_per_x_frames, 
            fps=args.fps
        )
        print(f" 총 {cnt}개 저장됨.\n")
    
    
if __name__ == "__main__":
    main()