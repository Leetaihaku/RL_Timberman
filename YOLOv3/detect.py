import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import Main_Header as Main
import RL_Header as RL
import keyboard
import pyautogui
import time

#################################################################################################################
#################################################################################################################

                                        # 탐지 및 탐지상태 변환함수

#################################################################################################################
#################################################################################################################
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # [수정 :: 강화학습 모듈 시작]
    ########################################################################################################
    # 에피소드 시작
    # Agent 생성 + 초기상태 설정 + 최초 행동 시행(Const = 'Start_button')
    ########################################################################################################
    Agent = RL.Agent()
    State = torch.zeros([1, 4], device='cuda')
    pyautogui.click(x=955, y=955)
    #임시 반복 카운터
    tmp_count = 0

    ########################################################################################################
    # 탐지 모듈(상태 생성기)
    ########################################################################################################
    for path, img, im0s, vid_cap in dataset:
        #반복문 내 변수 초기화
        Episode_Start = False  # 에피소드 시종제어
        Branch = ''  # 나뭇가지 상태 -> 신경망 입력 형변환
        Player = ''  # 나무꾼 상태 -> 신경망 입력 형변환
        Revive_Y = ''  # 이어하기_Y 상태 -> 신경망 입력 형변환
        Revive_N = ''  # 이어하기_N 상태 -> 신경망 입력 형변환

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            center_array = [] # center collection
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # [수정 :: 주석 변환]
            #save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        # [수정 :: func plot_one_box() -> 중심점 플로팅]
                        # 좌상 x = xyxy[0], 좌상 y = xyxy[1], 우하 x = xyxy[2], 우하 y = xyxy[3]
                        label = f'{names[int(cls)]} {conf:.2f}'
                        center = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        #객체 클래스 및 중심점 저장[클래스 이름, [중심좌표 x, 중심좌표 y]]
                        center_array.append([label[:-5], center])

            # [수정 :: 화면 이름 설정, 크기 조정]
            # Stream results
            if view_img:
                name = 'Detector'
                cv2.namedWindow(name)
                cv2.moveWindow(name, 1920, -550)
                cv2.resizeWindow(name, 1080, 720)
                im0 = cv2.resize(im0, (1080, 720))
                cv2.imshow(name, im0)

        ########################################################################################################
        # 탐지(raw status) -> 상태(converted status) 변환
        ########################################################################################################
        status = Main.ternary(center_array) #격자 상태 변환
        print('status : ', status)

        ########################################################################################################
        # 에피소드 종료 또는 객체 탐지 X 상태 추가
        ########################################################################################################
        for i in range(len(status)): # 신경망 입력 준비
            if status[i][0] == 'Branch':
                Branch += str(status[i][1])+str(status[i][2])
            elif status[i][0] == 'Player':
                Player += str(status[i][1])+str(status[i][2])
            elif status[i][0] == 'Revive_Y':
                Revive_Y += str(status[i][1])+str(status[i][2])
            elif status[i][0] == 'Revive_N':
                Revive_N += str(status[i][1])+str(status[i][2])
            elif status == 'Episode_Start':
                Episode_Start = True
            else:
                print('상태 스택 쌓기 모듈에 알 수 없는 에러 발생')

        ########################################################################################################
        # 널 값 점검 조건부 -> 만일의 널 값 대비
        ########################################################################################################
        Branch = str(0) if Branch == '' else Branch
        Player = str(0) if Player == '' else Player
        Revive_Y = str(0) if Revive_Y == '' else Revive_Y
        Revive_N = str(0) if Revive_N == '' else Revive_N

        ########################################################################################################
        # 다음 상태 추출(단, 종점이면 다음 상태 = [0, 0, 0, 0])
        ########################################################################################################
        Next_state = torch.tensor([int(Branch), int(Player), int(Revive_Y), int(Revive_N)], device='cuda') \
            if Episode_Start == False else torch.zeros([1,4])
        print('Next_state ', Next_state)

        ########################################################################################################
        # 에피소드 종료
        ########################################################################################################
        if(Episode_Start or tmp_count == 10):
            #마무리 및 저장 시퀀스 정리할 것!
            Reward = -1
            break
        else:
            tmp_count += 1
            Reward = 1
            State = Next_state
            Action = Agent.Action(State)
            keyboard.press_and_release(Action)
            ###############
            # 신경망 업데이트
            ###############
            time.sleep(0.1)



        # Action = Agent.Action(State)
        # print('Action ', Action)
        # keyboard.press_and_release(Action)


        # Next_state =
        # Reward
        # V_value =
        # Next_V_value =

        #if len(Batch_state) < RL.BATCH_SIZE

    ########################################################################################################
    # 학습 종료
    ########################################################################################################
    print('epi exit')
    exit()



        ########################################################################################################
        # 단일 탐지 싸이클 끝선
        ########################################################################################################
    ########################################################################################################
    # 탐지 종료 이후
    ########################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov3.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov3.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()