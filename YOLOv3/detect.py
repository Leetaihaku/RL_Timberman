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
import pyautogui
import time
import keyboard

#################################################################################################################

                                        # 탐지 및 탐지상태 변환함수

#################################################################################################################



#################################################################################################################

                                        # 모델파일 조건부 로드

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

    ########################################################################################################
    # 에피소드 초기 설정
    # 초기 설정(환경, 에이전트, 상태 생성, 에피소드 시종제어)
    Environment = RL.Environment()
    Agent = RL.Agent()
    State = torch.tensor([0, 0, 0, 0, 0], device='cuda')
    # 게임 활성화 클릭 에피소드 시작 준비
    # 활성화
    pyautogui.moveTo(x=960, y=640)
    pyautogui.doubleClick()
    # 에피소드 시작 준비(완전탐지 딜레이)
    keyboard.press_and_release('s')

    # Skip 변수 = 최초 상태 인식 위함
    # Detect 변수 = 유효 이미지 색출 위함
    Skip = True
    Detect = True
    # 임시 반복 카운터
    tmp_count = 0
    ########################################################################################################

    # 탐지 모듈(상태 생성기) 루프
    for path, img, im0s, vid_cap in dataset:
        ########################################################################################################
        # 에피소드 시작
        # 탐지 버퍼 초기화
        center_array = []
        # 행동 선택 및 시행
        if not Skip and Detect:
            print('###########################')
            print('State : ', State)
            Action = Agent.Action(State)
            print('Action ', Action)
            keyboard.press_and_release(Action)
            # 스텝 카운트
            tmp_count += 1
            print('tmp_count : ', tmp_count)
            time.sleep(0.1)
        ########################################################################################################

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
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # [수정 :: 주석 변환]
            # save_path = str(save_dir / p.name)  # img.jpg
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
                        # 객체 클래스 및 중심점 저장[클래스 이름, [중심좌표 x, 중심좌표 y]]
                        center_array.append([label[:-5], center])

            # 실시간 모니터링 화면 출력
            # Stream results
            if view_img:
                name = 'Detector'
                cv2.namedWindow(name)
                cv2.moveWindow(name, 1920, -550)
                cv2.resizeWindow(name, 1080, 720)
                im0 = cv2.resize(im0, (1080, 720))
                cv2.imshow(name, im0)



        ########################################################################################################
        # 다음상태 추출(단, 종점이면 다음 상태 = [0, 0, 0, 0])
        # 탐지(raw status) -> 상태(converted status) = 격자상태 변환
        Next_state = Environment.Step(center_array)
        if not torch.equal(State, Next_state):
            Detect = True
            print('Next_state : ', Next_state)
            print('###########################')
        else:
            Detect = False
            print('')
            continue
        ########################################################################################################



        ########################################################################################################
        # 프로세스 종합 분기점(최초 시작, 종료, 진행)
        # 최초 상태 할당 조건

        if Skip\
                and State.tolist() not in [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]\
                and Next_state.tolist() not in [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]\
                and State[1] != 0 and Next_state[1] != 0:
            # 이미지 버퍼 비워진 상태일 시,
            Skip = not Skip
            print('skip off')

        # 종료 확인 및 다음 프로세스 진행(보상 수여, 스택쌓기 => 신경망 업데이트) 조건
        elif not Skip and tmp_count == 10:
            # 마무리 및 저장 시퀀스 정리할 것!
            # Reward = -1
            if not Skip:
                print('cause skip')
            elif tmp_count == 50:
                print('max count')
            else:
                print('next state incorrect')
            print('all break')
            break

        # 다음상태 이어서 진행 조건
        else:
            # Reward =
            # V_value =
            # Next_V_value =
            ###############
            # 신경망 업데이트(N-step)
            # if len(Batch_state) < RL.BATCH_SIZE
            ###############
            # 상태 전달 및 다음상태 버퍼 비우기
            print('last of loop')
        State = Next_state


        #다음상태 넘기기

        ########################################################################################################



    ########################################################################################################
    # 학습 종료
    print('epi exit')
    exit()
    ########################################################################################################
    # 탐지 종료 이후 절차
    # (모델 저장)






    ########################################################################################################
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