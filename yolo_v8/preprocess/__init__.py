from .site_process import site_processor
from .image_splitter import image_split

import cv2
import os


def preprocess(image_path=f'{os.path.sep}.', save=True):

    # 이미지 패스가 주어졌다면 파일명과 확장자 변수화
    [filename, ext] = image_path.split(os.path.sep)[-1].split('.')

    # 불러올 이미지 경로 지정
    img_name = os.path.join(os.getcwd(), 'yolo_v8', 'img', f'{filename}.{ext}')
    image = cv2.imread(img_name)

    resize = cv2.resize(image, dsize=(1651, 2335), interpolation=cv2.INTER_AREA)
    gray_image = None
    
    # 이미지가 컬러(RGB or BGR)이라면 그레이스케일로 변환
    if (len(resize.shape) == 3):
        gray_image = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = resize

    # 노이즈 제거 전처리 (가우시안 블러)
    image = cv2.GaussianBlur(resize, (3, 7), 0)

    # 쓰레시홀드 임계치 지정 (이진화)
    ret, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

    # 컨투어를 통해 넓이가 넓은 영역순으로 구함
    # 가장 넓은 칸은 결과 기입창, 그 다음창은 표지소 선택창
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_array = sorted(contours, key=cv2.contourArea, reverse=True)

    # 이미지를 칸마다 쪼갬
    img_arr = image_split(
        thresh, contour_array[0], file_path=image_path, save=save)

    # 선택한 표지소 리턴
    site = site_processor(thresh, contour_array[1])
    return site['site_name'], os.path.join(os.getcwd(), 'yolo_v8', 'datasets', 'img', filename)
