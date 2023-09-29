import json
import re
import os
import torch
from glob import glob
from flask import Flask, request, Response
from ultralytics import YOLO

from yolo_v8.preprocess import preprocess

print('-------------------------')
print('initializing....', end='\t')


print('complete')
# load_inference = load.signatures["serving_default"]

# 플라스크 및 모델 선언
app = Flask('ocr')
model = None


# 점수 리턴 형식 지정
def score_validator(data: str):
    score_pattern = re.compile('\d1\d')
    score_digit_pattern = re.compile('\d\d')
    # 데이터가 1만 인식되었을 경우
    if data == '1':
        return None

    # N1N 형식으로 입력될 경우
    if score_pattern.match(data):
        score = re.findall(score_pattern, data)[0]
        score_slash = f"{score[0]}/{score[2]}"
        return score_slash

    # NN 형식으로 인식될 경우
    if score_digit_pattern.match(data):
        score = re.findall(score_pattern, data)[0]
        score_slash = f"{score[0]}/{score[1]}"
        return score_slash

    else:
        return None


@app.route('/ocr')
def test():
    return 'test'


@app.route('/ocr', methods=['POST'])
def main():
    if 'data' in request.files:
        data = request.files['data']
        filename, ext = data.filename.split('.')

        print(f'request inference \t :: {filename}')

        # 이미지 저장 디렉터리 지정 및 저장
        file_storage = data
        original_img_dir = os.path.join(os.getcwd(), 'yolo_v8', 'img')
        os.makedirs(original_img_dir, exist_ok=True)
        file_storage.save(os.path.join(original_img_dir, f'{filename}.{ext}'))

        # 이미 Json 결과 파일이 있는지 체크후 이미 있다면 기존 결과 리턴
        directory = os.path.join('.', 'yolo_v8', 'result')
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, str(filename) + '.json')

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return json.load(file)

        # 전처리 모듈 실행
        site, img_path = preprocess(image_file_name=data.filename)

        # 이미지 전처리 결과를 차례대로 배열로 불러와 Flat하게 반환 후 해당 배열 모델에 입력
        imgs = sum(
            [glob(img_path + f'{os.path.sep}{i}{os.path.sep}*.png') for i in range(0, 20)], [])
        results = model(source=imgs, conf=0.5, stream=True)

        # 반환 데이터 형식 선언
        row = {"ocr": [], 'site': site}
        col = {}

        # 모델 반환 결과 loop
        for box in results:

            # 각각 box.path에서 파일 이름 불러오기 (어떤 요소인지 파악 위함)
            sep = box.path.split(os.path.sep)
            id, folder = sep[-1].split('.')[0], sep[-2]

            # box 검출 좌표순으로 정렬 (왼쪽으로부터 x좌표 순)
            boxes = sorted(box.boxes.data.tolist())
            data = ''

            # Box내 데이터 리턴 (10.0의 경우 점을 의미)
            for j in boxes:
                if j[5] != 10.0:
                    data += str(int(j[5]))
                else:
                    data += '.'

            # 파일 이름의 id를 보고 어떤 요소인지 파악한 뒤 각각의 column에 저장
            if id == '0':
                col['frequency'] = data
            elif id == '1':
                col['txmain'] = score_validator(data)
            elif id == '2':
                col['rxmain'] = score_validator(data)
            elif id == '3':
                col['txstby'] = score_validator(data)
            elif id == '4':
                col['rxstby'] = score_validator(data)
            elif id == '5':
                col['angle'] = data
            elif id == '6':
                col['distance'] = data
            elif id == '7':
                col['height'] = data
                row['ocr'].append(col)
                print(f'row : {folder} | content : {col} ')
                col = {}

        # 결과값 Json으로 저장
        with open(file_path, 'w') as json_file:
            json.dump(row, json_file)

        return row
    else:
        return Response(status=400)


if __name__ == "__main__":
    # 모델 로드
    print('loading model....', end='\t')
    model = YOLO(os.path.join(os.getcwd(), 'yolo_v8', 'runs',
                 'detect', 'train3', 'weights', 'best.pt'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print('complete')
    app.run(host='0.0.0.0', port=7001, debug=True)
