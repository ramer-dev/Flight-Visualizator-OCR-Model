import json
import re
from glob import glob

import os
import torch
from flask import Flask, request, Response
from ultralytics import YOLO

from yolo_v8.preprocess import preprocess

print('-------------------------')
print('initializing....', end='\t')


print('complete')
# load_inference = load.signatures["serving_default"]  # 이게 무슨뜻?

app = Flask('ocr')
model = None
score_pattern = re.compile('\d1\d')


def score_validator(data: str):
    if data == '1':
        return None

    if score_pattern.match(data):
        score = re.findall(score_pattern, data)[0]
        score_slash = f"{score[0]}/{score[2]}"
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
        [filename, ext] = data.filename.split('.')

        print(f'request inference \t :: {filename}')

        # 원본 이미지 저장 장소
        file_storage = data
        original_img_dir = os.path.join(os.getcwd(), 'yolo_v8', 'img')
        os.makedirs(original_img_dir, exist_ok=True)
        file_storage.save(os.path.join(original_img_dir, f'{filename}.{ext}'))

        directory = os.path.join('.', 'yolo_v8', 'result')
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, str(filename) + '.json')

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return json.load(file)

        # Preprocess
        site, img_path = preprocess(data.filename)
        # Evaluate Process
        # imgs = glob(os.path.join(os.getcwd(), 'yolo_v8', f'{os.path.sep}*{os.path.sep}*.png')
        imgs = sum([glob(img_path + f'{os.path.sep}{i}{os.path.sep}*.png') for i in range(0, 20)], [])
        results = model(source=imgs, conf=0.5, stream=True)

        row = {"ocr": [], 'site': site}
        col = {}

        for box in results:
            sep = box.path.split(os.path.sep)

            id, folder = sep[-1].split('.')[0], sep[-2]
            # box 검출 좌표순으로 정렬
            boxes = sorted(box.boxes.data.tolist())
            data = ''
            # 컬럼 데이터 리셋

            # Box내 데이터 리턴
            for j in boxes:
                if j[5] != 10.0:
                    data += str(int(j[5]))
                else:
                    data += '.'

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

        with open(file_path, 'w') as json_file:
            json.dump(row, json_file)

        return row
    else:
        return Response(status=400)


@app.route('/inference', methods=['POST'])
def inference():
    print('request inference')
    data = request.json
    data = data['data']

    path = os.path.join(os.getcwd(), 'yolo_v8', 'dataset', 'test', 'images', data)
    results = model(source=path, conf=0.4)

    results = sorted(results[0].boxes.data.tolist())
    data = []
    for i in results:
        if i[5] != 10.0:
            data.append(str(int(i[5])))
        else:
            data.append('.')

    return ''.join(data)


if __name__ == "__main__":
    print('loading model....', end='\t')
    model = YOLO(os.path.join(os.getcwd(), 'yolo_v8', 'runs', 'detect', 'train', 'weights', 'best.pt'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print('complete')
    app.run(host='0.0.0.0', port=7001, debug=True)
