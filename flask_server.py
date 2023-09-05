import json

from flask import Flask, request
from glob import glob
import re

print('-------------------------')
print('initializing....', end='\t')

import torch, os
from yolo_v8.preprocess import preprocess
from ultralytics import YOLO

print('complete')
# load_inference = load.signatures["serving_default"]  # 이게 무슨뜻?

app = Flask('ocr')
model = None
score_pattern = re.compile('\d1\d')

print('loading model....', end='\t')
model = YOLO(os.path.join(os.getcwd(), 'yolo_v8', 'runs', 'detect', 'train', 'weights', 'best.pt'))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print('complete')

def score_validator(data: str):
    if data == '1':
        return None

    if score_pattern.match(data):
        score = re.findall(score_pattern, data)[0]
        score_slash = f"{score[0]}/{score[2]}"
        return score_slash
    else:
        return None


@app.route('/', methods=['POST'])
def main():
    data = request.json
    filename = data['data']['originalname']
    print(f'request inference \t :: {filename}')

    directory = f'./yolo_v8/result'
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, str(filename).split('.')[0] + '.json')

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    # Preprocess
    [site, directory] = preprocess(filename)
    # Evaluate Process
    imgs = glob(directory + '\\*\\*.png')

    results = model(source=imgs, stream=True, conf=0.4)

    row = {"ocr": [], 'site': site}
    col = {}
    no = 0

    for box in results:
        # box 검출 좌표순으로 정렬
        boxes = sorted(box.boxes.data.tolist())
        data = ''
        # 컬럼 데이터 리셋
        if no == 8:
            no = 0
            col = {}

        # Box내 데이터 리턴
        for j in boxes:
            if j[5] != 10.0:
                data += str(int(j[5]))
            else:
                data += '.'

        if no == 0:
            col['frequency'] = data
        elif no == 1:
            col['txmain'] = score_validator(data)
        elif no == 2:
            col['rxmain'] = score_validator(data)
        elif no == 3:
            col['txstby'] = score_validator(data)
        elif no == 4:
            col['rxstby'] = score_validator(data)
        elif no == 5:
            col['angle'] = data
        elif no == 6:
            col['distance'] = data
        elif no == 7:
            col['height'] = data
            row['ocr'].append(col)
        no += 1

    with open(file_path, 'w') as json_file:
        json.dump(row, json_file)
    print(row)
    return row


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

    app.run(debug=True, port=7001)
