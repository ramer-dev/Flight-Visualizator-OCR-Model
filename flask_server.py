from flask import Flask, request

print('-------------------------')
print('initializing....', end='\t')

import torch, os
from yolo_v8.preprocess import preprocess
from ultralytics import YOLO

print('complete')
# load_inference = load.signatures["serving_default"]  # 이게 무슨뜻?

app = Flask('ocr')
model = None


@app.route('/', methods=['POST'])
def main():
    print('request inference')
    data = request.json
    data = data['data']

    # Preprocess
    [site, directory] = preprocess(data)

    # Evaluate
    # print(site, directory)
    glob = directory + '/'

    results = model(source=glob, stream=True, conf=0.4)
    for i in results:
        print(i)

    return 'main'


@app.route('/inference', methods=['POST'])
def inference():
    print('request inference')
    data = request.json
    data = data['data']

    path = os.path.join(os.getcwd(), 'yolo_v8', 'dataset', 'test', 'images', data)
    results = model(source=path, conf=0.4)
    print(results[0].boxes.data)

    results = sorted(results[0].boxes.data.tolist())
    data = []
    for i in results:
        if i[5] != 10.0:
            data.append(str(int(i[5])))
        else:
            data.append('.')

    print(data)
    return ''.join(data)


if __name__ == "__main__":
    print('loading model....', end='\t')
    model = YOLO(os.path.join(os.getcwd(), 'yolo_v8', 'runs', 'detect', 'train', 'weights', 'best.pt'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print('complete')

    app.run(debug=True, port=1234)
