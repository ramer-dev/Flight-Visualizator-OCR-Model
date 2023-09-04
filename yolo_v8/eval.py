import torch, os, time
from ultralytics import YOLO

model = None


def inference(file):
    results = model(source=file, save=True)
    results = sorted(results[0].boxes.data.tolist())

    data = []
    for i in results:
        if i[5] != 10.0:
            data.append(str(int(i[5])))
        else:
            data.append('.')



if __name__ == '__main__':
    model = YOLO(os.path.join(os.getcwd(), 'runs', 'detect', 'train', 'weights', 'best.pt'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    file_name = '0_png.rf.375dbc86621799ce67ded2d2f78c2656.jpg'
    file_path = os.path.join(os.getcwd(), 'dataset', 'train', 'images', file_name)

    # 추론 한번에 0.6초 소모
    inference(file_path)
