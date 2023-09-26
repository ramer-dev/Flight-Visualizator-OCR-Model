import os, torch, time, re, json
from ultralytics import YOLO
from glob import glob
from yolo_v8.preprocess import preprocess

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


def inference(file):
    [filename, ext] = file.split(os.path.sep)[-1].split('.')
    directory = os.path.join('.', 'yolo_v8', 'result')
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, str(filename) + '.json')

    site, img_path = preprocess(file)
    # Evaluate Process
    # imgs = glob(os.path.join(os.getcwd(), 'yolo_v8', f'{os.path.sep}*{os.path.sep}*.png')
    imgs = sum(
        [glob(img_path + f'{os.path.sep}{i}{os.path.sep}*.png') for i in range(0, 20)], [])
    results = model(source=imgs, conf=0.5, stream=True)

    row = {"ocr": [], 'site': site}
    col = {}
    for box in results:
        
        sep = box.path.split(os.path.sep)

        id, folder = sep[-1].split('.')[0], sep[-2]
        
        os.makedirs(os.path.join(img_path,  folder, id), exist_ok=True)

        txt = open(os.path.join(img_path,  folder, f'{id}.txt'), 'w')

        # box 검출 좌표순으로 정렬
        xywhn = sorted(box.boxes.xywhn.tolist())
        boxes = sorted(box.boxes.data.tolist())

        for [x,y,w,h], i in zip(xywhn,boxes):
            txt.write(' '.join([str(int(i[5])), str(x), str(y), str(w), str(h)]) + '\n')
        
        txt.close()
        
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


if __name__ == '__main__':
    model = YOLO(os.path.join(os.getcwd(),  'yolo_v8', 'runs',
                 'detect', 'train', 'weights', 'best.pt'))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
# scan/20230925183245_00002.jpg
    file_names = glob(os.getcwd()+f'{os.path.sep}scan{os.path.sep}*.jpg')
    for file_name in file_names:
        # 추론 한번에 0.6초 소모
        inference(os.path.join(os.getcwd(), 'scan', file_name))
