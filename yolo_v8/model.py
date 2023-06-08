# import pprint
# import matplotlib.pyplot as plt
# from roboflow import Roboflow
# rf = Roboflow(api_key="wsjSrLiP8udUZFXd6FAT")
# project = rf.workspace().project("test-ld42b")
# model = project.version(1).model
#
# # infer on a local image
# pprint.pprint(model.predict("D:/Pycharm_Project/FlightVisualizator_OCR/yolo_v8/datasets/img/no-u-lim/0/0.png", confidence=40, overlap=30).json())
#
# # visualize your prediction
#
# model.predict("D:/Pycharm_Project/FlightVisualizator_OCR/yolo_v8/datasets/img/no-u-lim/0/0.png", confidence=40, overlap=30).save('test.png')
#
# # infer on an image hosted elsewhere
# # print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())

if __name__ == '__main__':
    import os
    from ultralytics import YOLO

    model = YOLO("yolov8m.yaml")

    model.train(data=os.path.join(os.getcwd(), 'dataset', 'data.yaml'), epochs=250)
    metrics = model.val()
    path = model.export(format="onnx")
