#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = torch.load('C:/Users/gml40/PycharmProjects/Flight-Visualizator-OCR-Model/yolo_v8/runs/detect/train/weights/last.pt', map_location=device)
#
# print(model)
# with torch.no_grad():
#     model.eval()
#     input = "C:/Users/gml40/PycharmProjects/Flight-Visualizator-OCR-Model/yolo_v8/dataset/valid/images/0_png.rf.6b29f3242f1617147db1b5da409a057d.jpg"
#     output = model(input)
#     print(output)

if __name__ == '__main__':
    import torch
    from ultralytics import YOLO
    model = YOLO('C:/Users/gml40/PycharmProjects/Flight-Visualizator-OCR-Model/yolo_v8/runs/detect/train/weights/last.pt')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # result = model("C:/Users/gml40/PycharmProjects/Flight-Visualizator-OCR-Model/yolo_v8/dataset/valid/images/0_png.rf.3f6808bef1fb25a1f114cadd717ed5e8.jpg")
    results = model(source="C:/Users/gml40/PycharmProjects/Flight-Visualizator-OCR-Model/yolo_v8/dataset/valid/images/0_png.rf.6b29f3242f1617147db1b5da409a057d.jpg",
                        show=True, show_labels=True)

    print(results[0].boxes.boxes)

    # for result in results:
    #     print(result)