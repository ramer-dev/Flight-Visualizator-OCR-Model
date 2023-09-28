from .site_process import site_processor
from .image_splitter import image_split

import cv2, os


def preprocess(image_file_name, image_path=f'{os.path.sep}.', save=True):
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--image", type=str, required=True, help="image_name")
    # parser.add_argument("--test", type=int, default=0, help="Popup Img Views")
    # args = parser.parse_args()
    [filename, ext] = image_path.split(os.path.sep)[-1].split('.')
    # img_name = os.path.join('yolo_v8', 'img', filename)

    img_name = os.path.join(os.getcwd(),'yolo_v8', 'img', image_file_name)
    image = cv2.imread(img_name)
    # image = cv2.imread(image_path)

    img = image
    print(img_name)
    # cv2.imshow("ctr", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    img = cv2.GaussianBlur(img, (3, 7), 0)

    # ret, thresh = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 5)
    # if args.test:
    # cv2.imshow("ctr", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_array = sorted(contours, key=cv2.contourArea, reverse=True)

    # if args.test:
    #     x, y, w, h = cv2.boundingRect(contour_array[0])
    #     cv2.imshow("ctr", thresh[y + 10:y + h - 10, x + 10:x + w - 10])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # 선택한 표지소 리턴
    img_arr = image_split(thresh, contour_array[0], filename=image_path, save=save)

    site = site_processor(thresh, contour_array[1])
    return site['site_name'], os.path.join(os.getcwd(), 'yolo_v8', 'datasets', 'img', image_file_name)

