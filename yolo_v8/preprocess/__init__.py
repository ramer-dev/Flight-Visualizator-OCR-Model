from .site_process import site_processor
from .image_splitter import image_split

import cv2, os


def preprocess(image, save=True):
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--image", type=str, required=True, help="image_name")
    # parser.add_argument("--test", type=int, default=0, help="Popup Img Views")
    # args = parser.parse_args()
    img_name = os.path.join(os.getcwd(), 'yolo_v8', 'img', image)

    image = cv2.imread(img_name)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 7), 0)

    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 6)
    # if args.test:
    #     cv2.imshow("ctr", thresh)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_array = sorted(contours, key=cv2.contourArea, reverse=True)

    # if args.test:
    #     x, y, w, h = cv2.boundingRect(contour_array[0])
    #     cv2.imshow("ctr", thresh[y + 10:y + h - 10, x + 10:x + w - 10])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # 선택한 표지소 리턴

    site = site_processor(thresh, contour_array[1])
    img_arr = image_split(thresh, contour_array[0], filename=img_name, save=save)

    return [site['site_name'], os.path.join(os.getcwd(), 'yolo_v8', 'datasets', 'img', img_name)]
