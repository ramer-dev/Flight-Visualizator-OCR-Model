import cv2
import argparse
import numpy as np
import os

from preprocess import site_processor, image_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, required=True, help="image_name")
    parser.add_argument("--scan", type=int, default=1, help="Flight Test Scanned image(1) or not(0)")
    parser.add_argument("--test", type=int, default=0, help="Popup Img Views")
    parser.add_argument("--save", type=int, default=0, help="Save Seperated Images")
    args = parser.parse_args()

    image = cv2.imread(args.image)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if args.test:
        cv2.imshow("ctr", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_array = sorted(contours, key=cv2.contourArea, reverse=True)

    # 선택한 표지소 리턴

    if args.scan:
        site = site_processor(thresh, contour_array[1])
        img_arr = image_split(thresh, contour_array[0], filename=args.image, save=args.save)

        print(site)
        print(img_arr.shape)
