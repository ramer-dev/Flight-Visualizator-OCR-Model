import cv2
import numpy as np
import os
import math
import shutil


def image_split(img, bound, filename='', save=True):
    x, y, w, h = cv2.boundingRect(bound)
    padding = 15
    table_data = img[y + padding:y + h - padding, x + padding:x + w - padding]

    # cv2.imshow('table', table_data)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    col_count = 8
    row_count = 20

    cell_width = round((w - padding * 2) / col_count)
    cell_height = round((h - padding * 2) / row_count)
    inner_padding = (1, 1, 1, 1)
    mask = np.zeros((cell_height, cell_width))

    mask[inner_padding[0]: -inner_padding[2], inner_padding[3]:-inner_padding[1]] = 1

    coord = [0, 0]
    img_arr = []
    for y_ in range(row_count):
        row = []
        # if y_ == 8:
        #     break

        for x_ in range(col_count):
            resize = table_data[coord[1]:coord[1] + cell_height, coord[0]:coord[0] + cell_width]

            if len(resize) < len(mask) or len(resize[0]) < len(mask[0]):
                resize = cv2.copyMakeBorder(
                    resize,
                    top=0,
                    left=0,
                    bottom=len(mask) - len(resize),
                    right=len(mask[0]) - len(resize[0]),
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )

            bitwise = resize * mask

            border = cv2.copyMakeBorder(
                bitwise.copy(),
                top=math.floor((256 - cell_height) / 2),
                bottom=math.ceil((256 - cell_height) / 2),
                left=math.floor((256 - cell_width) / 2),
                right=math.ceil((256 - cell_width) / 2),
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )
            resized_img = np.array(cv2.resize(border, dsize=(640, 640), interpolation=cv2.INTER_AREA))

            # cv2.floodFill(resized_img, None, (round(128 - cell_width / 2), round(128 - cell_height / 2)), 0)

            # horizontal = np.copy(resized_img)
            # vertical = np.copy(resized_img)
            # cols = resized_img.shape[1]
            # horizontal_size = cols // 15
            # horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
            #
            # horizontal = cv2.erode(horizontal, horizontal_structure)
            # horizontal = cv2.dilate(horizontal, horizontal_structure)
            #
            # rows = vertical.shape[0]
            # vertical_size = rows // 30
            #
            # vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
            #
            # vertical = cv2.erode(vertical, vertical_structure)
            # vertical = cv2.dilate(vertical, vertical_structure)
            #
            # vertical = cv2.bitwise_not(vertical)
            #
            # edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, -2)
            #
            # kernel = np.ones((2,2), np.uint8)
            # edges = cv2.dilate(edges, kernel)
            #
            # smooth = np.copy(vertical)
            #
            # smooth = cv2.blur(smooth, (2,2))
            #
            # (rows, cols) = np.where(edges != 0)
            # vertical[rows, cols] = smooth[rows, cols]
            #
            # filtered_image = cv2.medianBlur(resized_img, ksize=3)

            row.append(resized_img)

            coord[0] += cell_width

            # cv2.imshow(str(y_), resized_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        img_arr.append(np.array(row))
        coord[1] += cell_height
        coord[0] = 0

    img_arr = np.array(img_arr)

    if save:
        # print(filename)
        filename = filename.split(os.path.sep)[-1].split('.')[0]
        path = os.path.join(os.getcwd(), 'yolo_v8', 'datasets', 'img', filename)
        os.makedirs(path, exist_ok=True)

        for i in range(len(img_arr)):
            os.makedirs(os.path.join(path, str(i)), exist_ok=True)

            black_count = 0
            for j in range(len(img_arr[i])):
                cv2.imwrite(os.path.join(os.getcwd(), 'yolo_v8', 'datasets', 'img', filename, f'{i}', f'{j}.png'),
                            img_arr[i][j])
                if cv2.countNonZero(img_arr[i][j]) < 20:
                    black_count += 1

                if black_count > 3:
                    shutil.rmtree(os.path.join(os.getcwd(), 'yolo_v8', 'datasets', 'img', filename, str(i)))
                    break

                # if not cv2.imwrite(
                #         os.path.join(os.getcwd(),'datasets','img',f'{i}', f'{j}.png'),
                #         img_arr[i][j]):
                #     raise Exception("could not write image")
                # cv2.imshow("test",img_arr[i][j])
                # if cv2.waitKey() == 27:
                #     break
        print('image saved in :', os.path.join(os.getcwd(), 'yolo_v8', 'datasets', 'img', filename))
    print(img_arr.shape)
    return img_arr

    # return (row, col, height, width)
