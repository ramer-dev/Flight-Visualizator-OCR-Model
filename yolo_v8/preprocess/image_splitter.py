import cv2
import numpy as np
import os
import math


def image_split(img, bound, filename='', save=0):
    x, y, w, h = cv2.boundingRect(bound)
    table_data = img[y:y + h, x:x + w]

    col_count = 8
    row_count = 20

    cell_width = round(w / col_count)
    cell_height = round(h / row_count)
    print(cell_height)
    x_padding = 10
    y_padding = 8
    mask = np.zeros((cell_height, cell_width))

    mask[y_padding: -y_padding, x_padding:-x_padding] = 1

    coord = [x, y]
    img_arr = []
    for y_ in range(row_count):
        row = []
        for x_ in range(col_count):
            # if x_ == 0:
            # continue

            position = [coord[0], coord[1]]

            # cv2.rectangle(img, (position[0], position[1]),
            #               (position[0] + cell_width,
            #                position[1] + cell_height), (255, 255, 255),
            #               thickness=3)

            resize = img[position[1]:position[1] + cell_height,
                     position[0]:position[0] + cell_width]
            # bitwise = cv2.multiply(resize,mask)

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
            resized_img = cv2.resize(border, dsize=(256, 256), interpolation=cv2.INTER_AREA)


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

            cv2.imshow("ctr", resized_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            coord[0] += cell_width
        if y_ > 2:
            break

        img_arr.append(row)
        coord[1] += cell_height
        coord[0] = x

    img_arr = np.array(img_arr)

    if save:
        filename = filename.split('.')[0]
        if not os.path.isdir(os.path.join(os.getcwd(), 'datasets', 'img', filename)):
            print(os.getcwd())
            os.mkdir(os.path.join(os.getcwd(), 'datasets', 'img', filename))

        for i in range(len(img_arr)):
            if not os.path.isdir(os.path.join(os.getcwd(), 'datasets', 'img', filename, str(i))):
                os.mkdir(os.path.join(os.getcwd(), 'datasets', 'img', filename, str(i)))
            for j in range(len(img_arr[i])):
                print(os.path.join(os.getcwd(), 'datasets', 'img', filename, f'{i}', f'{j}.png'))
                cv2.imwrite(os.path.join(os.getcwd(), 'datasets', 'img', filename, f'{i}', f'{j}.png'),
                            img_arr[i][j])
                # if not cv2.imwrite(
                #         os.path.join(os.getcwd(),'datasets','img',f'{i}', f'{j}.png'),
                #         img_arr[i][j]):
                #     raise Exception("could not write image")
                # cv2.imshow("test",img_arr[i][j])
                # if cv2.waitKey() == 27:
                #     break
    return img_arr

    # return (row, col, height, width)
