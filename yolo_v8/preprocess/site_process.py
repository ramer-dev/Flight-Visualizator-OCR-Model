import cv2
import configparser


def site_processor(img, bound):
    config = configparser.ConfigParser()
    config.read('site_config.ini', encoding='UTF-8')

    x, y, w, h = cv2.boundingRect(bound)
    site_img = img[y:y + h, x:x + w]

    site_col_count = 12
    site_row_count = 4

    site_cell_width = round(w / site_col_count)
    site_cell_height = round(h / site_row_count)

    site_img_edit = site_img.copy()

    # 외곽선 지우기
    cv2.rectangle(site_img_edit, (x, y), (x + w, y + h), (255, 255, 255), 3)
    cv2.floodFill(site_img_edit, None, (0, 0), 255)
    cv2.floodFill(site_img_edit, None, (0, 0), 0)

    site_selected = {"selected_id": 0, "site_name": "", "val": 0}
    for row in range(1, site_row_count, 2):
        for col in range(site_col_count):
            mean = cv2.mean(site_img_edit[row * site_cell_height: (row + 1) * site_cell_height,
                            col * site_cell_width: (col + 1) * site_cell_width])[0]
            if site_selected['val'] < mean:
                site_selected = {"selected_id": (row % 2) * (col + 1) - 1, "val": mean}
                site_selected["site_name"] = config["site"][str(site_selected["selected_id"])]

    return site_selected
