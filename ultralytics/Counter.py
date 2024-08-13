from collections import defaultdict

import cv2
import numpy as np


def putTextWithWrap(img,
                    text_list,
                    org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                    max_width=500):
    y = org[1]

    for text in text_list:
        words = text.split(' ')
        lines = []
        current_line = words[0]

        # Create an image to measure text size
        test_img = img.copy()

        for word in words[1:]:
            # Check the size of the current line with the next word added
            size, _ = cv2.getTextSize(current_line + ' ' + word, fontFace, fontScale, thickness)
            if size[0] <= max_width:
                current_line += ' ' + word
            else:
                lines.append(current_line)
                current_line = word

        lines.append(current_line)

        # Draw the lines on the image
        for line in lines:
            cv2.putText(img, line, (org[0], y), fontFace, fontScale, color, thickness, lineType)
            y += size[1] + 10  # Move y coordinate for next line

    return img


def point2line_distance(p, line) -> float:
    """
    利用向量计算点到直线的距离
    如果点不在线上，则返回点到直线两端的最短距离
    Parameters
    ----------
    p : tuple
        点 (x, y)
    line : list
        直线，包含起点与终点 ((x1, y1), (x2, y2))

    Returns
    -------
    float
        距离，如果是正向则为正，反之为负
    """
    # 提取点和直线的坐标
    point = np.array(p)
    line_start = np.array(line[0])
    line_end = np.array(line[1])

    # 计算向量
    line_vec = line_end - line_start
    vec_ap = point - line_start

    # 计算两个向量的叉乘
    cross = np.cross(line_vec, vec_ap)

    # 计算距离
    return np.cross(line_vec, vec_ap) / np.linalg.norm(line_vec) if cross != 0 else min(point2point(p, line[0]),
                                                                                        point2point(p, line[1]))


def point2point(p1, p2) -> float:
    """
    求两点的欧氏距离，欧嘉俊式距离
    :param p1: 一个点
    :param p2: 另一个点
    :return: 距离
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class Counter:
    def __init__(self, names, lines):
        self.track_history = defaultdict(list)
        self.names = names
        self.lines = lines
        self.line_datas = [[{key: 0 for key in names}, {key: 0 for key in names}, set(), line[2], None] for line in
                           lines]
        self.yolo_names = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush"
        }

    def update_track_history(self, id, box, dis=1.5):
        """
        更新车辆的历史轨迹
        :param dis: 车辆轨迹更新的阈值
        :param id: 跟踪轨迹编号
        :param box: 跟踪框
        :return: None
        """
        now = ((box[0] + box[1]) / 2, box[3])
        if len(self.track_history[id]) and point2point(now, self.track_history[id][-1]) >= dis:
            self.track_history[id].append(now)
            if len(self.track_history[id]) > 2:
                self.track_history[id].pop(0)
        else:
            self.track_history[id].append(now)

    def box_line_judge(self, box) -> int:
        """
        判断box距离哪个线最近，返回线的索引
        Parameters
        ----------
        box 检测框，包含左上角和右下角的坐标
        lines 多条线，包含每条线的起点和终点

        Returns
        -------
        int 线的索引
        """

        mid_p = ((box[0] + box[2]) / 2, box[3])  # box的下中点
        dis = [abs(point2line_distance(mid_p, line)) for line in self.lines]
        return dis.index(min(dis))

    def draw_box_and_line(self, img, line_index):
        """
        将检测线和检测结果绘制到图片上
        Parameters
        ----------
        img 图片
        line 检测线
        line_datas 检测结果

        Returns
        -------
        无
        """
        # 检测线绘制设置
        color = (0, 255, 0)
        start_point = self.lines[line_index][0]  # 横线起始点
        end_point = self.lines[line_index][1]  # 横线终结点
        cv2.line(img, start_point, end_point, color, thickness=5)  # 画线

        # box检测结果字体等绘制设置
        color_car = (0, 0, 255)
        org = self.line_datas[line_index][3]  # 文字起始位置
        font = cv2.FONT_HERSHEY_COMPLEX
        fontScale = 1
        thickness = 3

        # 绘制结果
        result = []
        for name in self.names:
            result.append(
                f"{self.yolo_names[name]}: In:{self.line_datas[line_index][0][name]}, Out:{self.line_datas[line_index][1][name]}, last id:{self.line_datas[line_index][4]}")
        putTextWithWrap(img, result, org, font, fontScale, color_car, thickness)

    def angle_of_two(self, id, vec2=np.array([1, 0])) -> float:
        """
        计算box移动方向与指定向量的夹角
        :param vec2: 指定向量，默认为w轴正方向
        :param id: 跟踪的id
        :return: 与vec2的夹角
        """
        vec = np.array(self.track_history[id][1]) - np.array(self.track_history[id][0])
        return np.arccos(np.dot(vec, vec2) / (np.linalg.norm(vec) * 1))

    def count_obj(self, box, line_index: int, id: int, cls=0, dis=60, angle_dis=0.875):
        """
        通过跟踪结果（box和id），计算进入判定线和离开判定线的车辆数量。
        统一规定，判定线为由起点指向终点的向量。
        进入判定线则说明车辆前进方向与判定线夹角为pi/2，方向向判定线左侧；离开判定线则说明前进方向在判定线右侧。
        Parameters
        ----------
        box 检测框，包含左上角和右下角的坐标
        line_index 判定线的索引
        id 跟踪的id
        cls 类别，暂时无用，可以以后拓展
        dis 邻域宽度。
        angle_dis 行驶方向的夹角阈值

        """
        down_mid = ((box[0] + box[2]) / 2, box[3])  # box的下中点
        dis_down_mid = point2line_distance(down_mid, self.lines[line_index])  # 计算下中点到判定线的距离
        # 根据历史轨迹判断行驶方向，所需要的历史轨迹点至少有两个

        if len(self.track_history[id]) >= 2 and 0 < abs(dis_down_mid) < dis:
            # 说明box的底边中点在邻域中
            right = np.array([1, np.tan(self.lines[line_index][3])])  # 构建判定线正向的向量
            if abs(self.angle_of_two(id, right)) <= angle_dis and id not in self.line_datas[line_index][2]:
                # 说明box向判定线的正方向移动，且底边中点在邻域中
                self.line_datas[line_index][1][cls] += 1  # 离开判定线的车辆数量加1
                self.line_datas[line_index][2].add(id)
                self.line_datas[line_index][4] = id

            elif abs(self.angle_of_two(id, -right)) <= angle_dis and id not in self.line_datas[line_index][
                2]:  # 构建判定线负向的向量
                # 说明box向判定线的负方向移动，且底边中点在邻域中
                self.line_datas[line_index][0][cls] += 1  # 进入判定线的车辆数量加1
                self.line_datas[line_index][2].add(id)
                self.line_datas[line_index][4] = id
        # 历史轨迹少于两个点的情况，不做处理
