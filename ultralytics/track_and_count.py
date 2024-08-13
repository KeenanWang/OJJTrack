import cv2

from Counter import Counter
# from Counter import Counter
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

model = YOLO("yolov8l.pt", verbose=False)
cap = cv2.VideoCapture("../data/202408071905_test.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# 类别
names = [2, 3, 5, 7]
# 检测线（起点、终点、标签开始点、正方向角度）
lines = (

    ((1057, 700), (2, 859), (465, 778), 0.0),
    ((1686, 730), (1170, 730), (1012, 579), 1.5707963),

)
counter = Counter(names=names, lines=lines)
# Video writer
video_writer = cv2.VideoWriter("../data/counting_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


def extract_and_process_tracks(tracks, annotator):
    """Extracts and processes tracks for object counting in a video stream."""

    if tracks[0].boxes.id is not None:
        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        track_ids = tracks[0].boxes.id.int().cpu().tolist()

        # Extract tracks
        for box, track_id, cls in zip(boxes, track_ids, clss):
            counter.update_track_history(track_id, box)
            annotator.box_label(box,
                                label=f"{model.names[cls]}#{track_id}#{counter.angle_of_two(track_id) if len(counter.track_history[track_id]) >= 2 else 0}",
                                color=colors(int(track_id), True))
            counter.count_obj(box, counter.box_line_judge(box), track_id, cls)


while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    annotator = Annotator(im0, 2)
    tracks = model.track(im0, persist=True, show=False, classes=names)
    extract_and_process_tracks(tracks, annotator)
    im0 = annotator.result()
    for i in range(len(lines)):
        counter.draw_box_and_line(im0, i)

    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
