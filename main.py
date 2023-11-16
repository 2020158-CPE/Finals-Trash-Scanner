import cv2
import argparse
import numpy

from ultralytics import YOLO

#https://discuss.pytorch.org/t/torchvision-fails-when-trying-to-use-the-gpu/176898
#https://www.youtube.com/watch?v=QtsI0TnwDZs&list=WL&index=4&t=202s
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description ="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def plot_boxes(results,frame):
    xyxys = []
    confidences = []
    class_ids = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        xyxys = boxes.xyxy
    for xyxy in xyxys:
        cv2.rectangle(frame,(int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3])),(0,255,0),2)
    #xyxys.append(boxes.xyxy)
    confidences.append(boxes.conf)
    class_ids.append(boxes.cls)
    print(confidences)

    return frame

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("trash.pt")
    model.to('cuda')

    while True:
        ret, frame = cap.read()
        results = model(frame,show=True)
       
        frame = plot_boxes(results, frame)
        #cv2.imshow("",frame)
        if (cv2.waitKey(30)==27):
            break
    

if __name__=="__main__":
    main()