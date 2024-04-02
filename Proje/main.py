import cv2
import numpy as np


def draw_center_circle(frame, x, y):
    radius = 3
    color = (0, 0, 255)
    thickness = -1
    cv2.circle(frame, (x, y), radius, color, thickness)


def check_weight_center(frame, line_y, center_x, center_y):

    line_height = 4


    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), line_height)


    if center_y > line_y and center_y < line_y + line_height:
        return 1
    else:
        return 0

def main():
    video_path = 'video4.mp4'
    cap = cv2.VideoCapture(video_path)

    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    classes = []
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    line_y = 450
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 600))
        height, width, _ = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id in [2, 7]:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                draw_center_circle(frame, center_x, center_y)

                counter += check_weight_center(frame, line_y, center_x, center_y)

        cv2.putText(frame, f'Counter: {counter}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
