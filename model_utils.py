import numpy as np
import cv2

np.set_printoptions(threshold=0.6)



# Preprocess the image for the YOLO model
def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape
    #print("{},{}".format(new_h, new_w))
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) / new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) / new_h
        new_h = net_h

    new_w = int(new_w)
    new_h = int(new_h)
    #print("{},{}".format(new_h, new_w))
    resized = cv2.resize(image[:, :, ::-1] / 255., (int(new_w), int(new_h)))

    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[int((net_h - new_h) // 2):int((net_h + new_h) // 2), int((net_w - new_w) // 2):int((net_w + new_w) // 2), :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image

# Decode the output of the network
def decode_netout(netout, anchors, obj_thresh, nms_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = int(i / grid_w)
        col = i % grid_w

        for b in range(nb_box):
            objectness = netout[row][col][b][4]
            if objectness <= obj_thresh:
                continue

            x, y, w, h = netout[row][col][b][:4]

            x = (col + x) / grid_w
            y = (row + y) / grid_h
            w = anchors[2 * b + 0] * np.exp(w) / net_w
            h = anchors[2 * b + 1] * np.exp(h) / net_h

            classes = netout[row][col][b][5:]

            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)

            boxes.append(box)

    return boxes

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score

# Correct the YOLO boxes to fit the original image dimensions
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

 # Apply non-maxima suppression to suppress overlapping boxes
def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue

            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

# Calculate the Intersection Over Union (IOU) of two bounding boxes
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3

def preprocess_image(image_path, net_h, net_w):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read the image: {image_path}")
    image_h, image_w, _ = image.shape
    new_image = preprocess_input(image, net_h, net_w)
    return image, new_image, image_h, image_w

def get_nested_list_info(data):
    def get_dimensions(lst):
        if not isinstance(lst, list) or not lst:
            return 0
        return 1 + get_dimensions(lst[0])

    def count_nested_lists(lst):
        if not isinstance(lst, list):
            return 0
        return 1 + sum(count_nested_lists(item) for item in lst)

    def get_shape(lst):
        if isinstance(lst, list) and lst:
            return [len(lst)] + get_shape(lst[0])
        return []

    def get_types(lst):
        if isinstance(lst, list) and lst:
            return {type(lst[0]): get_types(lst[0])}
        return type(lst)

    if isinstance(data, str):
        data = eval(data)  # Convert string to Python object if needed

    dimensions = get_dimensions(data)
    nested_list_count = count_nested_lists(data) - 1  # subtract 1 to exclude the outermost list
    shape = get_shape(data)
    element_type = get_types(data)

    return {
        'dimensions': dimensions,
        'nested_list_count': nested_list_count,
        'shape': shape,
        'type': element_type,
    }

def process_yolo_response(response_yolos, anchors, obj_thresh, nms_thresh, net_h, net_w, image_h, image_w):
    
    yolos = response_yolos
    boxes = []
    for i in range(len(yolos)):
        boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)
    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
    do_nms(boxes, nms_thresh)
    return boxes

def predict_and_process_yolo(yolo_model, new_image, anchors, labels, obj_thresh, nms_thresh, net_h, net_w, image_h, image_w):
    
    yolos = yolo_model.predict(new_image)
    
    boxes = []
    for i in range(len(yolos)):
        boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)
    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
    do_nms(boxes, nms_thresh)
    return boxes

def get_first_detected_label(boxes, labels, obj_thresh):
    for box in boxes:
        score = box.get_score()
        if score > obj_thresh:  # Only consider boxes with a score above the threshold
            label = labels[box.get_label()]
            return label
    return

def get_all_detected_labels(boxes, labels, obj_thresh):
    detected_labels = []
    for box in boxes:
        score = box.get_score()
        if score > obj_thresh:  # Only consider boxes with a score above the threshold
            label = labels[box.get_label()]
            detected_labels.append(label)
    return detected_labels



