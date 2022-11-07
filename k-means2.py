""" A program which performs k-means clustering to get the the dimentions of anchor boxes """
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import sys
import config

matplotlib.use('agg')
#file_path = config.train_annotations_file

num_anchors = 5

def resize_image(image_data, size):
    """ Resizes the image without changing the aspect ratio with padding, so that
        the image size is as per model requirement.
        Input:
            image_data: array, original image data
            size: tuple, size the image is to e resized into
        Output:
            image: array, image data after resizing the image
    """

    image_height, image_width, _ = image_data.shape
    input_height, input_width = size

    # Getting the scale that is to be used for resizing the image
    scale = min(input_width / image_width, input_height / image_height)
    new_width = int(image_width * scale) # new image width
    new_height = int(image_height * scale) # new image height

    # getting the number of pixels to be padded
    dx = (input_width - new_width)
    dy = (input_height - new_height)

    # resizing the image
    image = cv2.resize(image_data, (new_width, new_height), 
        interpolation=cv2.INTER_CUBIC)


    top, bottom = dy//2, dy-(dy//2)
    left, right = dx//2, dx-(dx//2)

    color = [128, 128, 128] # color pallete to be used for padding
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color) # padding
    
    return new_image

def iou(boxes, clusters):
    """ Calculates Intersection over Union between the provided boxes and cluster centroids
        Input:
            boxes: Bounding boxes
            clusters: cluster centroids
        Output:
            IoU between boxes and cluster centroids
    """
    n = boxes.shape[0]
    k = np.shape(clusters)[0]

    #box_area = boxes[:, 0] * boxes[:, 1] # Area = width * height
    box_area = boxes[:]
    # Repeating the area for every cluster as we need to calculate IoU with every cluster
    box_area = box_area.repeat(k)
    box_area = np.reshape(box_area, (n, k))

    #cluster_area = clusters[:, 0] * clusters[:, 1]
    cluster_area = clusters[:]
    cluster_area = np.tile(cluster_area, [1, n])
    cluster_area = np.reshape(cluster_area, (n, k))


    #box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
    box_w_matrix = np.reshape(boxes.repeat(k), (n, k))
    #cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
    cluster_w_matrix = np.reshape(np.tile(clusters, (1, n)), (n, k))
    min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

    # box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
    # cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
    # min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
    # inter_area = np.multiply(min_w_matrix, min_h_matrix)
    inter_area = min_w_matrix

    result = inter_area / (box_area + cluster_area - inter_area)
    return result


def avg_iou(boxes, clusters):
    """ Calculates average IoU between the GT boxes and clusters 
        Input:
            boxes: array, having width and height of all the GT boxes
        Output:
            Returns numpy array of average IoU with all the clusters
    """
    return np.mean([np.max(iou(boxes, clusters), axis=1)])


def kmeans(boxes, k):
    """ Executes k-means clustering on the rovided width and height of boxes with IoU as
        distance metric.
        Input:
            boxes: numpy array containing with and height of all the boxes
        Output:
            clusters after convergence
    """
    num_boxes = boxes.shape[0]
    distances = np.empty((num_boxes, k))
    last_cluster = np.zeros((num_boxes, ))

    # Initializing the clusters
    np.random.seed()
    clusters = boxes[np.random.choice(num_boxes, k, replace=False)]

    # Optimizarion loop
    while True:

        distances = 1 - iou(boxes, clusters)
        mean_distance = np.mean(distances)
        sys.stdout.write('\r>> Mean loss: %f' % (mean_distance))
        sys.stdout.flush()

        current_nearest = np.argmin(distances, axis=1)
        if(last_cluster == current_nearest).all():
            break # The model is converged
        for cluster in range(k):
            clusters[cluster] = np.mean(boxes[current_nearest == cluster], axis=0)

        last_cluster = current_nearest

    return clusters


def dump_results(data):
    """ Writes the anchors after running k-means clustering onto the disk for further usage
        Input:
            data: array, containing the data for anchor boxes
    """
    f = open("./anchors.txt", 'w')
    row = np.shape(data)[0]
    for i in range(row):
        #x_y = "%d %d\n" % (data[i][0], data[i][1])
        x_y = "%d\n" % (data[i])
        f.write(x_y)
    f.close()


def get_boxes(file_path):
    """ Extracts the bounding boxes from the coco train.txt file 
        Input:
            file_path: path of train.txt made from coco annotations
        Output:
            numpy array containing all the bouding boxes width and height
    """
    with open(file_path, 'r') as f:
        dataSet = []
        for line in f:
            infos = line.split(' ')
            length = len(infos)
            sys.stdout.write('\r>> Reading image: %s' % (infos[0].split('/')[-1]))
            sys.stdout.flush()
            img = cv2.imread(infos[0])
            image_width, image_height = img.shape[1], img.shape[0]
            scale = np.minimum(config.input_shape / image_width, config.input_shape / image_height)
            new_width, new_height = image_width * scale, image_height * scale
            dx = (config.input_shape - new_width) / 2
            dy = (config.input_shape - new_height) / 2
            # In every line of train.txt the values are stored as:
            # [relative_image_path, x1, y1, x2, y2, class_id]
            for i in range(1, length):
                xmin, xmax = int(infos[i].split(',')[0]), int(infos[i].split(',')[2])
                ymin, ymax = int(infos[i].split(',')[1]), int(infos[i].split(',')[3])
                xmin = int(xmin * new_width/image_width + dx)
                xmax = int(xmax * new_width/image_width + dx)
                ymin = int(ymin * new_height/image_height + dy)
                ymax = int(ymax * new_height/image_height + dy)
                width = xmax - xmin
                height = ymax - ymin
                if (width == 0) or (height == 0):
                    continue
                dataSet.append([width, height])
    result = np.array(dataSet)
    return result


def get_clusters(num_clusters):
    """ Calls all the required functions to run k-means and get good anchor boxes 
        Input:
            num_clusters: number of clusters
            file_path: path of train.txt containing parsed annotations 
        Output:
            Returns avg_accuracy of computer anchor box over the whole dataset
    """
    #all_boxes = get_boxes(file_path)

    input_dir = "../../beat-tracking-dataset/labeled_data/train"
    output_dir = "save"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
 
    annotation_dims = []
    beat_intervals, downbeat_intervals = [], []

    #for dataset in ["ballroom", "hains", "carnatic"]:
    for dataset in ["ballroom", "hains"]:
        root_path = os.getcwd() + "/" + input_dir + "/" + dataset + "/label"
        file_names = os.listdir(root_path)

        for file_name in file_names:
            if file_name == ".DS_Store":
                continue

            f = open(os.path.join(root_path, file_name))
          
            lines = [line.rstrip('\n') for line in f.readlines()]
            #print(f"lines: {lines}")
         
        #     for line in lines:
        # #        line = line.replace('JPEGImages','labels')
        #         #line = line.replace('JPEGImages','YOLO')
        #         line = line.replace('.wav','.beats')
        #         #line = line.replace('.png','.txt')
        # #        print(line)
        #         f2 = open(line)
        #         for line in f2.readlines():
        #             line = line.rstrip('\n')
        # #            print(line)
        #             temp = line.strip().split(" ")
        # #            print(temp)
                    
        #             w = float(temp[3])
        #             h = float(temp[4])
        # #            w,h = line.split(' ')[3:]            
        #             #print(w,h)
        #             annotation_dims.append((float(w),float(h)))

            beat_times, downbeat_times = [], []
            for line in lines:
                if dataset == "carnatic":
                    beat_time, beat_type = line.split(",")
                elif dataset == "ballroom" or dataset == "hains" or dataset == "beatles":
                    line = line.replace('\t', ' ')
                    if dataset == "beatles":
                        line = line.replace('  ', ' ')
                    beat_time, beat_type = line.split(" ")

                if beat_type == "1":
                    downbeat_times.append(float(beat_time))
                #else:
                #    beat_times.append(float(beat_time))
                beat_times.append(float(beat_time))

            for beat_index, current_beat_location in enumerate(beat_times[:-1]):
                next_beat_location = beat_times[beat_index + 1]

                beat_length = (next_beat_location - current_beat_location) #* 22050
                annotation_dims.append(beat_length)
                beat_intervals.append(beat_length)
                #print(downbeat_length)

            for downbeat_index, current_downbeat_location in enumerate(downbeat_times[:-1]):
                next_downbeat_location = downbeat_times[downbeat_index + 1]

                downbeat_length = (next_downbeat_location - current_downbeat_location)# * 22050
                annotation_dims.append(downbeat_length)
                downbeat_intervals.append(downbeat_length)
                #print(downbeat_length)
            #break
        #print(f"beat lengths: {[ '%.2f' % elem for elem in beat_intervals ]}")
        #print(f"downbeat lengths: {[ '%.2f' % elem for elem in downbeat_intervals ]}")

    all_boxes = np.array(annotation_dims)
    #print([ '%.2f' % elem for elem in annotation_dims ])

    print('\n')
    result = kmeans(all_boxes, num_clusters)
    #result = result[np.lexsort(result.T[0, None])]
    #result = result.sort()
    dump_results(result)
    print("\n\n{} anchors:\n{}".format(num_clusters, result))
    print("\n\n{} anchors:\n{}".format(num_clusters, 60//result))
    avg_acc = avg_iou(all_boxes, result)*100
    print("Average accuracy: {:.2f}%".format(avg_acc))

    return avg_acc


if __name__ == '__main__':

    min_cluster, max_cluster = num_anchors, num_anchors + 1
    clusters = np.arange(min_cluster, max_cluster, dtype=int)
    avg_accuracy = []
    for i in clusters:
        avg_acc = get_clusters(i)
        avg_accuracy.append(avg_acc)

    if max_cluster - min_cluster > 1:
        plt.plot(clusters, avg_accuracy)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Average Accuracy')
        plt.savefig('./cluster.png')
