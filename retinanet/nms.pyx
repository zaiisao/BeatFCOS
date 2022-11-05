import numpy as np
#cimport numpy as np

#def cpu_soft_nms(np.ndarray[float, ndim=2] boxes, float sigma=0.5, float Nt=0.3, float threshold=0.001, unsigned int method=0):
def cpu_soft_nms(boxes, sigma=0.5, iou_threshold=0.3, score_threshold=0.001, method=0):
    N = boxes.shape[0]
    iw, ih, box_area = None, None, None
    ua = None
    pos = 0
    maxscore = 0
    maxpos = 0
    #cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov
    x1,x2,tx1,tx2,ts,area,weight,ov = None, None, None, None, None, None, None, None

    for i in range(N):
        #maxscore = boxes[i, 4]
        maxscore = boxes[i, 2]
        maxpos = i

        # tx1 = boxes[i,0]
        # ty1 = boxes[i,1]
        # tx2 = boxes[i,2]
        # ty2 = boxes[i,3]
        # ts = boxes[i,4]
        tx1 = boxes[i,0]
        tx2 = boxes[i,1]
        ts = boxes[i,2]

        pos = i + 1
	# get max box
        while pos < N:
            # if maxscore < boxes[pos, 4]:
            #     maxscore = boxes[pos, 4]
            if maxscore < boxes[pos, 2]:
                maxscore = boxes[pos, 2]
                maxpos = pos
            pos = pos + 1

	# add max box as a detection 
        # boxes[i,0] = boxes[maxpos,0]
        # boxes[i,1] = boxes[maxpos,1]
        # boxes[i,2] = boxes[maxpos,2]
        # boxes[i,3] = boxes[maxpos,3]
        # boxes[i,4] = boxes[maxpos,4]
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]

	# swap ith box with position of max box
        # boxes[maxpos,0] = tx1
        # boxes[maxpos,1] = ty1
        # boxes[maxpos,2] = tx2
        # boxes[maxpos,3] = ty2
        # boxes[maxpos,4] = ts
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = tx2
        boxes[maxpos,2] = ts

        # tx1 = boxes[i,0]
        # ty1 = boxes[i,1]
        # tx2 = boxes[i,2]
        # ty2 = boxes[i,3]
        # ts = boxes[i,4]
        tx1 = boxes[i,0]
        tx2 = boxes[i,1]
        ts = boxes[i,2]

        pos = i + 1
	# NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            # x1 = boxes[pos, 0]
            # y1 = boxes[pos, 1]
            # x2 = boxes[pos, 2]
            # y2 = boxes[pos, 3]
            # s = boxes[pos, 4]
            x1 = boxes[pos, 0]
            x2 = boxes[pos, 1]
            s = boxes[pos, 2]

            # area = (x2 - x1 + 1) * (y2 - y1 + 1)
            # iw = (min(tx2, x2) - max(tx1, x1) + 1)
            area = x2 - x1 + 1
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                # ih = (min(ty2, y2) - max(ty1, y1) + 1)
                # if ih > 0:
                # ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                ua = float((tx2 - tx1 + 1) + area - iw)
                # ov = iw * ih / ua #iou between max box and detection box
                ov = iw / ua #iou between max box and detection box

                if method == 1: # linear
                    if ov > iou_threshold: 
                        weight = 1 - ov
                    else:
                        weight = 1
                elif method == 2: # gaussian
                    weight = np.exp(-(ov * ov)/sigma)
                else: # original NMS
                    if ov > iou_threshold: 
                        weight = 0
                    else:
                        weight = 1

                # boxes[pos, 4] = weight*boxes[pos, 4]
                boxes[pos, 2] = weight*boxes[pos, 2]
		    
		    # if box score falls below threshold, discard the box by swapping with last box
		    # update N
                    # if boxes[pos, 4] < threshold:
                    #     boxes[pos,0] = boxes[N-1, 0]
                    #     boxes[pos,1] = boxes[N-1, 1]
                    #     boxes[pos,2] = boxes[N-1, 2]
                    #     boxes[pos,3] = boxes[N-1, 3]
                    #     boxes[pos,4] = boxes[N-1, 4]
                    if boxes[pos, 2] < score_threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep
