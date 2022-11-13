import madmom
import mir_eval
import numpy as np
import scipy.signal
import torch

def find_beats(t, p, 
                smoothing=127, 
                threshold=0.5, 
                distance=None, 
                sample_rate=44100, 
                beat_type="beat",
                filter_type="none",
                peak_type="simple"):
    # 15, 2

    # t is ground truth beats
    # p is predicted beats 
    # 0 - no beat
    # 1 - beat

    N = p.shape[-1]

    if filter_type == "savgol":
        # apply smoothing with savgol filter
        p = scipy.signal.savgol_filter(p, smoothing, 2)   
    elif filter_type == "cheby":
        sos = scipy.signal.cheby1(10,
                                  1, 
                                  45, 
                                  btype='lowpass', 
                                  fs=sample_rate,
                                  output='sos')
        p = scipy.signal.sosfilt(sos, p)

    # normalize the smoothed signal between 0.0 and 1.0
    p /= np.max(np.abs(p))                                     

    if peak_type == "simple":
        # by default, we assume that the min distance between beats is fs/4
        # this allows for at max, 4 BPS, which corresponds to 240 BPM 
        # for downbeats, we assume max of 1 downBPS
        if beat_type == "beat":
            if distance is None:
                distance = sample_rate / 4
        elif beat_type == "downbeat":
            if distance is None:
                distance = sample_rate / 2
        else:
            raise RuntimeError(f"Invalid beat_type: `{beat_type}`.")

        # apply simple peak picking
        est_beats, heights = scipy.signal.find_peaks(p, height=threshold, distance=distance)

    elif peak_type == "cwt":
        # use wavelets
        est_beats = scipy.signal.find_peaks_cwt(p, np.arange(1,50))

    # compute the locations of ground truth beats
    ref_beats = np.squeeze(np.argwhere(t==1).astype('float32'))
    est_beats = est_beats.astype('float32')

    # compute beat points (samples) to seconds
    ref_beats /= float(sample_rate)
    est_beats /= float(sample_rate)

    # store the smoothed ODF
    est_sm = p

    return ref_beats, est_beats, est_sm

def evaluate(pred, target, target_sample_rate, use_dbn=False):

    t_beats = target[0,:]
    t_downbeats = target[1,:]
    p_beats = pred[0,:]
    p_downbeats = pred[1,:]

    # ref_beats, est_beats, _ = find_beats(t_beats.numpy(), 
    #                                     p_beats.numpy(), 
    #                                     beat_type="beat",
    #                                     sample_rate=target_sample_rate)

    # ref_downbeats, est_downbeats, _ = find_beats(t_downbeats.numpy(), 
    #                                             p_downbeats.numpy(), 
    #                                             beat_type="downbeat",
    #                                             sample_rate=target_sample_rate)

    # if use_dbn:
    #     beat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
    #         min_bpm=55,
    #         max_bpm=215,
    #         transition_lambda=100,
    #         fps=target_sample_rate,
    #         online=False)

    #     downbeat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
    #         min_bpm=10,
    #         max_bpm=75,
    #         transition_lambda=100,
    #         fps=target_sample_rate,
    #         online=False)

    #     beat_pred = pred[0,:].clamp(1e-8, 1-1e-8).view(-1).numpy()
    #     downbeat_pred = pred[1,:].clamp(1e-8, 1-1e-8).view(-1).numpy()

    #     est_beats = beat_dbn.process_offline(beat_pred)
    #     est_downbeats = downbeat_dbn.process_offline(downbeat_pred)

    # evaluate beats - trim beats before 5 seconds.
    ref_beats = mir_eval.beat.trim_beats(ref_beats)
    est_beats = mir_eval.beat.trim_beats(est_beats)
    beat_scores = mir_eval.beat.evaluate(ref_beats, est_beats)

    # evaluate downbeats - trim beats before 5 seconds.
    ref_downbeats = mir_eval.beat.trim_beats(ref_downbeats)
    est_downbeats = mir_eval.beat.trim_beats(est_downbeats)
    downbeat_scores = mir_eval.beat.evaluate(ref_downbeats, est_downbeats)

    return beat_scores, downbeat_scores

def evaluate_beat(dataset, model, threshold=0.05):
    model.eval()
    
    with torch.no_grad():
        # start collecting results
        results = []
        image_ids = []

        for index, data in enumerate(dataset):
            #data = dataset[index]
            #scale = data['scale']
            audio, target, metadata = data
            #audio, target = data

            if torch.cuda.is_available():
                # move data to GPU
                audio = audio.to('cuda')
                target = target.to('cuda')

            # if we have metadata, it is only during evaluation where batch size is always 1
            metadata = metadata[0]

            nblocks = 10

            target_length = -(audio.size(dim=2) // -2**nblocks) * 2**nblocks
            audio_pad = (0, target_length - audio.size(dim=2))
            audio = torch.nn.functional.pad(audio, audio_pad, "constant", 0)

            # run network
            if torch.cuda.is_available():
                #scores, labels, boxes = model(audio.permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
                scores, labels, boxes = model((audio, target))
            else:
                #scores, labels, boxes = model(audio.permute(2, 0, 1).float().unsqueeze(dim=0))
                scores, labels, boxes = model((audio, target))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()

            # correct boxes for image scale
            #boxes /= scale

            # append image to list of processed images
            # image_ids.append(dataset.image_ids[index])

            # print progress
            #print('{}/{}'.format(index, len(dataset)), end='\r')

            #length = audio.size(dim=2) // 256
            length = audio.size(dim=2) // 128

            # wavebeat_format_pred_left = torch.zeros((2, length)).to(audio.device)
            # wavebeat_format_pred_average = torch.zeros((2, length)).to(audio.device)
            # wavebeat_format_pred_right = torch.zeros((2, length)).to(audio.device)
            # wavebeat_format_pred_weighted = torch.zeros((2, length)).to(audio.device)

            # wavebeat_format_target = torch.zeros((2, length)).to(audio.device)

            # box_scores_left = torch.zeros((2, length)).to(audio.device)
            # box_scores_right = torch.zeros((2, length)).to(audio.device)

            beat_pred_left_positions = []
            downbeat_pred_left_positions = []

            # first_pred_beat_index, first_pred_downbeat_index = None, None
            last_pred_beat_index, last_pred_downbeat_index = None, None
            last_target_beat_index, last_target_downbeat_index = None, None

            # construct pred tensor
            for box_id in range(boxes.shape[0]):
                score = float(scores[box_id])
                label = int(labels[box_id])
                box = boxes[box_id, :]

                # scores are sorted, so we can break
                if score < threshold:
                    continue

                # if beat (label 1), first row (index 0)
                # if downbeat (label 0), second row (index 1)
                # row = 1 - label
                left_position_index = int(box[0])
                right_position_index = int(box[1])

                if label == 0:
                    downbeat_pred_left_positions.append(left_position_index * 128 / 22050)
                elif label == 1:
                    beat_pred_left_positions.append(left_position_index * 128 / 22050)

                # wavebeat_format_pred_left[row, min(left_position_index, length - 1)] = 1
                # wavebeat_format_pred_right[row, min(right_position_index, length - 1)] = 1

                # box_scores_left[row, min(left_position_index, length - 1)] = score
                # box_scores_right[row, min(right_position_index, length - 1)] = score

                # if label == 0 and (first_pred_downbeat_index is None or left_position_index < first_pred_downbeat_index):
                #     first_pred_downbeat_index = left_position_index
                # elif label == 1 and (first_pred_beat_index is None or left_position_index < first_pred_beat_index):
                #     first_pred_beat_index = left_position_index

                if label == 0 and (last_pred_downbeat_index is None or right_position_index > last_pred_downbeat_index):
                    last_pred_downbeat_index = right_position_index
                elif label == 1 and (last_pred_beat_index is None or right_position_index > last_pred_beat_index):
                    last_pred_beat_index = right_position_index

            if last_pred_beat_index is not None:
                #wavebeat_format_pred_left[0, min(last_pred_beat_index, length - 1)] = 1
                beat_pred_left_positions.append(last_pred_beat_index * 128 / 22050)

            if last_pred_downbeat_index is not None:
                #wavebeat_format_pred_left[1, min(last_pred_downbeat_index, length - 1)] = 1
                downbeat_pred_left_positions.append(last_pred_downbeat_index * 128 / 22050)

            # if first_pred_beat_index is not None:
            #     wavebeat_format_pred_right[0, min(first_pred_beat_index, length - 1)] = 1

            # if first_pred_downbeat_index is not None:
            #     wavebeat_format_pred_right[1, min(first_pred_downbeat_index, length - 1)] = 1

            # box_scores_left = torch.nn.functional.pad(box_scores_left[box_scores_left != 0], (1, 1), "constant", 1)
            # box_scores_right = torch.nn.functional.pad(box_scores_right[box_scores_right != 0], (1, 1), "constant", 1)

            # positive_bbox_indices_left = wavebeat_format_pred_left.nonzero()
            # positive_bbox_indices_right = wavebeat_format_pred_right.nonzero()

            # left_weights = torch.clamp(positive_bbox_indices_left * box_scores_left[box_scores_left.nonzero()] / (
            #     positive_bbox_indices_left * box_scores_left[box_scores_left.nonzero()] +
            #     positive_bbox_indices_right * box_scores_right[box_scores_right.nonzero()]
            # ), min=0.0, max=0.5)

            # right_weights = torch.clamp(positive_bbox_indices_right * box_scores_right[box_scores_right.nonzero()] / (
            #     positive_bbox_indices_left * box_scores_left[box_scores_left.nonzero()] +
            #     positive_bbox_indices_right * box_scores_right[box_scores_right.nonzero()]
            # ), min=0.0, max=0.5)

            # average_indices = torch.round(positive_bbox_indices_left * 0.5 + positive_bbox_indices_right * 0.5)
            # for index_pair in average_indices:
            #     wavebeat_format_pred_average[index_pair[0].long(), index_pair[1].long()] = 1

            # weighted_indices = torch.round(positive_bbox_indices_left * left_weights + positive_bbox_indices_right * right_weights)
            # for index_pair in weighted_indices:
            #     index_pair[0] = 0 if torch.isnan(index_pair[0]) else 1
            #     index_pair[1] = torch.nan_to_num(index_pair[1], nan=0.5)
            #     wavebeat_format_pred_weighted[index_pair[0].long(), index_pair[1].long()] = 1


            beat_target_left_positions = []
            downbeat_target_left_positions = []
            # construct target tensor
            for beat_interval in target[0]:
                label = int(beat_interval[2])
                # row = 1 - label

                left_position_index = int(beat_interval[0])
                right_position_index = int(beat_interval[1])

                # wavebeat_format_target[row, min(left_position_index, length - 1)] = 1
                if label == 0:
                    downbeat_target_left_positions.append(left_position_index * 128 / 22050)
                elif label == 1:
                    beat_target_left_positions.append(left_position_index * 128 / 22050)

                if label == 0 and (last_target_downbeat_index is None or right_position_index > last_target_downbeat_index):
                    last_target_downbeat_index = right_position_index
                elif label == 1 and (last_target_beat_index is None or right_position_index > last_target_beat_index):
                    last_target_beat_index = right_position_index

            #wavebeat_format_target[0, min(last_target_beat_index, length - 1)] = 1
            #wavebeat_format_target[1, min(last_target_downbeat_index, length - 1)] = 1
            beat_target_left_positions.append(last_target_beat_index * 128 / 22050)

            downbeat_target_left_positions.append(last_target_downbeat_index * 128 / 22050)

            #target_sample_rate = 22050 // 256
            # target_sample_rate = 22050 // 128

            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()
            
            # beat_scores_left, downbeat_scores_left = evaluate(wavebeat_format_pred_left.view(2,-1),  
            #                                         wavebeat_format_target.view(2,-1), 
            #                                         target_sample_rate,
            #                                         use_dbn=False)
            beat_target_left_positions = np.array(beat_target_left_positions)
            beat_pred_left_positions = np.array(beat_pred_left_positions)
            downbeat_target_left_positions = np.array(downbeat_target_left_positions)
            downbeat_pred_left_positions = np.array(downbeat_pred_left_positions)

            beat_target_left_positions.sort()
            beat_pred_left_positions.sort()
            downbeat_target_left_positions.sort()
            downbeat_pred_left_positions.sort()
            # print(beat_target_left_positions)
            # print(beat_pred_left_positions)

            beat_target_left_positions = mir_eval.beat.trim_beats(beat_target_left_positions)
            beat_pred_left_positions = mir_eval.beat.trim_beats(beat_pred_left_positions)
            beat_scores = mir_eval.beat.evaluate(beat_target_left_positions, beat_pred_left_positions)

            # evaluate downbeats - trim beats before 5 seconds.
            downbeat_target_left_positions = mir_eval.beat.trim_beats(downbeat_target_left_positions)
            downbeat_pred_left_positions = mir_eval.beat.trim_beats(downbeat_pred_left_positions)
            downbeat_scores = mir_eval.beat.evaluate(downbeat_target_left_positions, downbeat_pred_left_positions)

            # try:
            #     dbn_beat_scores_left, dbn_downbeat_scores_left = evaluate(wavebeat_format_pred_left.view(2,-1), 
            #                                             wavebeat_format_target.view(2,-1), 
            #                                             target_sample_rate,
            #                                             use_dbn=True)
            # except:
            #     dbn_beat_scores_left = { 'F-measure': 0 }
            #     dbn_downbeat_scores_left = { 'F-measure': 0 }

            # beat_scores_right, downbeat_scores_right = evaluate(wavebeat_format_pred_right.view(2,-1),  
            #                                         wavebeat_format_target.view(2,-1), 
            #                                         target_sample_rate,
            #                                         use_dbn=False)

            # try:
            #     dbn_beat_scores_right, dbn_downbeat_scores_right = evaluate(wavebeat_format_pred_right.view(2,-1), 
            #                                             wavebeat_format_target.view(2,-1), 
            #                                             target_sample_rate,
            #                                             use_dbn=True)
            # except:
            #     dbn_beat_scores_right = { 'F-measure': 0 }
            #     dbn_downbeat_scores_right = { 'F-measure': 0 }

            # beat_scores_average, downbeat_scores_average = evaluate(wavebeat_format_pred_average.view(2,-1),  
            #                                         wavebeat_format_target.view(2,-1), 
            #                                         target_sample_rate,
            #                                         use_dbn=False)

            # try:
            #     dbn_beat_scores_average, dbn_downbeat_scores_average = evaluate(wavebeat_format_pred_average.view(2,-1), 
            #                                             wavebeat_format_target.view(2,-1), 
            #                                             target_sample_rate,
            #                                             use_dbn=True)
            # except:
            #     dbn_beat_scores_average = { 'F-measure': 0 }
            #     dbn_downbeat_scores_average = { 'F-measure': 0 }

            # beat_scores_weighted, downbeat_scores_weighted = evaluate(wavebeat_format_pred_weighted.view(2,-1),  
            #                                         wavebeat_format_target.view(2,-1), 
            #                                         target_sample_rate,
            #                                         use_dbn=False)

            # try:
            #     dbn_beat_scores_weighted, dbn_downbeat_scores_weighted = evaluate(wavebeat_format_pred_weighted.view(2,-1), 
            #                                             wavebeat_format_target.view(2,-1), 
            #                                             target_sample_rate,
            #                                             use_dbn=True)
            # except:
            #     dbn_beat_scores_weighted = { 'F-measure': 0 }
            #     dbn_downbeat_scores_weighted = { 'F-measure': 0 }


            print(f"{index}/{len(dataset)} {metadata['Filename']}")
            print(f"BEAT (F-measure): {beat_scores['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores['F-measure']:0.3f}")
            #print("LEFT")
            # print(f"BEAT (F-measure): {beat_scores_left['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores_left['F-measure']:0.3f}")
            # print(f"(DBN)  BEAT (F-measure): {dbn_beat_scores_left['F-measure']:0.3f} | DOWNBEAT (F-measure): {dbn_downbeat_scores_left['F-measure']:0.3f}")
            #print("RIGHT")
            #print(f"BEAT (F-measure): {beat_scores_right['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores_right['F-measure']:0.3f}")
            # print(f"(DBN)  BEAT (F-measure): {dbn_beat_scores_right['F-measure']:0.3f} | DOWNBEAT (F-measure): {dbn_downbeat_scores_right['F-measure']:0.3f}")
            # print("AVERAGE")
            # print(f"BEAT (F-measure): {beat_scores_average['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores_average['F-measure']:0.3f}")
            # print(f"(DBN)  BEAT (F-measure): {dbn_beat_scores_average['F-measure']:0.3f} | DOWNBEAT (F-measure): {dbn_downbeat_scores_average['F-measure']:0.3f}")
            # print("WEIGHTED")
            # print(f"BEAT (F-measure): {beat_scores_weighted['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores_weighted['F-measure']:0.3f}")
            # print(f"(DBN)  BEAT (F-measure): {dbn_beat_scores_weighted['F-measure']:0.3f} | DOWNBEAT (F-measure): {dbn_downbeat_scores_weighted['F-measure']:0.3f}\n")

            # beat_scores = beat_scores_left
            # downbeat_scores = downbeat_scores_left
            # dbn_beat_scores = dbn_beat_scores_left
            # dbn_downbeat_scores = dbn_downbeat_scores_left

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                #boxes[:, 2] -= boxes[:, 0]
                #boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        # break
                        continue

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id': metadata["Filename"],
                        #'category_id': dataset.label_to_coco_label(label),
                        'score': float(score),
                        'bbox': box.tolist(),
                        'beat_scores': beat_scores,
                        'downbeat_scores': downbeat_scores,
                        # 'beat_scores_left': beat_scores_left,
                        # 'downbeat_scores_left': downbeat_scores_left,
                        # 'beat_scores_right': beat_scores_right,
                        # 'downbeat_scores_right': downbeat_scores_right,
                        # 'beat_scores_average': beat_scores_average,
                        # 'downbeat_scores_average': downbeat_scores_average,
                        # 'beat_scores_weighted': beat_scores_weighted,
                        # 'downbeat_scores_weighted': downbeat_scores_weighted,
                        # 'dbn_beat_scores': dbn_beat_scores,
                        # 'dbn_downbeat_scores': dbn_downbeat_scores
                    }

                    # append detection to results
                    results.append(image_result)
        # END for index, data in enumerate(dataset)

        # if not len(results):
        #     return

        beat_mean_f_measure = np.mean([result['beat_scores']['F-measure'] for result in results])
        downbeat_mean_f_measure = np.mean([result['downbeat_scores']['F-measure'] for result in results])
        # left_beat_mean_f_measure = np.mean([result['beat_scores_left']['F-measure'] for result in results])
        # left_downbeat_mean_f_measure = np.mean([result['downbeat_scores_left']['F-measure'] for result in results])
        #right_beat_mean_f_measure = np.mean([result['beat_scores_right']['F-measure'] for result in results])
        #right_downbeat_mean_f_measure = np.mean([result['downbeat_scores_right']['F-measure'] for result in results])
        # average_beat_mean_f_measure = np.mean([result['beat_scores_average']['F-measure'] for result in results])
        # average_downbeat_mean_f_measure = np.mean([result['downbeat_scores_average']['F-measure'] for result in results])
        # weighted_beat_mean_f_measure = np.mean([result['beat_scores_weighted']['F-measure'] for result in results])
        # weighted_downbeat_mean_f_measure = np.mean([result['downbeat_scores_weighted']['F-measure'] for result in results])
        #dbn_beat_mean_f_measure = np.mean([result['dbn_beat_scores']['F-measure'] for result in results])
        #dbn_downbeat_mean_f_measure = np.mean([result['dbn_downbeat_scores']['F-measure'] for result in results])

        print(f"Average beat F-measure: {beat_mean_f_measure:0.3f}")
        print(f"Average downbeat F-measure: {downbeat_mean_f_measure:0.3f}")
        # print(f"Average left beat F-measure: {left_beat_mean_f_measure:0.3f}")
        # print(f"Average left downbeat F-measure: {left_downbeat_mean_f_measure:0.3f}")
        #print(f"Average right beat F-measure: {right_beat_mean_f_measure:0.3f}")
        #print(f"Average right downbeat F-measure: {right_downbeat_mean_f_measure:0.3f}")
        # print(f"Average average beat F-measure: {average_beat_mean_f_measure:0.3f}")
        # print(f"Average average downbeat F-measure: {average_downbeat_mean_f_measure:0.3f}")
        # print(f"Average weighted beat F-measure: {weighted_beat_mean_f_measure:0.3f}")
        # print(f"Average weighted downbeat F-measure: {weighted_downbeat_mean_f_measure:0.3f}")
        print()
        # print(f"(DBN) Average beat score: {dbn_beat_mean_f_measure:0.3f}")
        # print(f"(DBN) Average downbeat score: {dbn_downbeat_mean_f_measure:0.3f}")

        model.train()

        # beat_mean_f_measure = left_beat_mean_f_measure#average_beat_mean_f_measure
        # downbeat_mean_f_measure = left_downbeat_mean_f_measure#average_downbeat_mean_f_measure
        return beat_mean_f_measure, downbeat_mean_f_measure, 0, 0#dbn_beat_mean_f_measure, dbn_downbeat_mean_f_measure
