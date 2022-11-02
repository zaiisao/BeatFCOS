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

    ref_beats, est_beats, _ = find_beats(t_beats.numpy(), 
                                        p_beats.numpy(), 
                                        beat_type="beat",
                                        sample_rate=target_sample_rate)

    ref_downbeats, est_downbeats, _ = find_beats(t_downbeats.numpy(), 
                                                p_downbeats.numpy(), 
                                                beat_type="downbeat",
                                                sample_rate=target_sample_rate)

    if use_dbn:
        beat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
            min_bpm=55,
            max_bpm=215,
            transition_lambda=100,
            fps=target_sample_rate,
            online=False)

        downbeat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
            min_bpm=10,
            max_bpm=75,
            transition_lambda=100,
            fps=target_sample_rate,
            online=False)

        beat_pred = pred[0,:].clamp(1e-8, 1-1e-8).view(-1).numpy()
        downbeat_pred = pred[1,:].clamp(1e-8, 1-1e-8).view(-1).numpy()

        est_beats = beat_dbn.process_offline(beat_pred)
        est_downbeats = downbeat_dbn.process_offline(downbeat_pred)

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

            # if we have metadata, it is only during evaluation where batch size is always 1
            metadata = metadata[0]

            nblocks = 10

            target_length = -(audio.size(dim=2) // -2**nblocks) * 2**nblocks
            audio_pad = (0, target_length - audio.size(dim=2))
            audio = torch.nn.functional.pad(audio, audio_pad, "constant", 0)

            if torch.cuda.is_available():
                # move data to GPU
                audio = audio.to('cuda')
                target = target.to('cuda')

            # run network
            if torch.cuda.is_available():
                #scores, labels, boxes = model(audio.permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
                scores, labels, boxes, losses = model((audio, target))
            else:
                #scores, labels, boxes = model(audio.permute(2, 0, 1).float().unsqueeze(dim=0))
                scores, labels, boxes, losses = model((audio, target))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()
            losses = [loss.cpu() for loss in losses]

            # correct boxes for image scale
            #boxes /= scale

            # append image to list of processed images
            # image_ids.append(dataset.image_ids[index])

            # print progress
            #print('{}/{}'.format(index, len(dataset)), end='\r')

            length = audio.size(dim=2) // 256

            wavebeat_format_pred = torch.zeros((2, length))
            wavebeat_format_target = torch.zeros((2, length))

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
                row = 1 - label
                left_position_index = int(box[0])
                right_position_index = int(box[1])

                wavebeat_format_pred[row, left_position_index] = 1

                if label == 0 and (last_pred_downbeat_index is None or right_position_index > last_pred_downbeat_index):
                    last_pred_downbeat_index = right_position_index
                elif label == 1 and (last_pred_beat_index is None or right_position_index > last_pred_beat_index):
                    last_pred_beat_index = right_position_index

            if last_pred_beat_index is not None:
                wavebeat_format_pred[0, min(last_pred_beat_index, length - 1)] = 1

            if last_pred_downbeat_index is not None:
                wavebeat_format_pred[1, min(last_pred_downbeat_index, length - 1)] = 1

            # construct target tensor
            for beat_interval in target[0]:
                label = int(beat_interval[2])
                row = 1 - label

                left_position_index = int(beat_interval[0])
                right_position_index = int(beat_interval[1])

                wavebeat_format_target[row, min(left_position_index, length - 1)] = 1

                if label == 0 and (last_target_downbeat_index is None or right_position_index > last_target_downbeat_index):
                    last_target_downbeat_index = right_position_index
                elif label == 1 and (last_target_beat_index is None or right_position_index > last_target_beat_index):
                    last_target_beat_index = right_position_index

            wavebeat_format_target[0, min(last_target_beat_index, length - 1)] = 1
            wavebeat_format_target[1, min(last_target_downbeat_index, length - 1)] = 1

            target_sample_rate = 22050 // 256

            beat_scores, downbeat_scores = evaluate(wavebeat_format_pred.view(2,-1),  
                                                    wavebeat_format_target.view(2,-1), 
                                                    target_sample_rate,
                                                    use_dbn=False)

            try:
                dbn_beat_scores, dbn_downbeat_scores = evaluate(wavebeat_format_pred.view(2,-1), 
                                                        wavebeat_format_target.view(2,-1), 
                                                        target_sample_rate,
                                                        use_dbn=True)
            except:
                dbn_beat_scores = { 'F-measure': 0 }
                dbn_downbeat_scores = { 'F-measure': 0 }

            # print(f"{index}/{len(dataset)} {metadata['Filename']} CLS: {losses[0]} | REG: {losses[1]}")
            print(f"(PEAK) BEAT (F-measure): {beat_scores['F-measure']:0.3f} | DOWNBEAT (F-measure): {downbeat_scores['F-measure']:0.3f}")
            print(f"(DBN)  BEAT (F-measure): {dbn_beat_scores['F-measure']:0.3f} | DOWNBEAT (F-measure): {dbn_downbeat_scores['F-measure']:0.3f}\n")

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
                        'losses': losses,
                        'beat_scores': beat_scores,
                        'downbeat_scores': downbeat_scores,
                        'dbn_beat_scores': dbn_beat_scores,
                        'dbn_downbeat_scores': dbn_downbeat_scores
                    }

                    # append detection to results
                    results.append(image_result)

        # if not len(results):
        #     return

        beat_mean_f_measure = np.mean([result['beat_scores']['F-measure'] for result in results])
        downbeat_mean_f_measure = np.mean([result['downbeat_scores']['F-measure'] for result in results])
        dbn_beat_mean_f_measure = np.mean([result['dbn_beat_scores']['F-measure'] for result in results])
        dbn_downbeat_mean_f_measure = np.mean([result['dbn_downbeat_scores']['F-measure'] for result in results])

        # print(f"Average beat score: {beat_mean_f_measure:0.3f}")
        # print(f"Average downbeat score: {downbeat_mean_f_measure:0.3f}")
        # print(f"(DBN) Average beat score: {dbn_beat_mean_f_measure:0.3f}")
        # print(f"(DBN) Average downbeat score: {dbn_downbeat_mean_f_measure:0.3f}")

        model.train()

        return beat_mean_f_measure, downbeat_mean_f_measure, dbn_beat_mean_f_measure, dbn_downbeat_mean_f_measure
