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
    print("p1", p)

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
    print("p2", p)
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

        print(beat_pred.sum())

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
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : metadata["Filename"],
                        #'category_id' : dataset.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            # image_ids.append(dataset.image_ids[index])

            # print progress
            #print('{}/{}'.format(index, len(dataset)), end='\r')
            print(f'{index}/{len(dataset)} CLS: {losses[0]} | REG: {losses[1]}')

        if not len(results):
            return

        # # write output
        # json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)

        # # load results in COCO evaluation tool
        # coco_true = dataset.coco
        # coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

        # # run COCO evaluation
        # coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        # coco_eval.params.imgIds = image_ids
        # coco_eval.evaluate()
        # coco_eval.accumulate()
        # coco_eval.summarize()

        model.train()

        return
