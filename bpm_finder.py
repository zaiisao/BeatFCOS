import os

dataset_dir = "../../beat-tracking-dataset/labeled_data/train"

annotation_dims = []
beat_intervals, downbeat_intervals = [], []

#for dataset in ["ballroom", "hains", "carnatic"]:
for dataset in ["ballroom", "hains"]:
    root_path = os.getcwd() + "/" + dataset_dir + "/" + dataset + "/label"
    file_names = os.listdir(root_path)

    bpms = []

    for file_name in file_names:
        if file_name == ".DS_Store":
            continue

        f = open(os.path.join(root_path, file_name))
      
        lines = [line.rstrip('\n') for line in f.readlines()]

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

            beat_times.append(float(beat_time))

        number_of_beats = len(beat_times)
        beat_audio_length = max(beat_times) - min(beat_times)

        bpms.append(number_of_beats/beat_audio_length * 60)

    print(f"Minimum BPM for {dataset} | Min: {min(bpms) // 1} | Max: {max(bpms) // 1} | Average: {sum(bpms) // len(bpms)}")