import numpy as np

file_path = 'Dataset/test/raw/test_003_raw.npy'
raw_data = np.load(file_path)
N = 50
idx_start = 10000
idx_end = idx_start + 2048*N
raw_segment = raw_data[idx_start:idx_end]
segment_filename = file_path.replace('.npy', '_segment.npy')
np.save(segment_filename, raw_segment)