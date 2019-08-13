import pandas as pd
import os

directory_txt = '/SSD/bogunoda-a/HMDB_data/splits/'
save_to = '/SSD/bogunoda-a/HMDB_data/'

filenames = list(os.listdir(directory_txt))
len_all_files = len(filenames)
print(len_all_files)
with open(os.path.join(save_to, 'all_splits.txt'), 'w') as outfile:
    for i, fname in enumerate(filenames):
        if fname.endswith('1.txt'):
            print('Processing {} ({} out of {})'.format(fname, i, len_all_files))
            with open(os.path.join(directory_txt,fname)) as infile:
                for line in infile:
                    outfile.write(line)


