import os


def preprocess_splits(hmdb_directory):
    directory_txt = os.path.join(hmdb_directory, 'splits')
    save_to = hmdb_directory

    filenames = list(os.listdir(directory_txt))
    len_all_files = len(filenames)

    with open(os.path.join(save_to, 'all_splits.txt'), 'w') as outfile:
        for i, fname in enumerate(filenames):
            if fname.endswith('1.txt'):
                print('Processing {}'.format(fname, i, len_all_files))
                with open(os.path.join(directory_txt,fname)) as infile:
                    for line in infile:
                        outfile.write(line)




