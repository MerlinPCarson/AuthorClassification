import os
import re
import sys
import time
import argparse
from glob import glob
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Feature creator for author classification')
    parser.add_argument('--text_dir', type=str, default='txts')

    return parser.parse_args()

def find_unique_words(txt):
    regex = re.compile('[^a-zA-Z]')
    clean_words = []
    with open(txt, 'r') as f:
        text = f.read()
        words = text.split()
        for word in words:
            word = regex.sub('', word)
            # skip anything with 1 or less alpha chars
            if len(word) < 2:
                continue
            clean_words.append(word.lower())

    return list(set(clean_words))


def create_dictionary(text_dir):

    dict_words = []
    for txt in tqdm(glob(os.path.join(text_dir, '*.txt'), recursive=True)):
        words = find_unique_words(txt)
        dict_words.extend(words)

    dict_words = set(dict_words)

    dict_words = {key: value for value, key in enumerate(sorted(dict_words))}

    return dict_words

def main(args):
    start = time.time()

    # load all words from all texts in text dir
    create_dictionary(args.text_dir)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
