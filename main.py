"""
Lane Detection Pipeline

Usage:
    main.py <media> <input_path> <output_path>

Options:
    -h --help           Show help screen.
    <media>             Flag to process image or video.
    <input_path>        Path from root to input media.
    <output_path>       Path from root to output media.
"""


import docopt
opt = docopt(__doc__)
             
from method import *


def main(media, input_path, output_path):
    lanedetection = LaneDetection()

    if media == 'image':
        lanedetection.process_image(input_path, output_path)
    elif media == 'video':
        lanedetection.process_video(input_path, output_path)
    else:
        print('Please specify either image or video for your media choice.')


if __name__ == '__main__':
    main(opt['<media>'], opt['<input_path>'], opt['<output_path>'])