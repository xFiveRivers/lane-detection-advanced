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


from docopt import docopt
from method import *


def main():
    args = docopt(__doc__)

    lanedetection = LaneDetection()

    if args['<media>'] == 'image':
        lanedetection.process_image(args['<input_path>'], args['<output_path>'])
    elif args['<media>'] == 'video':
        lanedetection.process_video(args['<input_path>'], args['<output_path>'])
    else:
        print('Please specify either image or video for your media choice.')


if __name__ == '__main__':
    main()