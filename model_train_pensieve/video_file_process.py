import os

VIDEO_FILE_PATH = '../video_info/'
BITRATE_LEVELS = 5
TOTAL_CHUNK_NUM = 120

for bitrate in xrange(BITRATE_LEVELS):
    with open('video_size_' + str(bitrate), 'wb') as f:
        with open(VIDEO_FILE_PATH+'bitrate_level_'+str(bitrate)) as g:
            for line in g:
                parse = line.split()
                chunk_size = parse[1]
                f.write(str(chunk_size) + '\n')
