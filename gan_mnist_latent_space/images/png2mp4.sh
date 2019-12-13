#!/bin/bash

ffmpeg -r 24 -f image2 -pattern_type glob -i "*?png" -vcodec libx264 -crf 20 -pix_fmt yuv420p output.mp4
