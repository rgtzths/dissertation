from __future__ import print_function, division
import os
import configparser

config = configparser.RawConfigParser()
filename = os.path.expanduser('~/.neuralnilm')
config.read(filename)
