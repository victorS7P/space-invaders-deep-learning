import sys
import getopt

arguments, _ = getopt.getopt(sys.argv[1:], 'c:m:r')

should_render = False
should_replay_checkpoint = False
should_replay_model = False
path = ''


for currentArgument, currentValue in arguments:
  if currentArgument == '-c':
    should_replay_checkpoint = True
    path = currentValue

  if currentArgument == '-m':
    should_replay_model = True
    path = currentValue

  if currentArgument == '-r':
    should_render = True
