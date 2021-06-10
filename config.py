import sys
import getopt

arguments, _values = getopt.getopt(sys.argv[1:], 'c:')

should_replay_checkpoint = False
checkpoint_path = ''

for currentArgument, currentValue in arguments:
  if currentArgument == '-c':
    should_replay_checkpoint = True
    checkpoint_path = currentValue
