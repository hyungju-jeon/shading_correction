import sys


if sys.version_info[0] < 3 or sys.version_info[1] < 8:
    sys.stderr.write('Error: need python 3.8 or higher\n')
    sys.exit(1)
