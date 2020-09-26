import sys
import json




if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        r = json.load(f)
