import os
import glob
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='counting_outputs')
    parser.add_argument('--output', type=str, default='submission_output')
    args = parser.parse_args()
    
    count_outputs = sorted(glob.glob(os.path.join(args.root, 'cam*txt')))
    assert len(count_outputs) == 25
    
    output = os.path.join(args.output, 'submission.txt')
    fp  = open(output, 'w')
    for count in count_outputs:
        with open(count, 'r') as f:
            for line in f:
                line = line.strip()
                if line != "":
                    fp.write(line + "\n")
    fp.close()        
    

    