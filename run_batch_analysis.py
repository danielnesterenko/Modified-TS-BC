from run_agent import main as run_agent_main
from distance_fns import DISTANCE_FUNCTIONS
import argparse


seeds = [123]#, 456, 789, 211, 256, 356, 889, 523, 164, 177]

goals = ['stay in front of a tree and chop it']#, 'get dirt', 'travel']

def main(args):

    for goal in range(len(goals)):
        for seed in range(len(seeds)):
            args.seed = seeds[seed]
            args.goal = goals[goal]
            args.run = seed
            run_agent_main(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--goal', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--search-folder', type=str, default='weights/ts_bc/')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--max-frames', type=int, default=1*60*20) # 20 = 1 second
    parser.add_argument('--run', type=int, default='0')
    parser.add_argument('--title', type=str, default='recording')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--distance-fn', type=str, default='euclidean', choices=DISTANCE_FUNCTIONS.keys())

    args = parser.parse_args()

    main(args)