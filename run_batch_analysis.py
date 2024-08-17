from run_agent_patches import main as run_agent_main
#from run_agent import main as run_agent_main
from analyse import main as run_analysis_main
import matplotlib.pyplot as plt
from distance_fns import DISTANCE_FUNCTIONS
import argparse
import os
import sys

#seeds = [1231, 4561, 7891, 2111, 2561, 3561, 8891, 5231, 1641, 1771]
seeds = [123, 456, 789, 211, 256, 356, 889, 523, 164, 177]

goals = ['stay in front of a tree and chop it']#, 'chop a tree', 'get dirt', 'go swimming']

def main(args):

    for goal in range(len(goals)):
        for seed in range(len(seeds)):
            args.seed = seeds[seed]
            args.goal = goals[goal]
            args.run = seed
            run_agent_main(args)
            #run_analysis_main(args)
        
        #plot_results('output/programmatic_results')
        #exit()


def plot_results(directory):
    collected_items_dict = parse_txt_files(directory)
    for run in range(len(flatten_dict(collected_items_dict))):
        plot_dict_barplot(collected_items_dict)


def parse_txt_files(directory):
    data_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data = {}
                for line in file:
                    key, value = line.strip().split(': ')
                    data[key] = float(value)
                data_dict[filename] = data
    return data_dict


def plot_dict_barplot(data_dict):
    fig, axs = plt.subplots(len(data_dict), figsize=(8, 6*len(data_dict)))
    
    for i, (filename, data) in enumerate(data_dict.items()):
        axs[i].bar(data.keys(), data.values(), color='skyblue')
        axs[i].set_xlabel('Keys')
        axs[i].set_ylabel('Values')
        axs[i].set_title(f'Bar plot for {filename}')
        axs[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('output/programmatic_results/batch_run_results.png')
    plt.show()

def flatten_dict(dictionary):
    items = []
    for key, value in dictionary.items():
        if isinstance(value, dict):
            items.extend(flatten_dict(value))
        else:
            items.append(key)
    return list(set(items))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--goal', type=str, default='new goal')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--search-folder', type=str, default='weights/ts_bc/')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--max-frames', type=int, default=1*60*20)
    parser.add_argument('--run', type=int, default='0')

    parser.add_argument('--title', type=str, default='recording')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--description', type=str, default='')
    parser.add_argument('--distance-fn', type=str, default='euclidean', choices=DISTANCE_FUNCTIONS.keys())

    args = parser.parse_args()

    main(args)