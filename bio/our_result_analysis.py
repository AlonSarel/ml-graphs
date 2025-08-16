import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import os
import seaborn as sns
import copy
from pathlib import Path
from typing import Union, Dict, List, Tuple
import argparse

# Goals: 
# 1. Discover all of the directories in a given directory ('outs/60_emb/5_head' / 'outs/60_emb/3_head')
# 2. Now if all the files in all of the directories we are in seeds mode
# 3. Now for each one of the dirs build a result name for the dir path ignoring the final pkl name (ASSERT each dir has one pkl)
# 4. If we are not in seeds mode create a fective seed with a single result
# 5. Now get the result for each seed_result

FIGURE_FILE_TYPE = 'png'
FIGURE_LABEL_NAMES = True


def most_stable_index(arr, window_size, max_stuff, best_range=0.001):
    arr = np.array(arr)
    n = len(arr)
    if window_size > n:
        raise ValueError("Window size must be <= length of array")
    
    # Compute variances for each sliding window
    means = np.array([
        np.mean(arr[i:i+window_size])  # population variance
        for i in range(n - window_size + 1)
    ])

    in_range_list = ((max_stuff * (1-best_range)) < means) & (means < (max_stuff * (1+best_range)))
    return np.argmax(in_range_list)
    # # Find the index of the smallest variance
    # min_var_idx = np.argmax(variances)
    # return min_var_idx, variances[min_var_idx]

def our_analysis(target_path: Union[Path, str],
                 dest_dir="figures",
                 compare_all=True,
                 comparsion_list=[],
                 translation_table={},
                 map_all=False,
                 allow_mix=False,
                 use_special_idx_select=False):
    if type(target_path) == str:
        target_path = Path(target_path)

    assert target_path.exists(), 'Target file ("{}") doesn\'t exists'.format(target_path.absolute().as_posix())
    assert target_path.is_dir(), 'Target file ("{}") is not a dir'.format(target_path.absolute().as_posix())

    print('Looking for results')
    run_mods: List[Path] = list(target_path.glob('*'))
    run_mods_with_seeds = dict()
    is_seed_mode_set = set()
    for run_mod in run_mods:
        run_mod_name = run_mod.as_posix().replace('/', '_')
        run_mods_with_seeds[run_mod_name] = list()
        all_dirs = True
        for file in run_mod.glob('*'):
            if not file.is_dir():
                all_dirs = False
                assert len(list(run_mod.rglob('*.pkl'))) == 1, 'Each non seed run mod must contain exactly one pkl'
                run_mods_with_seeds[run_mod_name].append(run_mod)
                break
            assert len(list(file.rglob('*.pkl'))) == 1, 'Each seed dir in the run mod must contain exactly one pkl'
            run_mods_with_seeds[run_mod_name].append(file)
        
        is_seed_mode_set.add(all_dirs)
    if len(is_seed_mode_set) == 1:
        is_seed_mode = is_seed_mode_set.pop() # This should'nt have any effect on the code
    else:
        assert allow_mix, 'All run mods must either contain only dirs (aka seeds) or contain a pkl in them, please use allow mixed'
        is_seed_mode = True

    print('Currently running in seed mode' if is_seed_mode else 'Currently not running in seed mode')
    print('Current run modes:')
    for run_mod_name in run_mods_with_seeds.keys():
        print('\t- {}'.format(run_mod_name))

    # Our result dict will be run_mode: seed: results
    best_result_dict: Dict[str, Dict[str, Dict[str, int]]] = dict()
    mean_result_dict: Dict[str, Dict[str, float]] = dict()
    median_result_dict: Dict[str, Dict[str, float]] = dict()
    std_result_dict: Dict[str, Dict[str, float]] = dict()

    print('Loading results and calculating means')
    for run_mod_name, seeds in run_mods_with_seeds.items():
        best_result_dict[run_mod_name] = dict()
        test_easy_list = []
        test_hard_list = []
        for seed in seeds:
            seed_name = seed.as_posix()
            best_result_dict[run_mod_name][seed_name] = dict()
            pkl_file = list(seed.rglob('*.pkl'))[0] # This is validated above
            with open(pkl_file, "rb") as f:
                result = pickle.load(f)
            
            best_result_dict[run_mod_name][seed_name] = dict()
            val_ave = np.average(result['val'], axis = 1)
            best_epoch = np.argmax(val_ave)
            print(f"best_epoch {run_mod_name} {seed_name}: {best_epoch}")

            if use_special_idx_select:
                WINDOW_SIZE = 8
                idx = most_stable_index(val_ave, WINDOW_SIZE, val_ave[best_epoch])
                best_epoch = idx + WINDOW_SIZE // 2
                print(f"new best_epoch {run_mod_name} {seed_name}: {best_epoch}")

            test_easy_best = result["test_easy"][best_epoch]            
            test_hard_best = result["test_hard"][best_epoch]
            best_result_dict[run_mod_name][seed_name]['test_easy'] = test_easy_best
            best_result_dict[run_mod_name][seed_name]['test_hard'] = test_hard_best

            test_easy_list.append(test_easy_best)
            test_hard_list.append(test_hard_best)
        
        mean_result_dict[run_mod_name] = dict()
        mean_result_dict[run_mod_name]['test_easy'] = np.array(test_easy_list).mean(axis=1).mean()
        mean_result_dict[run_mod_name]['test_hard'] = np.array(test_hard_list).mean(axis=1).mean()

        median_result_dict[run_mod_name] = dict()
        median_result_dict[run_mod_name]['test_easy'] = np.median(np.array(test_easy_list).mean(axis=1))
        median_result_dict[run_mod_name]['test_hard'] = np.median(np.array(test_hard_list).mean(axis=1))
        std_result_dict[run_mod_name] = dict()
        std_result_dict[run_mod_name]['test_easy'] = np.array(test_easy_list).mean(axis=1).std()
        std_result_dict[run_mod_name]['test_hard'] = np.array(test_hard_list).mean(axis=1).std()

    print('Averages (mean ± std):')
    sorted_test_hard = sorted(mean_result_dict.items(), key=lambda kv: kv[1]['test_hard'], reverse=True)
    for k, _ in sorted_test_hard:
        print('\t- {}: {} ± {} median: {}'.format(k, mean_result_dict[k]['test_hard']*100, std_result_dict[k]['test_hard']*100, median_result_dict[k]['test_hard']*100))
        # print('\t- {} - median: {}'.format(k, median_result_dict[k]['test_hard']*100))

    os.makedirs(dest_dir, exist_ok=True)

    # plot rocs comparison graphs and save
    def plot_scatter(x_data, y_data):
        ax = sns.scatterplot(x=x_data, y=y_data)
        ax.plot([0, 1], [0, 1], 'red', linewidth=1)
        ax.set(ylim=(0, 1))
        ax.set(xlim=(0, 1))
        return ax

    def get_fig_name(method_y, method_x):
        fig_name = 'bio_pairwise_' +  method_y + '_vs_' + method_x + f'.{FIGURE_FILE_TYPE}'
        return fig_name

    def save_fig(method_y, method_x, ax):
        fig_name = get_fig_name(method_y, method_x)
        fig = ax.get_figure()
        fig.savefig(os.path.join(dest_dir, fig_name))
    
    mean_task_result_dict: Dict[str, np.array] = dict()
    for experiment, seeds in best_result_dict.items():
        test_hard_list = []
        for _, seed_result in seeds.items():
            test_hard_list.append(seed_result['test_hard'])
        mean_task_result_dict[experiment] = np.array(test_hard_list).mean(axis=0)
    
    experiment_pairs: List[Tuple[str, str]] = copy.deepcopy(comparsion_list)
    if compare_all:
        assert len(experiment_pairs) == 0, 'In compare all mode we ignore comapre requests'
        experiment_keys = list(mean_task_result_dict.keys())
        for first in range(len(experiment_keys)):
            for second in range(first + 1, len(experiment_keys)):
                experiment_pairs.append((experiment_keys[first], experiment_keys[second]))

    print('Comparing:')    
    for exp_y, exp_x in experiment_pairs:
        method_y = exp_y
        method_x = exp_x
        if 'finetune_only' in method_y or 'only_finetune' in method_y:
            method_y, method_x = method_x, method_y
        print('\t- {} vs {}'.format(method_y, method_x))
        y_data, x_data = mean_task_result_dict[method_y], mean_task_result_dict[method_x]
        assert len(y_data) == len(x_data), 'Data length mismatch'
        ax = plot_scatter(x_data, y_data)
        fig = ax.get_figure()
        if FIGURE_LABEL_NAMES:
            if map_all:
                for method in [method_x, method_y]:
                    assert method in translation_table, '"{}" doesn\'t have a mapped name'.format(method)
            method_x_label = translation_table.get(method_x, method_x)
            method_y_label = translation_table.get(method_y, method_y)
            ax.set_xlabel(method_x_label, fontsize=16)
            ax.set_ylabel(method_y_label, fontsize=16)
        fig.set_size_inches(6, 6)
        save_fig(method_y, method_x, ax)
        plt.close(fig) 

        if exp_x == '_nopretrain':
            print("\t\t- Negative transfer of " + exp_y[1:])
            print(np.sum(x_data > y_data + 0.001))

def parse_dict(s: str):
    """Parses 'key:value' into a dictionary entry."""
    parts = s.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid dict entry: {s}. Expected format 'key:value'.")
    return parts[0], parts[1]

def parse_pair(s: str):
    """Parses 'a:b' into a tuple ('a', 'b')."""
    parts = s.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid pair: {s}. Expected format 'key:value'.")
    return tuple(parts)

def main():
    parser = argparse.ArgumentParser(description='result analysis')
    parser.add_argument('--path', type=Path, required=True,
                        help='Path to folder with results')
    parser.add_argument('--dest-dir', type=str, default="figures",
                    help='Path to save the results')

    parser.add_argument('--allow-mix', action='store_true',
                        help='Allow folders to have a mix of seed mode and non seed mode')
    parser.add_argument('--use-special-index-select', action='store_true',
                    help='Use a special logic to choose the most stable idx compared to the max')

    parser.add_argument('--map-all', action='store_true',
                    help='Force it so all of the comparsion pairs would have a renamed result')
    parser.add_argument(
        "--name-mapping",
        type=parse_dict,
        nargs="+",  # multiple entries allowed
        help="Dictionary of name translations as key:value",
        required=False
    )

    parser.add_argument('--no-compare-all', action='store_false',
                help='Using this mode we would only compare some of the results',
                dest='compare_all')
    parser.add_argument(
        "--comparsion-pairs",
        type=parse_pair,
        nargs="+",  # multiple pairs allowed
        help="List of modes to compare."
    )

    args = parser.parse_args()
    name_mapping = dict(args.name_mapping) if args.name_mapping is not None else dict()
    comparsion_pairs = args.comparsion_pairs if args.comparsion_pairs else list()

    our_analysis(args.path,
                 dest_dir=args.dest_dir,
                 compare_all=args.compare_all,
                 comparsion_list=comparsion_pairs,
                 translation_table=name_mapping,
                 map_all=args.map_all,
                 allow_mix=args.allow_mix,
                 use_special_idx_select=args.use_special_index_select)
    

if __name__ == "__main__":
    main()
