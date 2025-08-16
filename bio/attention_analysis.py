import argparse
import numpy as np
import pathlib
import torch
from typing import List

def load_attention_and_softmax(attention_path: pathlib.Path) -> np.array:
    print('Loading attention from "{}"'.format(attention_path))
    loaded_attn = torch.load(attention_path)
    softmaxed_attn = torch.nn.functional.softmax(loaded_attn, dim=1)
    return np.array(softmaxed_attn.cpu())

def load_attentions_and_softmax_from_paths(attention_paths: List[pathlib.Path]) -> np.array:
    attentions: List[np.array] = []
    for attention_path in attention_paths:
        attentions.append(load_attention_and_softmax(attention_path))
    return np.concatenate(attentions, axis=0)

def validate_all_paths_exists(attention_paths: List[pathlib.Path]) -> None:
    for attention_path in attention_paths:
        assert attention_path.exists(), 'The attention file ("{}") doesn\'t exists'.format(attention_path.absolute().as_posix())

def main():
    parser = argparse.ArgumentParser(description='Attention analysis tools for this paper') 
    parser.add_argument('--attention-path', type=pathlib.Path,
                         required=True, help='The path to the saved attention matrix', nargs="+")
    args = parser.parse_args()
    validate_all_paths_exists(args.attention_path)

    
    attention_matrix = load_attentions_and_softmax_from_paths(args.attention_path)
    print('Resulting shape: {}'.format(attention_matrix.shape))

    # Looking at each attention per task and performing means and stuff on them.
    top2 = np.partition(attention_matrix, -2, axis=1)[:, -2:]
    average_top_2_sum_array = np.sum(top2, axis=1, keepdims=True)
    average_top_2_sum = '{} ± {}'.format(average_top_2_sum_array.mean() * 100 , average_top_2_sum_array.std() * 100)

    smallest2 = np.partition(attention_matrix, 2, axis=1)[:, :2]
    average_min_2_sum_array = np.sum(smallest2, axis=1, keepdims=True)
    average_min_2_sum = '{} ± {}'.format(average_min_2_sum_array.mean() * 100 , average_min_2_sum_array.std() * 100)
    
    top_mean_array = attention_matrix.max(axis=1)
    top_mean = '{} ± {}'.format(top_mean_array.mean() * 100 , top_mean_array.std() * 100)
    buttom_mean_array = attention_matrix.min(axis=1)
    buttom_mean = '{} ± {}'.format(buttom_mean_array.mean() * 100 , buttom_mean_array.std() * 100)
    cv = attention_matrix.std(axis=1, keepdims=True) / attention_matrix.mean(axis=1, keepdims=True)
    cv_mean = '{} ± {}'.format(cv.mean() * 100 , cv.std() * 100)

    print('Per task statistics:')
    for entry in ['average_top_2_sum', 'average_min_2_sum', 'top_mean', 'buttom_mean', 'cv_mean']:
        entry_val = locals()[entry]
        print('\t- {} = {}'.format(entry, entry_val))

    for attention_path in args.attention_path:
        print('Loading from {}'.format(attention_path.absolute().as_posix()))
        single_attention_matrix = load_attention_and_softmax(attention_path)
        print('Shape: {}', single_attention_matrix.shape)
        min_for_each_task = single_attention_matrix.min(axis=0)
        max_for_each_task = single_attention_matrix.max(axis=0)
        average_for_each_task = single_attention_matrix.mean(axis=0)
        average_for_each_task_sum = np.sum(average_for_each_task)
        cv_for_each_task = single_attention_matrix.std(axis=0) / average_for_each_task
        print('Accross task statistics:')
        for entry in ['min_for_each_task', 'max_for_each_task', 'average_for_each_task', 'average_for_each_task_sum', 'cv_for_each_task']:
            entry_val = locals()[entry]
            print('\t- {} = {}'.format(entry, entry_val))



if __name__ == "__main__":
    main()