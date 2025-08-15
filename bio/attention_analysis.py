import argparse
import numpy as np
import pathlib
import torch

def load_attention_and_softmax(attention_path: pathlib.Path) -> np.array:
    loaded_attn = torch.load(attention_path)
    softmaxed_attn = torch.nn.functional.softmax(loaded_attn, dim=1)
    return np.array(softmaxed_attn.cpu())

def main():
    parser = argparse.ArgumentParser(description='Attention analysis tools for this paper') 
    parser.add_argument('--attention-path', type=pathlib.Path,
                         required=True, help='The path to the saved attention matrix')
    args = parser.parse_args()
    attention_path: pathlib.Path = args.attention_path
    assert attention_path.exists(), 'The attention file ("{}") doesn\'t exists'.format(attention_path.absolute().as_posix())

    print('Loading attention from "{}"'.format(attention_path))
    attention_matrix = load_attention_and_softmax(attention_path)
    print('Resulting shape: {}'.format(attention_matrix.shape))

    # Looking at each attention per task and performing means and stuff on them.
    top2 = np.partition(attention_matrix, -2, axis=1)[:, -2:]
    average_top_2_sum = np.sum(top2, axis=1, keepdims=True).mean()
    smallest2 = np.partition(attention_matrix, 2, axis=1)[:, :2]
    average_min_2_sum = np.sum(smallest2, axis=1, keepdims=True).mean()
    top_mean, buttom_mean = attention_matrix.max(axis=1).mean(), attention_matrix.min(axis=1).mean()
    cv = attention_matrix.std(axis=1, keepdims=True) / attention_matrix.mean(axis=1, keepdims=True)
    cv_mean = cv.mean()

    print('Per task statistics:')
    for entry in ['average_top_2_sum', 'average_min_2_sum', 'top_mean', 'buttom_mean', 'cv_mean']:
        entry_val = locals()[entry]
        print('\t- {} = {}'.format(entry, entry_val))

    min_for_each_task = attention_matrix.min(axis=0)
    max_for_each_task = attention_matrix.max(axis=0)
    average_for_each_task = attention_matrix.mean(axis=0)
    average_for_each_task_sum = np.sum(average_for_each_task)
    cv_for_each_task = attention_matrix.std(axis=0) / average_for_each_task
    print('Accross task statistics:')
    for entry in ['min_for_each_task', 'max_for_each_task', 'average_for_each_task', 'average_for_each_task_sum', 'cv_for_each_task']:
        entry_val = locals()[entry]
        print('\t- {} = {}'.format(entry, entry_val))



if __name__ == "__main__":
    main()