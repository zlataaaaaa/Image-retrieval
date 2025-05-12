import os
import cv2
import numpy as np
import json
import logging

from os import environ
from sys import argv, exit
from pathlib import Path

LEADERBOARD_SCORE = 10.

def run_single_test(data_dir, out_dir):
    from retrieval import predict_image

    try:
        files = json.load(Path(data_dir, 'files.json').open('r'))
    except:
        raise Exception("File `files.json` is missing in the test directory")

    bboxes = {}

    for idx, (img_path, query_img_path) in files.items():
        img = cv2.imread(str(Path(data_dir, img_path)))
        query_img = cv2.imread(str(Path(data_dir, query_img_path)))
        try:
            bboxes[idx] = len(predict_image(img, query_img))
        except Exception as e:
            raise Exception(f"Failed to predict bboxes in predict_image function: {e}")
    json.dump(bboxes, Path(out_dir, "bboxes.json").open('w'))


def check_test(data_dir):
    try:
        bboxes = json.load(Path(data_dir, "output", 'bboxes.json').open('r'))
    except:
        raise Exception(f"File `bboxes.json` is missing in {data_dir}")
    try:
        gt = json.load(Path(data_dir, "gt", 'gt.json').open('r'))
    except:
        raise Exception(f"File `gt.json` is missing in {Path(data_dir, 'gt')}")
    scores = []
    for idx in gt.keys():
        n_pred = bboxes[idx]
        n_true = gt[idx]
        exists_in_the_image = 0.2 * float((n_true > 0.5) == (n_pred > 0.5))
        accuracy = np.clip(0.8 - 0.2 * np.abs(n_true - n_pred), 0, 0.8)
        score = np.round(exists_in_the_image + accuracy, 2)
        scores.append(score)
    res = f'Ok, per_image_scores: {scores}'

    if environ.get('CHECKER'):
        print(res)

    return res


def grade(data_dir):
    results = json.load(Path(data_dir, 'results.json').open('r'))
    result = results[-1]['status']

    if not result.startswith('Ok'):
        res = {'description': '', 'mark': 0}
    else:
        scores = json.loads(result[22:])
        res = {'description': result[22:], 'mark': np.round(np.mean(scores) * LEADERBOARD_SCORE, 2)}

    if environ.get('CHECKER'):
        print(json.dumps(res))

    return res


if __name__ == '__main__':
    if environ.get('CHECKER'):
        mode = argv[1]
        data_dir = argv[2]
        out_dir = argv[3]

        if mode == 'run_single_test':
            run_single_test(data_dir, out_dir)
        elif mode == 'check_test':
            # Put a mark for each test result
            check_test(data_dir)
        elif mode == 'grade':
            # Put overall mark
            grade(data_dir)
    else:
        if len(argv) != 2:
            print('Usage: %s data_dir' % argv[0])
            exit(0)
        
        from pathlib import Path
        from retrieval import predict_image

        test_dirs = Path(argv[1]).glob('[0-9][0-9]_*_input')
        for test_dir in test_dirs:
            try:
                files = json.load(Path(test_dir, 'files.json').open('r'))
            except:
                raise Exception("File `files.json` is missing in the test directory")
            try:
                gt = json.load(Path(str(test_dir).replace("input", "gt"), 'gt.json').open('r'))
            except:
                raise Exception("File `gt.json` is missing in the test directory")
            
            scores = []

            for idx, (img_path, query_img_path) in files.items():
                img = cv2.imread(str(Path(test_dir, img_path)))
                query_img = cv2.imread(str(Path(test_dir, query_img_path)))
                try:
                    bboxes = predict_image(img, query_img)
                    n_true = gt[idx]
                    n_pred = len(bboxes)

                    exists_in_the_image = 0.2 * float((n_true > 0.5) == (n_pred > 0.5))
                    accuracy = np.clip(0.8 - 0.2 * np.abs(n_true - n_pred), 0, 0.8)
                    score = np.round(exists_in_the_image + accuracy, 2)
                    scores.append(score)
                except Exception as e:
                    raise Exception(f"Failed to predict bboxes in predict_image function: {e}")
            print('Mark:', np.round(np.mean(scores) * LEADERBOARD_SCORE, 2))