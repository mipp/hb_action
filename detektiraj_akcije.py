"""
Detekcija akcija u videu.
"""
import argparse
import datetime
import csv
import os
import sys
import traceback
import cv2
import numpy as np
from multiprocessing import Process, Queue, active_children
from pandas import read_csv

import model_paths
import build_graph_novo
from kolaz_klasifikacija import Kolaz, ActionModel, LabelMapper, get_box_size

def ds2cv(res):
    return res[1], res[2], res[3], res[4], res[0]


def sort2cv(res):
    ul_x = res[1]
    ul_y = res[2]
    width = res[3]
    height = res[4]

    lr_x = ul_x + width
    lr_y = ul_y + height
    return ul_x, ul_y, lr_x, lr_y, res[0]


def pandas_open_tracking_annotation(file_name):
    try:
        dets = read_csv(file_name, header=None, dtype=float)
    except:
        return None
    return dets.astype('int32')


def detect(df, frame_no):
    return df.loc[df[0] == frame_no].to_numpy()


def get_crop(image, orig_box, w, h):
    """izrezuje iz image kutiju dimenzija wxh na lokaciji orig_box"""
    cx = (orig_box[0] + orig_box[2]) / 2
    cy = (orig_box[1] + orig_box[3]) / 2
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    try:
        crop = image[y1 : y1 + h, x1 : x1 + w]
    except Exception:
        traceback.print_exc()
        crop = None
    return crop


def blit(dest, src, loc):
    dest[loc[0] : loc[0] + src.shape[0], loc[1] : loc[1] + src.shape[1], :] = src
    return dest


class Dummy:
    pass


def kolaz_task(args, cap, actionModel, labelMapper, detections_pd, window_length, start_frame):
    vid_fname = args.video 
    first_frame = start_frame
    last_frame = first_frame + window_length - 1

    window_of_interest = detections_pd.loc[(detections_pd[0]>=first_frame) & (detections_pd[0]<=last_frame)]

    detections = window_of_interest.to_numpy()
    player_ids = np.unique(window_of_interest[1])

    kolazi = {}
    for player_id in player_ids:
        current_detections = window_of_interest.loc[window_of_interest[1]==player_id]
        kolazi[player_id] = (Kolaz(current_detections.to_numpy(), -1, cap.get(cv2.CAP_PROP_FPS), everynth = 9, num_frames=window_length)) 
        kolazi[player_id].player_id = player_id
        kolazi[player_id].export_fname = os.path.join(
        args.output_dir,
        os.path.basename(vid_fname).split(".")[0],
        str(player_id) + ".mp4",
    )
    
    frame_cache = [] 
    frame_no = first_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    while cap.isOpened() and frame_no <= last_frame:
        ret, im = cap.read()
        frame_cache.append(im)
        boxes = detect(window_of_interest, frame_no)
        # frame za snimanje u kolaž:
        for box in boxes:
            if args.tracker_ds:
                temp_box = ds2cv(box[1:6])
            else:
                temp_box = sort2cv(box[1:6])
            player_id = box[1]
            export_frame = get_crop(im, temp_box, kolazi[player_id].w, kolazi[player_id].h)
            kolazi[player_id].setNextFrame(export_frame)
        
        for player_id in player_ids:
            kolazi[player_id]._cur_frame += 1

        frame_no += 1

    scores, predictions, player_ids = actionModel.eval_kolazi(kolazi.values())
    return scores, predictions, frame_cache, player_ids


# Prođe sve i snimi output u video i npy
def main_export(args):
    detections_fname = args.input

    # Dugotrajne varijable
    actionModel = ActionModel(beta=0.25, model_name=args.model)
    labelMapper = LabelMapper(_LABEL_MAP_PATH)
    cap = cv2.VideoCapture(args.video)
    
    detections_pd = pandas_open_tracking_annotation(
        os.path.join(args.input, detections_fname)
    )

    # Sam detection task

    # Ako je model učen na cijelim izrezanim akcijama ("time stretch"), window je prosječno trajanje akcija,
    # ako je učen s fiksnim brojem frameova u sredini, koristi se broj frameova iz modela.
    # U svakom slučaju se iz videa izrezuje WINDOW_LEN frameova, a onda kasnije funkcija eval_kolazi iz kolaža
    # izreže potreban broj frameova.
    # (Bez obzira koliko time stretch model uzima u obzir frameova, mi mu proslijedimo koliko ih je prosječno u jednoj akciji.)
    # TODO: Dodati mogućnost preklapanja izrezanih prozora (parametar skip...)
    if model_dict['TIME_STRETCH']:
        WINDOW_LEN = 40
    else:
        WINDOW_LEN = model_dict['FRAMES']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    filename = os.path.join(args.output_dir, os.path.basename(args.video))
    txtfname = os.path.join(args.output_dir, os.path.basename(args.video)+'.csv')
    csv_fhandle = open(txtfname, 'w')
    csv_writer = csv.writer(csv_fhandle, delimiter=';')
    out_fhandle = cv2.VideoWriter(filename, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    allstart = datetime.datetime.now()
    for start_frame in range(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-WINDOW_LEN, WINDOW_LEN):
        tic = datetime.datetime.now()
        scores, predictions, frame_cache, player_ids = kolaz_task(args, cap, actionModel, labelMapper, detections_pd, WINDOW_LEN, start_frame)
        toc = datetime.datetime.now()
        elapsed = toc-tic
        print(elapsed.total_seconds())
        csv_writer.writerow((start_frame, player_ids, scores, predictions))
        append_export(args, detections_pd, labelMapper, predictions, player_ids, start_frame, frame_cache, out_fhandle)
    out_fhandle.release()
    csv_fhandle.close()
    print("Total time: ")
    totaltime = datetime.datetime.now() - allstart
    print(totaltime.total_seconds())
    
def append_export(args, detections_pd, labelMapper, predictions, player_ids, start_frame, frame_cache, fhandle):
    
    frame_no = start_frame
    last_frame = start_frame + 39
    
    for im in frame_cache:
        boxes = detect(detections_pd, frame_no)
        # frame za snimanje u kolaž:
        for n,box in enumerate(boxes):
            if args.tracker_ds:
                temp_box = ds2cv(box[1:6])
            else:
                temp_box = sort2cv(box[1:6])
            player_id = box[1]
            result_idx = player_ids.index(player_id)
            cv2.rectangle(im, temp_box[0:2], temp_box[2:4], (255,255,255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = labelMapper.reverse_map[predictions[result_idx][0]]
            if label != 'background':
                cv2.putText(im, label, temp_box[0:2], font, 0.8, (0, 200, 0), 2, cv2.LINE_AA)
        fhandle.write(im)
        frame_no += 1


def replay(args, detections_pd, labelMapper, predictions, player_ids, start_frame, frame_cache):
    
    frame_no = start_frame
    last_frame = start_frame + 39
    
    for im in frame_cache:
        boxes = detect(detections_pd, frame_no)
        # frame za snimanje u kolaž:
        for n,box in enumerate(boxes):
            if args.tracker_ds:
                temp_box = ds2cv(box[1:6])
            else:
                temp_box = sort2cv(box[1:6])
            player_id = box[1]
            result_idx = player_ids.index(player_id)
            cv2.rectangle(im, temp_box[0:2], temp_box[2:4], (255,255,255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = labelMapper.reverse_map[predictions[result_idx][0]]
            if label != 'background':
                cv2.putText(im, label, temp_box[0:2], font, 0.8, (0, 200, 0), 2, cv2.LINE_AA)
        cv2.imshow("Preview", im)
        cv2.waitKey(10)
        frame_no += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="Tracks file (result of e.g. deep sort.)")
    ap.add_argument("--tracker_ds", dest="tracker_ds", default=True, action="store_false", help="Tracks file format, True - Deep SORT (default), False - SORT")
    ap.add_argument("--video", help="Input video")
    ap.add_argument("--model", dest="model", default=None, help="Model weights file.")
    ap.add_argument("--output_dir", help="Output directory")
    args = ap.parse_args(sys.argv[1:])
    
    print("Available models:")
    model_paths.decribe_models()

    if args.model is None:
        print("--model keyword argument not set. Using default model (flip_2708)")
        print("\n")
        args.model = "flip_2708"
    
    model_dict = model_paths.model_paths[args.model]
    _LABEL_MAP_PATH = model_dict["_LABEL_MAP_PATH"]
    
    print(model_dict["info"], "\n")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    main_export(args)
    
