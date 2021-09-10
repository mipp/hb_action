"""
Prikaz kolaža i detektiranih klasa izrezanih akcija 
Input: video i folder s .txt (csv) file s rezutlatima u MOT formatu.
    format: frame_id, track_id, x1, y1, x2, y2, confidence, ~, activity, klasa
    klase: 0 - neaktivan, 1,2,3: aktivni igrači
"""
import argparse
import os
import sys
import traceback
from multiprocessing import Process, Queue, active_children

import cv2
import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

from pandas import read_csv

import build_graph_novo
from process_video import VideoClip
import model_paths

SCREEN_X = 1920
SCREEN_Y = 1080


class ActionModel:
    def __init__(self, beta, model_name):
        super().__init__()
        self.model_dict = model_paths.model_paths[model_name]
        self.checkpoint_paths = self.model_dict["_CHECKPOINT_PATHS"]
        self.image_size = self.model_dict["IMAGE_SIZE"]
        self.num_frames = self.model_dict["FRAMES"]
        self.crop_flow = self.model_dict["CROP_FLOW"]
        try:
            self.time_stretch = self.model_dict["TIME_STRETCH"]
        except:
            self.time_stretch = True
        print("flow crop ", self.crop_flow)
        self.batch_size = 1
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        tf.compat.v1.reset_default_graph()

        inputs, outputs, savers, tf_summaries = build_graph_novo.build_graph(beta=beta)
        self.learning_rate, self.rgb_input, self.flow_input, self.is_training, self.y = inputs
        self.scores, self.loss, self.loss_minimize = outputs
        self.rgb_saver, self.flow_saver, self.training_saver = savers
        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.rgb_saver.restore(self.sess, self.checkpoint_paths['rgb_novo'])
        tf.compat.v1.logging.info('RGB checkpoint restored')
        self.flow_saver.restore(self.sess, self.checkpoint_paths['flow_novo'])
        tf.compat.v1.logging.info('Flow checkpoint restored')
        try:
            self.training_saver.restore(self.sess, self.checkpoint_paths['training'])
            tf.compat.v1.logging.info('Training checkpoint restored')
        except Exception as e:
            traceback.print_exc(e)


    def eval_kolazi(self, kolazi):
        def _prep_feed_dict(kolazi, q):
            rgb_input = []
            flow_input = []
            for k in kolazi:
                clip = VideoClip(
                    k,
                    self.image_size,
                    nframes=self.num_frames,
                    flip=False,
                    flow_crop=self.crop_flow,
                    time_stretch=self.time_stretch,
                    randomize_timecrop=False
                )
                rgb_data = clip.rgb_data_uncropped()
                rgb_input.append(rgb_data[0])
                flow_input.append(clip.flow_data()[0])
                if len(rgb_input) == self.batch_size:
                    feed_dict_ = {
                        'rgb_input': np.array(rgb_input),
                        'flow_input': np.array(flow_input)
                    }
                    q.put([feed_dict_, k.act_idx, k.player_id])
                    rgb_input = []
                    flow_input = []

        
        q = Queue(maxsize=2)
        prep_process = Process(target=_prep_feed_dict, args=(kolazi, q), daemon=True)
        prep_process.start()

        i = 0
        scores = []
        predictions = []
        player_ids = []
        while i < len(kolazi):
            feed_dict_, act_idx, player_id = q.get()
            feed_dict = {
                self.rgb_input: feed_dict_['rgb_input'],
                self.flow_input: feed_dict_['flow_input'],
                self.is_training:0
            }
            scores_np = self.sess.run(self.scores, feed_dict=feed_dict)
            y_pred = scores_np.argmax(axis=1)
            scores.append(scores_np)
            predictions.append(y_pred)
            player_ids.append(player_id)
            #print(act_idx)
            #print(scores_np), y_pred
            print(".", end='')
            i += 1
        prep_process.join()
        return scores, predictions, player_ids
####

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


def get_box_size(detections, act_idx=0):
    """Za izvoz aktivnog igrača u video/kolaž dimenzije kutije moraju biti iste za sve frameove.
    Funkcija određuje potrebne dimenzije na osnovu stvarnih dimenzija u svakom frameu.
    """
    w = 0
    h = 0
    for d in detections:
        if (
            d[-1] == act_idx
        ):  # ako je aktivan igrač zadnji stupac bi trebao biti 1, 2 je drugi najaktivniji itd.
            box_w = d[4] - d[2]
            box_h = d[5] - d[3]
            w = max(w, box_w)
            h = max(h, box_h)
    return w, h


def get_crop(image, orig_box, w, h):
    """izrezuje iz image kutiju dimenzija wxh na lokaciji orig_box"""
    # orig_box = ds2cv(orig_box) # NOTE:što će biti input...
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

def gamma_corr(image, gamma=.67):
    lut = np.linspace(0, 1, 256) ** gamma * 255
    return cv2.LUT(image, lut.astype(np.uint8))


class Kolaz:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    def __init__(self, detections, act_idx, fps, everynth=None, num_frames=None):
        """
        
        num_frames: fiksni broj frameova kad se kolaž koristi za klasifikaciju u dugačkom videu (fiksni window)
        everynth: svaki koji frame se uzima za preview
        """
        self.player_id = None
        self.w, self.h = get_box_size(detections, act_idx)
        self.act_idx = act_idx
        frame_no = detections[0][0]
        last_frame = detections[-1][0]
        if num_frames is not None:
            self.num_frames = num_frames
        else:
            self.num_frames = last_frame - frame_no + 1
        if everynth is not None:
            self.everynth = everynth
        else:
            self.everynth = np.ceil((last_frame - frame_no - 1) * self.w / SCREEN_X)

        self.image = np.ones(
            (self.h, int(np.ceil((self.num_frames / self.everynth))) * self.w, 3),
            dtype=np.uint8,
        )
        self.video_buffer = np.zeros(
            (self.h, self.w, 3, self.num_frames), dtype=np.uint8
        )
        self.valid_frames = np.zeros(self.num_frames, dtype=np.bool)
        self._cur_frame = 0
        self.FPS = fps
        

    def setNextFrame(self, frame):
        try:
            self.video_buffer[:, :, :, self._cur_frame] = frame
            self.valid_frames[self._cur_frame] = True
        except:
            self.valid_frames[self._cur_frame] = False
        if self._cur_frame % self.everynth == 0:
            #cv2.putText(
            #    frame,
            #    str(self._cur_frame),
            #    (10, 10),
            #    cv2.FONT_HERSHEY_PLAIN,
            #    0.7,
            #    (255, 255, 255),
            #)
            self.image = blit(
                self.image, frame, (0, int(self._cur_frame / self.everynth * self.w))
            )
        # self._cur_frame += 1

    def export(self):
        os.makedirs(os.path.dirname(self.export_fname), exist_ok=True)
        outv = cv2.VideoWriter(
            self.export_fname, Kolaz.fourcc, self.FPS, (self.w, self.h)
        )
        for i in range(self.video_buffer.shape[-1]):
            if self.valid_frames[i]:
                outv.write(self.video_buffer[:, :, :, i])
        outv.release()

    def play(self):
        for i in range(self.video_buffer.shape[-1]):
            if self.valid_frames[i]:
                cv2.imshow("image", self.video_buffer[:, :, :, i])
                cv2.waitKey(20)

    def scaled_preview(self, width, height):
        y_scale = height/self.h
        new_frame_width = y_scale * self.w
        pick_frames = int(width//new_frame_width)
        
        valid_frames_count = np.sum(self.valid_frames)
        if pick_frames > valid_frames_count:
            pick_frames = valid_frames_count

        buf = np.zeros((self.h, self.w * pick_frames, 3), dtype=np.uint8)
        for count, frame_idx in enumerate(np.linspace(0, valid_frames_count, pick_frames, endpoint=False, dtype=np.int)):
            idx = np.where(self.valid_frames)[0][frame_idx]
            frame = self.video_buffer[:, :, :, idx]
            buf = blit(
                buf, frame, (0, int(count * self.w))
            )
        try:
            return cv2.resize(buf, (0, 0), fy=y_scale, fx=y_scale)
        except:
            return np.zeros((height, width, 3), dtype=np.uint8)


def merge_display(kolazi, predictions, scores, labelMapper):
    WIN_X = int(0.9 * SCREEN_X)
    WIN_Y = int(0.9 * SCREEN_Y)
    new_height = WIN_Y // len(kolazi)
    image = np.zeros((WIN_Y, WIN_X, 3), dtype=np.uint8)
    for i, kolaz in enumerate(kolazi):
        kolaz_img = kolaz.scaled_preview(WIN_X, new_height)
        ##
        cur_scores = scores[i][0]
        BR_TOP_KLASA = -NUM_CLASSES if NUM_CLASSES < 5 else -4
        ind = np.argpartition(cur_scores, BR_TOP_KLASA)[BR_TOP_KLASA:]
        ind = ind[np.argsort(cur_scores[ind])]
        legend_string = "; ".join([labelMapper.reverse_map[i] + " " + str(cur_scores[i]) for i in np.flip(ind)])
        ##

        image = blit(
            image,
            kolaz_img,
            (i * new_height, 0)
        )
        cv2.putText(
            image,
            legend_string,
            (20, int((i + 0.95) * new_height)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 200),
        )
    return image


def merge_display_targetclass(kolazi, predictions, sc, labelMapper): # prikaže confidence samo za target class
    WIN_X = int(0.9 * SCREEN_X)
    WIN_Y = int(0.9 * SCREEN_Y)
    new_height = WIN_Y // len(kolazi)
    image = np.zeros((WIN_Y, WIN_X, 3), dtype=np.uint8)
    order = np.argsort(sc)
    
    for i, kolaz in enumerate(kolazi):
        kolaz_img = kolaz.scaled_preview(WIN_X, new_height)
        legend_string = str(sc[i])
        print(i, order[i])
        image = blit(
            image,
            kolaz_img,
            (order[i] * new_height, 0)
        )
        cv2.putText(
            image,
            legend_string,
            (20, int((order[i] + 0.95) * new_height)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 200),
        )
    return image


class Dummy:
    pass


class LabelMapper: 
    def __init__(self, map_path):
        super().__init__()
        self.label_map = {}
        with open(map_path, 'r') as lmf:
            for ln, line in enumerate(lmf):
                #self.label_map[line[0:3]] = ln # NOTE:samo prva tri slova zbog filenameova
                self.label_map[line.strip()] = ln # NOTE:samo prva tri slova zbog filenameova
        self.reverse_map = dict(zip(self.label_map.values(), self.label_map.keys()))
    
    def get_idx(self, fname):
        for lab in self.label_map.keys():
            if fname.startswith(lab): # Komplikacija jer filenameovi nemaju underscore :(
                return self.label_map[lab]
        if fname.startswith('skok'):
            return self.label_map['skok-sut']
        print("Missing model for class of file {}".format(fname))
        return 0


def kolaz_task(args, detection_fnames, q):
    vid_fname = args.video
    cap = cv2.VideoCapture(vid_fname)
    
    actionModel = ActionModel(beta=0.25, model_name=args.model)
    labelMapper = LabelMapper(_LABEL_MAP_PATH)

    for detection_filename in detection_fnames:
        detections_pd = pandas_open_tracking_annotation(
            os.path.join(args.input, detection_filename)
        )
        if detections_pd is None:
            q.put((None, [],[],[], detection_filename))
            continue
        detections = detections_pd.to_numpy()
        BR_KOLAZA = max(detections[:,-1])
        kolazi = []
        for i in range(BR_KOLAZA):
            frame_no = detections[0][0]
            last_frame = detections[-1][0]
            kolazi.append(Kolaz(detections, i, cap.get(cv2.CAP_PROP_FPS), everynth=5)) # FIXME: dinamicki izracun everynth
            kolazi[i].export_fname = os.path.join(
            args.output_dir,
            os.path.basename(vid_fname).split(".")[0],
            detection_filename.split(".")[0] + ".mp4",
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        while cap.isOpened() and frame_no <= last_frame:
            ret, im = cap.read()
            boxes = detect(detections_pd, frame_no)
            # frame za snimanje u kolaž:
            for box in boxes:
                if args.tracker_ds:
                    temp_box = ds2cv(box[1:6])
                else:
                    temp_box = sort2cv(box[1:6])
                for i in range(BR_KOLAZA):
                    if box[-1] == i:
                        export_frame = get_crop(im, temp_box, kolazi[i].w, kolazi[i].h)
                        kolazi[i].setNextFrame(export_frame)
            
            for i in range(BR_KOLAZA):    
                kolazi[i]._cur_frame += 1

            frame_no += 1

        target_class = labelMapper.get_idx(detection_filename)
        scores, predictions, player_ids = actionModel.eval_kolazi(kolazi)
        sc = [s[0][target_class] for s in scores]
        q.put((kolazi, sc, scores, predictions, detection_filename))
    print("Kolaz task: Done processing all actions.")


def main(args):
    detection_fnames = os.listdir(args.input)
    
    cv2.namedWindow("image")
    cv2.moveWindow("image", 0, 0)
    q = Queue(maxsize=4)
    p = Process(target=kolaz_task, args=(args, detection_fnames, q))
    p.start()
    labelMapper = LabelMapper(_LABEL_MAP_PATH)
    for detection_filename in detection_fnames:
        kolazi, sc, scores, predictions, got_detections_filename = q.get()
        assert(detection_filename == got_detections_filename)
        if kolazi is None:
            continue
        if SHOW_PREDICTED_CLASSES:
            merge_image = gamma_corr(merge_display(kolazi, predictions, scores, labelMapper))
        else:
            merge_image = gamma_corr(merge_display_targetclass(kolazi, predictions, sc, labelMapper))
        
        model_pick = np.argmax(sc)
        cv2.setWindowTitle("image", detection_filename)
        cv2.imshow("image", merge_image)
        key = cv2.waitKey()
        if key & 0xFF == 27:
            for px in active_children():
                print('Terminating', px)
                px.terminate()
            break
    p.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    SHOW_PREDICTED_CLASSES = True
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--tracker_sort", dest="tracker_ds", default=True, action="store_false")
    ap.add_argument("--video")
    ap.add_argument("--model", dest="model", default=None)
    ap.add_argument("--output_dir")
    args = ap.parse_args(sys.argv[1:])
    
    if args.model is None:
        print("--model keyword argument not set. Using default model (flip_2708)")
        print("Available models:")
        model_paths.decribe_models()
        print("\n")
        args.model = "flip_2708"
    
    model_dict = model_paths.model_paths[args.model]
    _LABEL_MAP_PATH = model_dict["_LABEL_MAP_PATH"]
    NUM_CLASSES = model_dict["NUM_CLASSES"]
    print(model_dict["info"], "\n")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    main(args)
