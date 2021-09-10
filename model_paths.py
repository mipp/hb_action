def decribe_models():
    for m in model_paths.keys():
        print(m + ":", model_paths[m]["info"])

def decribe_models_verbose():
    for m in model_paths.keys():
        print(m + ":", model_paths[m]["_VIDEO_DIR"], model_paths[m]["_LABEL_MAP_PATH"],model_paths[m]["NUM_CLASSES"],model_paths[m]["IMAGE_SIZE"],model_paths[m]["FRAMES"],model_paths[m]["CROP_FLOW"], model_paths[m]["TIME_STRETCH"])

def get_ensemble_paths(model_name):
    mdict = model_paths[model_name]
    classes = [x.strip() for x in open(mdict["_LABEL_MAP_PATH"])]
    ensemble_dict = dict()
    for i, c in enumerate(classes):
        if c=="background":
            continue
        ensemble_dict[c+"_"+model_name] = {
        "info": mdict["info"] + ", klasa " + c,
        "_VIDEO_DIR": mdict["_VIDEO_DIR"],
        "_LABEL_MAP_PATH": mdict["_LABEL_MAP_PATH"],
        "_CHECKPOINT_PATHS": {
            'rgb_imagenet': mdict["_CHECKPOINT_PATHS"]['rgb_imagenet'],
            'flow_imagenet': mdict["_CHECKPOINT_PATHS"]['flow_imagenet'],
            'training': mdict["_CHECKPOINT_PATHS"]['training']+c+'/model.ckpt',
            'rgb_novo': mdict["_CHECKPOINT_PATHS"]['rgb_novo']+c+'_rgb/model.ckpt',
            'flow_novo': mdict["_CHECKPOINT_PATHS"]['flow_novo']+c+'_flow/model.ckpt',
        },
        "_STATS": {
            'train_acc': 'data/stats/train_acc_' + model_name + '_' + c + '_0409.npy',
            'val_acc': 'data/stats/val_acc_' + model_name + '_' + c + '_0409.npy',
            'loss': 'data/stats/loss_' + model_name + '_' + c + '_0409.npy'
        },
        "NUM_CLASSES": 2,
        "IMAGE_SIZE": mdict["IMAGE_SIZE"],
        "FRAMES": mdict["FRAMES"][i],
        "BATCH_SIZE": mdict["BATCH_SIZE"],
        "CROP_FLOW": mdict["CROP_FLOW"],
        "TIME_STRETCH": mdict["TIME_STRETCH"]
        }
    return ensemble_dict

model_paths = {
    "flip_2708": {
        "info": "Model s 10 klasa + background, kod učenja korišten random flip.",
        "_VIDEO_DIR": "videos_mateo_kristian_sveklase_background",
        "_LABEL_MAP_PATH": "data/exercises_label_map_rukomet_extra_background.txt",
        "_CHECKPOINT_PATHS": {
            'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
            'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
            'training': 'data/checkpoints/mk_background_flip_2708/model.ckpt',
            'rgb_novo': 'data/checkpoints/rgb_mk_background_flip_2708/model.ckpt',
            'flow_novo': 'data/checkpoints/flow_mk_background_flip_2708/model.ckpt'
        },
        "_STATS": {
            'train_acc': 'data/stats/train_acc_mk_background_flip_2708.npy',
            'val_acc': 'data/stats/val_acc_mk_background_flip_2708.npy',
            'loss': 'data/stats/loss_mk_background_flip_2708.npy'
        },
        "NUM_CLASSES": 11,
        "IMAGE_SIZE": 224,
        "FRAMES": 40,
        "BATCH_SIZE": 10,
        "CROP_FLOW": True,
        "TIME_STRETCH": True
    },
}