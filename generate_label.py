import torch
import numpy as np
import os
import argparser
import cv2
from PIL import Image
from dataset import transform as tf
from tqdm import tqdm

_transform = tf.Compose([
    tf.ToTensor(),
    tf.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

def _build_model():
    from segmentation_module import make_model, make_model_v2
    import tasks

    assert opts.net_pytorch, print("opt net_pytorch wrong!")
    model_old = make_model_v2(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1))
    path = opts.step_ckpt
    step_checkpoint = torch.load(path, map_location="cpu")
    if opts.net_pytorch:
        net_dict_old = model_old.state_dict()
        pretrained_dict = {k.replace('module.', ''): v for k, v in step_checkpoint['model_state'].items() if
                            (k.replace('module.', '') in net_dict_old)}  # and (
        # v.shape == net_dict[k.replace('module.', '')].shape)
        net_dict_old.update(pretrained_dict)
        model_old.load_state_dict(net_dict_old, strict=True)
        del net_dict_old
    else:
        model_old.load_state_dict(step_checkpoint['model_state'], strict=True)  # Load also here old parameters
    print("Previous model loaded from {path}")
    # logger.info(f"[!] Previous model loaded from {path}")
        # clean memory
    del step_checkpoint['model_state']
    model_old.eval()
    for p in model_old.parameters():
        p.requires_grad = False
    return model_old



if __name__=="__main__":
    parser = argparser.get_argparser()
    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)
    opts.data_root = r"D:\\ADAXI\\Datasets\\VOC_SDR"
    opts.replay = True
    opts.mix = True
    opts.task = '19-1'
    opts.dataset = 'voc'
    opts.step = 1
    opts.step_ckpt = r"./logs/19-1/19-1-voc_FT/19-1-voc_FT_0.pth"

    root_path = r"C:\ADAXI\Replay_Data_for_train\19-1"
    file_list = os.listdir(root_path)
    device_ = torch.device('cuda')
    model_old = _build_model()
    model_old = model_old.to(device=device_)
    for file_ in file_list:
        print(file_)
        base_path = os.path.join(root_path, file_)
        img_path = os.path.join(base_path, "image")
        lbl_path = os.path.join(base_path, "label")
        if not os.path.exists(lbl_path):
            os.makedirs(lbl_path)
        img_list = os.listdir(img_path)
        for img_name in tqdm(img_list):
            lbl_name = os.path.join(lbl_path, img_name[:-3] + "png")
            if os.path.isfile(lbl_name):
                continue
            img = Image.open(os.path.join(img_path, img_name)).convert('RGB')
            img = _transform(img)
            img = img.to(device_, dtype=torch.float32)
            label_predicted = model_old(img.unsqueeze(0))[0].cpu().numpy().squeeze()
            label_predicted = np.argmax(label_predicted, axis=0).astype(np.uint8)
            label_predicted = Image.fromarray(label_predicted)
            label_predicted.save( lbl_name )
        # break
    print()