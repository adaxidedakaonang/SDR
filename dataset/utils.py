import torch
import numpy as np
import os
from PIL import Image, ImageFile
from dataset import transform as tf
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch import distributed, zeros_like, unique
import copy

def group_images(dataset, labels):
    # Group images based on the label in LABELS (using labels not reordered)
    idxs = {lab: [] for lab in labels}

    labels_cum = labels + [0, 255]
    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if all(x in labels_cum for x in cls):
            for x in cls:
                if x in labels:
                    idxs[x].append(i)
    return idxs


def filter_images(dataset, labels, labels_old=None, overlap=True):
    # Filter images without any label in LABELS (using labels not reordered)
    idxs = []

    if 0 in labels:
        labels.remove(0)

    print(f"Filtering images...")
    if labels_old is None:
        labels_old = []
    labels_cum = labels + labels_old + [0, 255]

    if overlap:
        fil = lambda c: any(x in labels for x in cls)
    else:  # disjoint and no_mask (ICCVW2019) datasets are the same, only label space changes
        fil = lambda c: any(x in labels for x in cls) and all(x in labels_cum for x in c)

    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if fil(cls):
            idxs.append(i)
        if i % 1000 == 0:
            print(f"\t{i}/{len(dataset)} ...")
    return idxs

def mix_labels(img_list, idxs, labels, tmp_path="./tmp", opts=None):
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
        print(f"Previous model loaded from {path}")
        # logger.info(f"[!] Previous model loaded from {path}")
            # clean memory
        del step_checkpoint['model_state']
        model_old.eval()
        for p in model_old.parameters():
            p.requires_grad = False
        return model_old

    def _mix_label(label_under, label_on):
        _mask = label_on != 0
        r,c = label_on.shape
        mixed_label = np.zeros([r,c])
        mixed_label[_mask] = label_on[_mask]
        mixed_label[~_mask] = label_under[~_mask]
        return mixed_label.astype(np.uint8)
    
    def _create_folder(path_):
        import os
        if not os.path.exists(path_):
            os.makedirs(path_)

    def _clear_folder(path_):
        import os
        if os.listdir(path_):
            # print("Clear the directory !!!")
            for _ in os.listdir(path_):
                os.remove( os.path.join(path_, _) )
                print("Remove file: " + os.path.join(path_, _))

    def _tmp_funct3(x):
        tmp = copy.deepcopy(x)
        for value in np.unique(x):
            if value in labels + [255]:
                continue
            else:
                new_value = 0
            tmp[x == value] = new_value
        return tmp

    import time
    from tqdm import tqdm

    start_time = time.time()

    img_path = os.path.join(tmp_path, opts.task + "/step_" + str(opts.step) + "/image")
    mix_path = os.path.join(tmp_path, opts.task + "/step_" + str(opts.step) + "/mixed_label")
    on_path = os.path.join(tmp_path, opts.task + "/step_" + str(opts.step) + "/src_label")
    under_path = os.path.join(tmp_path, opts.task + "/step_" + str(opts.step) + "/predict_label")
    _create_folder(img_path)
    _create_folder(mix_path)
    _create_folder(on_path)
    _create_folder(under_path)
    # print("Clear the directory !!!")
    # _clear_folder(img_path)
    # _clear_folder(mix_path)
    # _clear_folder(on_path)
    # _clear_folder(under_path)
    device_ = torch.device('cuda')
    model_old = _build_model()
    model_old = model_old.to(device=device_)

    for idx in tqdm(idxs):
        image_path = img_list[idx][0]
        label_path = img_list[idx][1]
        img_name = os.path.split(image_path)[1][:-4] + ".png"
        new_image_path = os.path.join( img_path, img_name )
        new_label_path = os.path.join( mix_path, img_name ) 
        if not (os.path.isfile(new_image_path) and os.path.isfile(new_label_path)):
            # print(new_image_path)
            _img = Image.open(image_path).convert('RGB')
            _target = Image.open(label_path)
            img, target = _transform(_img, _target)
            
            img = img.to(device_, dtype=torch.float32)
            label_under = model_old(img.unsqueeze(0))[0].cpu().numpy().squeeze()
            label_under = np.argmax(label_under, axis=0).astype(np.uint8)
            label_on = np.array(_target)
            label_on = _tmp_funct3(label_on)
            label_on[ label_on==255 ] = 0
            mixed_label = _mix_label(label_under, label_on)
            source_img = _img
            label_under = Image.fromarray(label_under)
            label_on = Image.fromarray(label_on)
            label_mix = Image.fromarray(mixed_label)
            source_img.save( os.path.join( img_path, img_name ) )
            label_under.save( os.path.join( under_path, img_name ) )
            label_on.save( os.path.join( on_path, img_name ) )
            label_mix.save( os.path.join( mix_path, img_name ) )
        img_list[idx][0] = new_image_path
        img_list[idx][1] = new_label_path
    print("Finish mixing labels.")
    print("Total time: " + str(time.time() - start_time) + " seconds")
    return img_list

class Subset(torch.utils.data.Dataset):
    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform (callable): way to transform the images and the targets
        target_transform(callable): way to transform the target labels
    """

    def __init__(self, dataset, indices, transform=None, target_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        # print(f"subset {idx}")
        sample, target = self.dataset[self.indices[idx]]

        if self.transform is not None:
            sample, target = self.transform(sample, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.indices)


class Replayset(torch.utils.data.Dataset):
    """
    A dataset that return the flickr downloaded dataset.
    Arguments:
        path (string): dir path of replay images
        labels_old (list): old label
    """

    def __init__(self, path, num_per_class, labels_old, transform=None):
        import os
        print("Adding replay images")
        self.base_path = path
        self.num = num_per_class
        assert self.num>0, "number of per class should be larger than 0."
        self.labels_old = labels_old
        self.transform = transform
        self.replay_lists = []
        self._get_datalist()
        print(f"Loading path: {self.base_path}.")
        print(f"Finish loading {len(self.replay_lists)} images, {len(self.replay_lists)//len(self.labels_old[1:])} images per class.")
        
    def _get_datalist(self):
        files = os.listdir(self.base_path)
        for i in self.labels_old[1:]:
            print(files[i-1])
            full_path = os.path.join(self.base_path, files[i-1] + "/train_fullPath.txt")
            with open(full_path, 'r') as f:
                file_names = [x[:-1].split(' ') for x in f.readlines()]
            print("Check the label path !!!")
            print(file_names[0])
            tmp = [ (x[0][:], x[1][:]) for x in file_names]
            tmp_len = len(tmp)
            if not self.num>tmp_len:
                self.replay_lists+=(tmp[:self.num])
            else:
                print(f"class {files[i-1]} not satisfais the number {self.num}, actual number is {tmp_len}")    
                self.replay_lists+=(tmp)
        print()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # for i in range(len(self.replay_lists)):
        #     img = Image.open(self.replay_lists[i][0]).convert('RGB')
        #     print(self.replay_lists[i][0])
        # print()
        # print(f"replay {index}")
        img = Image.open(self.replay_lists[index][0]).convert('RGB')
        target = Image.open(self.replay_lists[index][1])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.replay_lists)

class ConcateDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
    
    def __getitem__(self, idx):
        return tuple(d[idx % len(d)] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)

    

class MaskLabels:
    """
    Use this class to mask labels that you don't want in your dataset.
    Arguments:
    labels_to_keep (list): The list of labels to keep in the target images
    mask_value (int): The value to replace ignored values (def: 0)
    """
    def __init__(self, labels_to_keep, mask_value=0):
        self.labels = labels_to_keep
        self.value = torch.tensor(mask_value, dtype=torch.uint8)

    def __call__(self, sample):
        # sample must be a tensor
        assert isinstance(sample, torch.Tensor), "Sample must be a tensor"

        sample.apply_(lambda t: t.apply_(lambda x: x if x in self.labels else self.value))

        return sample
