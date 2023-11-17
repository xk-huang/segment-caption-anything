# %%
import csv, operator
import json
import os.path as osp
import warnings
from collections import OrderedDict, defaultdict

# import mmcv
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset

from PIL import Image
import torch
import random
import pycocotools.mask as maskUtils
import pickle
from .utils import DATA_PATH_TABLE

# from mmcv.runner import get_dist_info
# from mmcv.utils import print_log

# from mmdet.core import eval_map
# from .builder import DATASETS
# from .custom import CustomDataset


class OpenImagesDataset(Dataset):
    """Open Images dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        label_file (str): File path of the label description file that
            maps the classes names in MID format to their short
            descriptions.
        image_level_ann_file (str): Image level annotation, which is used
            in evaluation.
        get_supercategory (bool): Whether to get parent class of the
            current class. Default: True.
        hierarchy_file (str): The file path of the class hierarchy.
            Default: None.
        get_metas (bool): Whether to get image metas in testing or
            validation time. This should be `True` during evaluation.
            Default: True. The OpenImages annotations do not have image
            metas (width and height of the image), which will be used
            during evaluation. We provide two ways to get image metas
            in `OpenImagesDataset`:

            - 1. `load from file`: Load image metas from pkl file, which
              is suggested to use. We provided a script to get image metas:
              `tools/misc/get_image_metas.py`, which need to run
              this script before training/testing. Please refer to
              `config/openimages/README.md` for more details.

            - 2. `load from pipeline`, which will get image metas during
              test time. However, this may reduce the inference speed,
              especially when using distribution.

        load_from_file (bool): Whether to get image metas from pkl file.
        meta_file (str): File path to get image metas.
        filter_labels (bool): Whether filter unannotated classes.
            Default: True.
        load_image_level_labels (bool): Whether load and consider image
            level labels during evaluation. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self, task_indicator, split="train"):
        data_path = DATA_PATH_TABLE["openimages"]
        self.data_path = data_path
        self.split = split
        self.task_indicator = task_indicator
        if task_indicator in ["objectdet", "objectloc", "classification", "inpainting"]:
            ann_file = osp.join(
                data_path,
                "oidv6-train-annotations-bbox.csv"
                if split == "train"
                else "validation-annotations-bbox.csv",
            )
        elif task_indicator in ["insseg", "objseg", "imagesynthesisfromseg"]:
            ann_file = osp.join(
                data_path,
                "train-annotations-object-segmentation_sorted.csv"
                if split == "train"
                else "validation-annotations-object-segmentation_sorted.csv",
            )
            self.segs_path = osp.join(data_path, "segs_merge")
        elif task_indicator in ["relationdet"]:
            ann_file = osp.join(data_path, f"oidv6-{split}-annotations-vrd.csv")
        elif task_indicator in ["caption"]:
            ann_file = (
                [
                    osp.join(
                        data_path,
                        f"open_images_train_v6_localized_narratives-0000{i}-of-00010.jsonl",
                    )
                    for i in range(9)
                ]
                if split == "train"
                else [
                    osp.join(
                        data_path, f"open_images_{split}_localized_narratives.jsonl"
                    )
                ]
            )
        self.ann_file = ann_file
        self.cat2label = defaultdict(str)
        label_file = osp.join(data_path, "oidv7-class-descriptions-boxable.csv")
        if task_indicator in ["relationdet"]:
            attr_label_file = osp.join(data_path, "oidv6-attributes-description.csv")
            class_names = self.get_classes_from_csv_rel(label_file, attr_label_file)
            relation_label_file = osp.join(
                data_path, "oidv6-relationships-description.csv"
            )
            self.relation_map = self.get_relation_from_csv(relation_label_file)
        else:
            class_names = self.get_classes_from_csv(label_file)
        self.CLASSES = class_names

        self.data_infos = self.load_annotations(self.ann_file)

    def get_relation_from_csv(self, label_file):
        relation_map = {}
        with open(label_file, "r") as f:
            reader = csv.reader(f)
            for line in reader:
                relation_map[line[0]] = line[1]
        return relation_map

    def get_classes_from_csv_rel(self, label_file, attr_label_file):
        """Get classes name from file.

        Args:
            label_file (str): File path of the label description file that
                maps the classes names in MID format to their short
                descriptions.

        Returns:
            list[str]: Class name of OpenImages.
        """

        index_list = []
        classes_names = []
        with open(label_file, "r") as f:
            reader = csv.reader(f)
            for line in reader:
                self.cat2label[line[0]] = line[1]
                classes_names.append(line[1])
                index_list.append(line[0])
        self.attr_start = len(index_list)
        with open(attr_label_file, "r") as f:
            reader = csv.reader(f)
            for line in reader:
                assert line[0] not in self.cat2label
                self.cat2label[line[0]] = line[1]
                classes_names.append(line[1])
                index_list.append(line[0])
        self.index_dict = {index: i for i, index in enumerate(index_list)}
        return classes_names

    def get_classes_from_csv(self, label_file):
        """Get classes name from file.

        Args:
            label_file (str): File path of the label description file that
                maps the classes names in MID format to their short
                descriptions.

        Returns:
            list[str]: Class name of OpenImages.
        """

        index_list = []
        classes_names = []
        with open(label_file, "r") as f:
            reader = csv.reader(f)
            for line in reader:
                self.cat2label[line[0]] = line[1]
                classes_names.append(line[1])
                index_list.append(line[0])
        self.index_dict = {index: i for i, index in enumerate(index_list)}
        return classes_names

    def load_annotations(self, ann_file):
        if self.task_indicator == "objectdet":
            return self.load_annotations_det(ann_file)
        elif self.task_indicator == "objectloc":
            return self.load_annotations_loc(ann_file)
        elif self.task_indicator in ["insseg", "imagesynthesisfromseg"]:
            return self.load_annotations_insseg(ann_file)
        elif self.task_indicator == "objseg":
            return self.load_annotations_objseg(ann_file)
        elif self.task_indicator in ["classification", "inpainting"]:
            return self.load_annotations_cls(ann_file)
        elif self.task_indicator in ["relationdet"]:
            return self.load_annotations_rel(ann_file)
        elif self.task_indicator in ["caption"]:
            return self.load_annotations_cap(ann_file)

    def load_annotations_det(self, ann_file):
        data_infos = []
        last_img_id = None

        with open(ann_file, "r") as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                img_id = line[0]
                label_id = line[2]
                assert label_id in self.index_dict
                label = int(self.index_dict[label_id])
                bbox = [
                    float(line[4]),  # xmin
                    float(line[6]),  # ymin
                    float(line[5]),  # xmax
                    float(line[7]),  # ymax
                ]
                is_occluded = True if int(line[8]) == 1 else False
                is_truncated = True if int(line[9]) == 1 else False
                is_group_of = True if int(line[10]) == 1 else False
                is_depiction = True if int(line[11]) == 1 else False
                is_inside = True if int(line[12]) == 1 else False

                if img_id != last_img_id:
                    if last_img_id is not None:
                        data_infos.append(
                            dict(
                                img_id=last_img_id,
                                bboxes=torch.tensor(last_bbox),
                                labels=torch.tensor(last_label),
                                is_occluded=torch.tensor(last_is_occluded),
                                is_truncated=torch.tensor(last_is_truncated),
                                is_group_of=torch.tensor(last_is_group_of),
                                is_depiction=torch.tensor(last_is_depiction),
                                is_inside=torch.tensor(last_is_inside),
                            )
                        )
                    # reset
                    last_img_id = img_id
                    last_bbox = []
                    last_label = []
                    last_is_occluded = []
                    last_is_truncated = []
                    last_is_group_of = []
                    last_is_depiction = []
                    last_is_inside = []

                last_bbox.append(bbox)
                last_label.append(label)
                last_is_occluded.append(is_occluded)
                last_is_truncated.append(is_truncated)
                last_is_group_of.append(is_group_of)
                last_is_depiction.append(is_depiction)
                last_is_inside.append(is_inside)

            data_infos.append(
                dict(
                    img_id=last_img_id,
                    bboxes=torch.tensor(last_bbox),
                    labels=torch.tensor(last_label),
                    is_occluded=torch.tensor(last_is_occluded),
                    is_truncated=torch.tensor(last_is_truncated),
                    is_group_of=torch.tensor(last_is_group_of),
                    is_depiction=torch.tensor(last_is_depiction),
                    is_inside=torch.tensor(last_is_inside),
                )
            )

        return data_infos

    def collect_by_label(self, res, save_select_id=False):
        """collect dict by labels."""
        img_id = res.pop("img_id")
        collect_res = []
        labels = res.pop("labels")
        for label_select in labels.unique():
            select_ids = torch.nonzero(label_select == labels, as_tuple=True)[0]
            tmp_res = dict(
                img_id=img_id,
                labels=label_select,
                **{k: v[select_ids] for k, v in res.items()},
            )
            if save_select_id:
                tmp_res["select_ids"] = select_ids
            collect_res.append(tmp_res)
        return collect_res

    def load_annotations_loc(self, ann_file):
        data_infos = []
        last_img_id = None

        with open(ann_file, "r") as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                img_id = line[0]
                label_id = line[2]
                assert label_id in self.index_dict
                label = int(self.index_dict[label_id])
                bbox = [
                    float(line[4]),  # xmin
                    float(line[6]),  # ymin
                    float(line[5]),  # xmax
                    float(line[7]),  # ymax
                ]
                is_occluded = True if int(line[8]) == 1 else False
                is_truncated = True if int(line[9]) == 1 else False
                is_group_of = True if int(line[10]) == 1 else False
                is_depiction = True if int(line[11]) == 1 else False
                is_inside = True if int(line[12]) == 1 else False

                if img_id != last_img_id:
                    if last_img_id is not None:
                        data_infos.extend(
                            self.collect_by_label(
                                dict(
                                    img_id=last_img_id,
                                    bboxes=torch.tensor(last_bbox),
                                    labels=torch.tensor(last_label),
                                    is_occluded=torch.tensor(last_is_occluded),
                                    is_truncated=torch.tensor(last_is_truncated),
                                    is_group_of=torch.tensor(last_is_group_of),
                                    is_depiction=torch.tensor(last_is_depiction),
                                    is_inside=torch.tensor(last_is_inside),
                                )
                            )
                        )
                    # reset
                    last_img_id = img_id
                    last_bbox = []
                    last_label = []
                    last_is_occluded = []
                    last_is_truncated = []
                    last_is_group_of = []
                    last_is_depiction = []
                    last_is_inside = []

                last_bbox.append(bbox)
                last_label.append(label)
                last_is_occluded.append(is_occluded)
                last_is_truncated.append(is_truncated)
                last_is_group_of.append(is_group_of)
                last_is_depiction.append(is_depiction)
                last_is_inside.append(is_inside)

            data_infos.extend(
                self.collect_by_label(
                    dict(
                        img_id=last_img_id,
                        bboxes=torch.tensor(last_bbox),
                        labels=torch.tensor(last_label),
                        is_occluded=torch.tensor(last_is_occluded),
                        is_truncated=torch.tensor(last_is_truncated),
                        is_group_of=torch.tensor(last_is_group_of),
                        is_depiction=torch.tensor(last_is_depiction),
                        is_inside=torch.tensor(last_is_inside),
                    )
                )
            )

        self.cls2index = defaultdict(list)
        for i, data in enumerate(data_infos):
            self.cls2index[self.CLASSES[data["labels"]]].append(i)

        return data_infos

    def load_annotations_cls(self, ann_file):
        data_infos = []

        with open(ann_file, "r") as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                img_id = line[0]
                label_id = line[2]
                assert label_id in self.index_dict
                label = int(self.index_dict[label_id])
                bbox = [
                    float(line[4]),  # xmin
                    float(line[6]),  # ymin
                    float(line[5]),  # xmax
                    float(line[7]),  # ymax
                ]
                is_occluded = True if int(line[8]) == 1 else False
                is_truncated = True if int(line[9]) == 1 else False
                is_group_of = True if int(line[10]) == 1 else False
                is_depiction = True if int(line[11]) == 1 else False
                is_inside = True if int(line[12]) == 1 else False

                data_infos.append(
                    dict(
                        img_id=img_id,
                        bboxes=torch.tensor(bbox),
                        labels=label,
                        is_occluded=is_occluded,
                        is_truncated=is_truncated,
                        is_group_of=is_group_of,
                        is_depiction=is_depiction,
                        is_inside=is_inside,
                    )
                )

        return data_infos

    def load_annotations_insseg(self, ann_file):
        data_infos = []
        last_img_id = None

        with open(ann_file, "r") as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                img_id = line[1]
                label_id = line[2]
                assert label_id in self.index_dict
                label = int(self.index_dict[label_id])
                bbox = [
                    float(line[4]),  # xmin
                    float(line[6]),  # ymin
                    float(line[5]),  # xmax
                    float(line[7]),  # ymax
                ]
                predictediou = float(line[8])

                if img_id != last_img_id:
                    if last_img_id is not None:
                        data_infos.append(
                            dict(
                                img_id=last_img_id,
                                bboxes=torch.tensor(last_bbox),
                                labels=torch.tensor(last_label),
                                predictediou=torch.tensor(last_predictediou),
                            )
                        )
                    # reset
                    last_img_id = img_id
                    last_bbox = []
                    last_label = []
                    last_predictediou = []

                last_bbox.append(bbox)
                last_label.append(label)
                last_predictediou.append(predictediou)

            data_infos.append(
                dict(
                    img_id=last_img_id,
                    bboxes=torch.tensor(last_bbox),
                    labels=torch.tensor(last_label),
                    predictediou=torch.tensor(last_predictediou),
                )
            )

        return data_infos

    def load_annotations_objseg(self, ann_file):
        data_infos = []
        last_img_id = None

        with open(ann_file, "r") as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                img_id = line[1]
                label_id = line[2]
                assert label_id in self.index_dict
                label = int(self.index_dict[label_id])
                bbox = [
                    float(line[4]),  # xmin
                    float(line[6]),  # ymin
                    float(line[5]),  # xmax
                    float(line[7]),  # ymax
                ]
                predictediou = float(line[8])

                if img_id != last_img_id:
                    if last_img_id is not None:
                        data_infos.extend(
                            self.collect_by_label(
                                dict(
                                    img_id=last_img_id,
                                    bboxes=torch.tensor(last_bbox),
                                    labels=torch.tensor(last_label),
                                    predictediou=torch.tensor(last_predictediou),
                                ),
                                save_select_id=True,
                            )
                        )
                    # reset
                    last_img_id = img_id
                    last_bbox = []
                    last_label = []
                    last_predictediou = []

                last_bbox.append(bbox)
                last_label.append(label)
                last_predictediou.append(predictediou)

            data_infos.extend(
                self.collect_by_label(
                    dict(
                        img_id=last_img_id,
                        bboxes=torch.tensor(last_bbox),
                        labels=torch.tensor(last_label),
                        predictediou=torch.tensor(last_predictediou),
                    ),
                    save_select_id=True,
                )
            )

        self.cls2index = defaultdict(list)
        for i, data in enumerate(data_infos):
            self.cls2index[self.CLASSES[data["labels"]]].append(i)
        return data_infos

    def load_annotations_rel(self, ann_file):
        data_infos = []

        with open(ann_file, "r") as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                img_id = line[0]
                label_id1 = line[1]
                label_id2 = line[2]
                label1 = int(self.index_dict[label_id1])
                label2 = int(self.index_dict[label_id2])
                bbox1 = [
                    float(line[3]),  # xmin
                    float(line[5]),  # ymin
                    float(line[4]),  # xmax
                    float(line[6]),  # ymax
                ]
                bbox2 = [
                    float(line[7]),  # xmin
                    float(line[9]),  # ymin
                    float(line[8]),  # xmax
                    float(line[10]),  # ymax
                ]
                relation_label = line[11]

                data_infos.append(
                    dict(
                        img_id=img_id,
                        bbox1=torch.tensor(bbox1),
                        label1=label1,
                        bbox2=torch.tensor(bbox2),
                        label2=label2,
                        relation=self.relation_map[relation_label],
                    )
                )

        return data_infos

    def load_annotations_cap(self, ann_file_list):
        data_infos = []
        for ann_file in ann_file_list:
            with open(ann_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    data_infos.append(
                        dict(
                            img_id=data["image_id"],
                            annotator_id=data["annotator_id"],
                            caption=data["caption"],
                        )
                    )

        return data_infos

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def get_img_path(self, filename):
        if self.split == "train":
            dirname = f"train_{filename[0]}"
        elif self.split in ["val", "validation"]:
            dirname = "validation"
        elif self.split == "test":
            dirname = "test"
        return osp.join(self.data_path, "v6img", dirname, filename)

    def get_seg(self, img_id, select_ids=None):
        with open(osp.join(self.segs_path, f"{img_id}.pkl"), "rb") as f:
            seg = pickle.load(f)
        if select_ids is None:
            masks = maskUtils.decode(seg)
        else:
            masks = maskUtils.decode([seg[idx.item()] for idx in select_ids])
        return masks.transpose(2, 0, 1)

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn(
                "OpenImageDatasets does not support " "filtering empty gt images."
            )
        valid_inds = [i for i in range(len(self))]
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio."""
        self.flag = np.zeros(len(self), dtype=np.uint8)
        # TODO: set flag without width and height

    def get_relation_matrix(self, hierarchy_file):
        """Get hierarchy for classes.

        Args:
            hierarchy_file (sty): File path to the hierarchy for classes.

        Returns:
            ndarray: The matrix of the corresponding relationship between
            the parent class and the child class, of shape
            (class_num, class_num).
        """

        if self.data_root is not None:
            if not osp.isabs(hierarchy_file):
                hierarchy_file = osp.join(self.data_root, hierarchy_file)
        with open(hierarchy_file, "r") as f:
            hierarchy = json.load(f)
        class_num = len(self.CLASSES)
        class_label_tree = np.eye(class_num, class_num)
        class_label_tree = self._convert_hierarchy_tree(hierarchy, class_label_tree)
        return class_label_tree

    def _convert_hierarchy_tree(
        self, hierarchy_map, class_label_tree, parents=[], get_all_parents=True
    ):
        """Get matrix of the corresponding relationship between the parent
        class and the child class.

        Args:
            hierarchy_map (dict): Including label name and corresponding
                subcategory. Keys of dicts are:

                - `LabeName` (str): Name of the label.
                - `Subcategory` (dict | list): Corresponding subcategory(ies).
            class_label_tree (ndarray): The matrix of the corresponding
                relationship between the parent class and the child class,
                of shape (class_num, class_num).
            parents (list): Corresponding parent class.
            get_all_parents (bool): Whether get all parent names.
                Default: True

        Returns:
            ndarray: The matrix of the corresponding relationship between
            the parent class and the child class, of shape
            (class_num, class_num).
        """

        if "Subcategory" in hierarchy_map:
            for node in hierarchy_map["Subcategory"]:
                if "LabelName" in node:
                    children_name = node["LabelName"]
                    children_index = self.index_dict[children_name]
                    children = [children_index]
                else:
                    continue
                if len(parents) > 0:
                    for parent_index in parents:
                        if get_all_parents:
                            children.append(parent_index)
                        class_label_tree[children_index, parent_index] = 1

                class_label_tree = self._convert_hierarchy_tree(
                    node, class_label_tree, parents=children
                )

        return class_label_tree

    def add_supercategory_ann(self, annotations):
        """Add parent classes of the corresponding class of the ground truth
        bboxes."""
        for i, ann in enumerate(annotations):
            assert (
                len(ann["labels"]) == len(ann["bboxes"]) == len(ann["gt_is_group_ofs"])
            )
            gt_bboxes = []
            gt_is_group_ofs = []
            gt_labels = []
            for j in range(len(ann["labels"])):
                label = ann["labels"][j]
                bbox = ann["bboxes"][j]
                is_group = ann["gt_is_group_ofs"][j]
                label = np.where(self.class_label_tree[label])[0]
                if len(label) > 1:
                    for k in range(len(label)):
                        gt_bboxes.append(bbox)
                        gt_is_group_ofs.append(is_group)
                        gt_labels.append(label[k])
                else:
                    gt_bboxes.append(bbox)
                    gt_is_group_ofs.append(is_group)
                    gt_labels.append(label[0])
            annotations[i] = dict(
                bboxes=np.array(gt_bboxes).astype(np.float32),
                labels=np.array(gt_labels).astype(np.int64),
                bboxes_ignore=ann["bboxes_ignore"],
                gt_is_group_ofs=np.array(gt_is_group_ofs).astype(np.bool),
            )

        return annotations

    def load_image_label_from_csv(self, image_level_ann_file):
        """Load image level annotations from csv style ann_file.

        Args:
            image_level_ann_file (str): CSV style image level annotation
                file path.

        Returns:
            defaultdict[list[dict]]: Annotations where item of the defaultdict
            indicates an image, each of which has (n) dicts.
            Keys of dicts are:

                - `image_level_label` (int): Label id.
                - `confidence` (float): Labels that are human-verified to be
                  present in an image have confidence = 1 (positive labels).
                  Labels that are human-verified to be absent from an image
                  have confidence = 0 (negative labels). Machine-generated
                  labels have fractional confidences, generally >= 0.5.
                  The higher the confidence, the smaller the chance for
                  the label to be a false positive.
        """

        item_lists = defaultdict(list)
        with open(image_level_ann_file, "r") as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                img_id = line[0]
                item_lists[img_id].append(
                    dict(
                        image_level_label=int(self.index_dict[line[2]]),
                        confidence=float(line[3]),
                    )
                )
        return item_lists

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        return self.get_ann_info(idx)["labels"].astype(np.int).tolist()

    def __getitem__(self, index):
        if self.task_indicator in ["objectdet"]:
            ann_info = self.data_infos[index]
            res = dict(
                image=Image.open(self.get_img_path(f"{ann_info['img_id']}.jpg")),
                class_name=[self.CLASSES[l] for l in ann_info["labels"]],
                box=ann_info["bboxes"],
            )
            return res
        elif self.task_indicator in ["objectloc"]:
            ann_info = self.data_infos[index]
            res = dict(
                image=Image.open(self.get_img_path(f"{ann_info['img_id']}.jpg")),
                class_name=self.CLASSES[ann_info["labels"]],
                box=ann_info["bboxes"],
            )
            return res
        elif self.task_indicator in ["classification", "inpainting"]:
            ann_info = self.data_infos[index]
            res = dict(
                image=Image.open(self.get_img_path(f"{ann_info['img_id']}.jpg")),
                class_name=self.CLASSES[ann_info["labels"]],
                box=ann_info["bboxes"],
            )
            return res
        elif self.task_indicator in ["insseg", "imagesynthesisfromseg"]:
            ann_info = self.data_infos[index]
            res = dict(
                image=Image.open(self.get_img_path(f"{ann_info['img_id']}.jpg")),
                class_name=[self.CLASSES[l] for l in ann_info["labels"]],
                box=ann_info["bboxes"],
                mask=torch.from_numpy(self.get_seg(ann_info["img_id"])).bool(),
            )
            return res
        elif self.task_indicator in ["objseg"]:
            ann_info = self.data_infos[index]
            res = dict(
                image=Image.open(self.get_img_path(f"{ann_info['img_id']}.jpg")),
                class_name=self.CLASSES[ann_info["labels"]],
                box=ann_info["bboxes"],
                mask=torch.from_numpy(self.get_seg(ann_info["img_id"], select_ids=ann_info["select_ids"])).bool(),
            )
            return res
        elif self.task_indicator in ["relationdet"]:
            ann_info = self.data_infos[index]
            assert ann_info["label1"] < self.attr_start  # first image always object
            res = dict(
                image=Image.open(self.get_img_path(f"{ann_info['img_id']}.jpg")),
                box1=ann_info["bbox1"],
                box2=ann_info["bbox2"],
                label1=self.CLASSES[ann_info["label1"]],
                label2=self.CLASSES[ann_info["label2"]],
                is_attr1=ann_info["label1"] >= self.attr_start,
                is_attr2=ann_info["label2"] >= self.attr_start,
                relation=ann_info["relation"],
            )
            return res
        elif self.task_indicator in ["caption"]:
            ann_info = self.data_infos[index]
            res = dict(
                image=Image.open(self.get_img_path(f"{ann_info['img_id']}.jpg")),
                text=ann_info["caption"],
            )
            return res


# testing
# objectdet = OpenImagesDataset('/data/home/v-jining/data/openimage-v6', task_indicator='objectdet', split='validation')
# print(objectdet[3])
# objectloc = OpenImagesDataset('/data/home/v-jining/data/openimage-v6', task_indicator='objectloc', split='validation')
# print(objectloc[2])
# objectcat = OpenImagesDataset('/data/home/v-jining/data/openimage-v6', task_indicator='classification', split='validation')
# print(objectcat[0])
# insseg = OpenImagesDataset('/data/home/v-jining/data/openimage-v6', task_indicator='insseg', split='validation')
# print(insseg[1])
# objseg = OpenImagesDataset('/data/home/v-jining/data/openimage-v6', task_indicator='objseg', split='train')
# print(objseg[0])
# relationdet = OpenImagesDataset('/data/home/v-jining/data/openimage-v6', task_indicator='relationdet', split='validation')
# print(relationdet[3])
# caption = OpenImagesDataset('/data/home/v-jining/data/openimage-v6', task_indicator='caption', split='validation')
# print(caption[0])

# exit()


# %%
