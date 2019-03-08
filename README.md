# PartNet Dataset

![Dataset Overview](https://github.com/daerduoCarey/partnet_dataset/blob/master/images/data_visu.png)

**Figure 1. The PartNet Dataset Example Visualzation.**

We present PartNet: a consistent, large-scale dataset of 3D objects annotated with fine-grained, instance-level, and hierarchical 3D part information. Our dataset consists of 573,585 part instances over 26,671 3D models covering 24 object categories. This dataset enables and serves as a catalyst for many tasks such as shape analysis, dynamic 3D scene modeling and simulation, affordance analysis, and others. Using our dataset, we establish three benchmarking tasks for evaluating 3D part recognition: fine-grained semantic segmentation, hierarchical semantic segmentation, and instance segmentation. We benchmark four state-of-the-art 3D deep learning algorithms for fine-grained semantic segmentation and three baseline methods for hierarchical semantic segmentation. We also propose a novel method for part instance segmentation and demonstrate its superior performance over existing methods.

## About the paper

PartNet is accepted to CVPR 2019. 

Our team: [Kaichun Mo](https://cs.stanford.edu/~kaichun), [Shilin Zhu](http://cseweb.ucsd.edu/~shz338/), [Angel X. Chang](https://angelxuanchang.github.io/), [Li Yi](https://cs.stanford.edu/~ericyi/), [Subarna Tripathi](https://subarnatripathi.github.io/), [Leonidas J. Guibas](https://geometry.stanford.edu/member/guibas/) and [Hao Su](http://cseweb.ucsd.edu/~haosu/) from Stanford, UCSD, SFU and Intel AI Lab.

Arxiv Version: https://arxiv.org/abs/1812.02713

Project Page: https://cs.stanford.edu/~kaichun/partnet/

Video: https://youtu.be/7pEuoxmb-MI

## Citation

    @article{mo2018partnet,
        title={{PartNet}: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level {3D} Object Understanding},
        author={Mo, Kaichun and Zhu, Shilin and Chang, Angel and Yi, Li and Tripathi, Subarna and Guibas, Leonidas and Su, Hao},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2019}
    }

## About this repository

This repository provides the meta-files for PartNet release v0. 


```
    stats/
        all_valid_anno_info.txt         # Store all valid PartNet Annotation meta-information
                                        # <anno_id, version_id, category, shapenet_model_id, annotator_id>
        before_merging_label_ids/       # Store all expert-defined part semantics before merging
            Chair.txt
            ...
        merging2_hierarchy_mapping/     # Store all merging criterion
            Chair.txt
            ...
        after_merging2_label_ids/       # Store the part semantics after merging
            Chair.txt                   # all part semantics
            Chair-hier.txt              # all part semantics that are selected for Sec 5.2 experiments
            Chair-level-1.txt           # all part semantics that are selected for Sec 5.1 and 5.3 experiments for chair level-1
            Chair-level-2.txt           # all part semantics that are selected for Sec 5.1 and 5.3 experiments for chair level-2
            Chair-level-3.txt           # all part semantics that are selected for Sec 5.1 and 5.3 experiments for chair level-3
            ...
        train_val_test_split/           # An attemptive train/val/test splits (may be changed for official v1 release and PartNet challenges)
            Chair.train.json
            Chair.val.json
            Chair.test.json
    scripts/
            gen_h5_ins_seg_after_merging2.py    # An example usage python script to load PartNet data, check the file for more information
    data/                                       # Download PartNet data from Google Drive and unzip them here
        42/
            result.json                 # A JSON file storing the part hierarchical trees from raw user annotation
            meta.json                   # A JSON file storing all the related meta-information
            objs/                       # A folder containing several part obj files indexed by `result.json`
                                        # Note that the parts here are not the final parts. Each individual obj may not make sense.
                                        # Please refer to `result.json` and read each part's obj files. Maybe many obj files make up one part.
                original-*.obj          # Indicate this is an exact part mesh from the original ShapeNet model
                new-*.obj               # Indicate this is a smoothed and cut-out part mesh in PartNet annotation cutting procedure
            tree_hier.html              # A simple HTML visualzation for the hierarchical annotation
            part_renders/               # A folder with rendered images supporting `tree_hier.html` visualization
            point_sample/               # We sample 10,000 points for point cloud learning
                pts-10000.txt                               # point cloud directly sampled from the combination of part meshes under `objs/`
                label-10000.txt
                sample-points-all-pts-nor-rgba-10000.txt    # point cloud directly sampled from the whole ShapeNet model with labels transferred from `label-10000.txt`
                sample-points-all-label-10000.txt

```

## Questions

Please post issues for questions and more helps on this Github repo page.


## License

MIT Licence

