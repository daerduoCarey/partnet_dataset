# PartNet: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding 

![Dataset Overview](https://github.com/daerduoCarey/partnet_dataset/blob/master/images/data_visu.png)

**Figure 1. The PartNet Dataset Example Visualization.**

## Introduction

We present PartNet: a consistent, large-scale dataset of 3D objects annotated with fine-grained, instance-level, and hierarchical 3D part information. Our dataset consists of 573,585 part instances over 26,671 3D models covering 24 object categories. This dataset enables and serves as a catalyst for many tasks such as shape analysis, dynamic 3D scene modeling and simulation, affordance analysis, and others. Using our dataset, we establish three benchmarking tasks for evaluating 3D part recognition: fine-grained semantic segmentation, hierarchical semantic segmentation, and instance segmentation. We benchmark four state-of-the-art 3D deep learning algorithms for fine-grained semantic segmentation and three baseline methods for hierarchical semantic segmentation. We also propose a novel method for part instance segmentation and demonstrate its superior performance over existing methods.

## About the paper

PartNet is accepted to CVPR 2019. See you at Long Beach, LA.

Our team: [Kaichun Mo](https://cs.stanford.edu/~kaichun), [Shilin Zhu](http://cseweb.ucsd.edu/~shz338/), [Angel X. Chang](https://angelxuanchang.github.io/), [Li Yi](https://cs.stanford.edu/~ericyi/), [Subarna Tripathi](https://subarnatripathi.github.io/), [Leonidas J. Guibas](https://geometry.stanford.edu/member/guibas/) and [Hao Su](http://cseweb.ucsd.edu/~haosu/) from Stanford, UCSD, SFU and Intel AI Lab.

Arxiv Version: https://arxiv.org/abs/1812.02713

Project Page: https://partnet.cs.stanford.edu/

Video: https://youtu.be/7pEuoxmb-MI

## About the Dataset

PartNet is part of the ShapeNet efforts and we provide the PartNet data downloading instructions on [the ShapeNet official webpage](https://www.shapenet.org/download/parts). You need to become a registered user in order to download the data. Please fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSetsP7aj-Hy0gvP2FxRT3aTIrc_IMqSqR-5Xl8P3x2awDkQbw/viewform?usp=sf_link) if you have any feedback to us for improving PartNet.

## Data visualization

We make visualization pages for the PartNet data. For the raw annotation (before-merging) one, use [https://partnet.cs.stanford.edu/visu_htmls/42/tree_hier.html](https://partnet.cs.stanford.edu/visu_htmls/42/tree_hier.html). For the final data (after-merging) one, use [https://partnet.cs.stanford.edu/visu_htmls/42/tree_hier_after_merging.html](https://partnet.cs.stanford.edu/visu_htmls/42/tree_hier_after_merging.html). Replace 42 with any annotation id for your model.

## Errata

We have tried our best to design the annotation interface, instruct the annotators on providing high-quality annotations and get cross validations among different workers. We also conducted two-round of data verifications to date to elliminate obvious data annotation errors. 
However, provided that annotating such large-scale fine-grained part segmentation is challenging, there could still be some annotation errors in PartNet. We believe that the error rate should be below 1% counted in parts through a rough examination.

Dear PartNet users, we need your help on improving the quality of PartNet while you use it. If you find any problematic annotation, please let us know by filling in [this errata](https://docs.google.com/spreadsheets/d/1Q_6r9EblZdP9Grhhm2ob4u0FQ8xurAThlgK-qAcjYP0/edit?usp=sharing) for PartNet v0 release. We will fix the errors in the next PartNet release. Thank you!

## Annotation System (3D Web-based GUI)

We release our Annotation Interface in [this repo](https://github.com/daerduocarey/partnet_anno_system).

## PartNet Experiments

Please refer to [this repo](https://github.com/daerduocarey/partnet_seg_exps) for the segmentation experiments (Section 5) in the paper.

## TODOs

* We will host online PartNet challenges on 3D shape fine-grained semantic segmentation, hierarchical semantic segmentation and fine-grained instance segmentation tasks. Stay tuned.
* More annotations are coming. Please fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSetsP7aj-Hy0gvP2FxRT3aTIrc_IMqSqR-5Xl8P3x2awDkQbw/viewform?usp=sf_link) to tell us what annotations you want us to add in PartNet.
* We are integrating PartNet visualization as part of ShapeNet visualization.

## Citations

    @InProceedings{Mo_2019_CVPR,
        author = {Mo, Kaichun and Zhu, Shilin and Chang, Angel X. and Yi, Li and Tripathi, Subarna and Guibas, Leonidas J. and Su, Hao},
        title = {{PartNet}: A Large-Scale Benchmark for Fine-Grained and Hierarchical Part-Level {3D} Object Understanding},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2019}
    }

Please also cite ShapeNet if you use ShapeNet models.

    @article{chang2015shapenet,
        title={Shapenet: An information-rich 3d model repository},
        author={Chang, Angel X and Funkhouser, Thomas and Guibas, Leonidas and Hanrahan, Pat and Huang, Qixing and Li, Zimo and Savarese, Silvio and Savva, Manolis and Song, Shuran and Su, Hao and others},
        journal={arXiv preprint arXiv:1512.03012},
        year={2015}
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
        merging_hierarchy_mapping/      # Store all merging criterion
            Chair.txt
            ...
        after_merging_label_ids/        # Store the part semantics after merging
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
        merge_result_json.py                    # Merge `result.json` (raw annotation) to `result_merging.json` (after semantic clean-up)
                                                # This file will generate a `result_merging.json` in `../data/[anno_id]/` directory
        gen_h5_ins_seg_after_merging.py         # An example usage python script to load PartNet data, check the file for more information
        geometry_utils.py                       # Some useful helper functions for geometry processing
    data/                                       # Download PartNet data from Google Drive and unzip them here
        42/
            result.json                     # A JSON file storing the part hierarchical trees from raw user annotation
            result_after_merging.json       # A JSON file storing the part hierarchical trees after semantics merging (the final data)
            meta.json                       # A JSON file storing all the related meta-information
            objs/                           # A folder containing several part obj files indexed by `result.json`
                                            # Note that the parts here are not the final parts. Each individual obj may not make sense.
                                            # Please refer to `result.json` and read each part's obj files. Maybe many obj files make up one part.
                original-*.obj              # Indicate this is an exact part mesh from the original ShapeNet model
                new-*.obj                   # Indicate this is a smoothed and cut-out part mesh in PartNet annotation cutting procedure
            tree_hier.html                  # A simple HTML visualzation for the hierarchical annotation (before merging)
            part_renders/                   # A folder with rendered images supporting `tree_hier.html` visualization
            tree_hier_after_merging.html    # A simple HTML visualzation for the hierarchical annotation (after merging)
            part_renders_after_merging/     # A folder with rendered images supporting `tree_hier_after_merging.html` visualization
            point_sample/                   # We sample 10,000 points for point cloud learning
                pts-10000.txt                               # point cloud directly sampled from the combination of part meshes under `objs/`
                label-10000.txt                             # the labels are the id in `result.json`
                sample-points-all-pts-nor-rgba-10000.txt    # point cloud directly sampled from the whole ShapeNet model with labels transferred from `label-10000.txt`
                sample-points-all-label-10000.txt           # labels propagated to `sample-points-all-pts-nor-rgba-10000.txt`

```

## Questions

Please post issues for questions and more helps on this Github repo page. For data annotation error, please fill in [this errata](https://docs.google.com/spreadsheets/d/1Q_6r9EblZdP9Grhhm2ob4u0FQ8xurAThlgK-qAcjYP0/edit?usp=sharing).


## License

MIT Licence

## Updates

* [March 29, 2019] Data v0 with updated data format (`result.json` and `result_after_merging.json`) released. Please re-download the data.
* [March 12, 2019] Data v0 released.

