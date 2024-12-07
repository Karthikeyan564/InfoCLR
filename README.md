# InfoSogCLR and HybridSogCLR: CSCE 636 Final Project.

## Authors:
Karthikeyan Renga Rajan (535004635) : krengara@gmail.com

Fred Cui (335006966)

## Getting Started

This repo contains the code for training the CLIP model. We have added two new loss functions: InfoSogCLR(ita_type=sogclr_mine) by Karthikeyan Renga Rajan and HybridSogCLR(ita_type=hybrid) by Fred Cui.

### Environment

Setting up a new virtual environment with Conda:
````bash
env_name='csce636_proj'
conda create -n "$env_name" python=3.10
conda activate "$env_name"
pip install -r requirements.txt
````

### Training and Evaluation

1. Download the data: [cc3m_subset_100k.tar.gz](https://drive.google.com/file/d/142zQjlOw0Xw4tKzXMrQjYE6NtGRTeasT/view?usp=drive_link), a 100k subset of the [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/) dataset; [mscoco_val.tar.gz](https://drive.google.com/file/d/142tMsnclHTTPpnTXHSeNgTUlBk4She6o/view?usp=drive_link), a 5k subset of the [COCO](https://cocodataset.org/#home) val2014 dataset; [clip_train.tar.gz](https://drive.google.com/file/d/142xxRoMaHxX3BIfCw_1b_G_dgu-02Yq3/view?usp=drive_link), captions of the previous datasets; [imagenet/val.tar](https://drive.google.com/file/d/1NXhfhwFy-nhdABACkodgYqm9pomDKE39/view?usp=sharing), [ImageNet](https://www.image-net.org/challenges/LSVRC/index.php) validation set. The code and data should be structured as follows:
    ```
    .
    +--bimodal_exps (code)
    |
    +--clip_train (captions)
    |  +--cc3m_train_subset.json
    |  +--coco_val.json
    |
    +--datasets (images)
    |  +--cc3m_subset_100k
    |  +--mscoco_val
    |  +--imagnet
    |  |  +-- val
    ```
2. To train a model on cc3m, use `run.slurm` if slurm is supported or run
    ```bash
    export PYTHONPATH="$PYTHONPATH:./bimodal_exps"
    export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

    data_path=./datasets
    ann_path=./clip_train
    train_image_root=cc3m_subset_100k/
    data=cc3m
    train_file=${data}_train_subset.json
    gamma=0.8
    rho=6.0
    epochs=30
    ita_type=isogclr_new

    CUDA_VISIBLE_DEVICES=0 python ./bimodal_exps/clip.py \
        --data_path ${data_path} \
        --ann_path ${ann_path} \
        --train_file ${train_file} \
        --train_image_root ${train_image_root} \
        --output_dir ../output/ \
        --init_model \
        --use_amp \
        --ita_type ${ita_type} \
        --tau_init 0.01 \
        --sogclr_gamma ${gamma} --rho_init ${rho} \
        --eta_init 0.03 --sched cosine \
        --no-distributed \
        --epochs ${epochs}
    ```
3. To test the performance of a model on MSCOCO and ImageNet, use `eval.slurm` if slurm is supported or run
    ```bash
    export PYTHONPATH="$PYTHONPATH:./bimodal_exps"
    export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

    data_path=./datasets
    ann_path=./clip_train
    train_image_root=cc3m_subset_100k/
    data=cc3m
    train_file=${data}_train_subset.json
    gamma=0.8
    rho=6.0
    epochs=30
    ita_type=isogclr_new

    CUDA_VISIBLE_DEVICES=0 python ./bimodal_exps/clip.py \
        --data_path ${data_path} \
        --ann_path ${ann_path} \
        --train_file ${train_file} \
        --train_image_root ${train_image_root} \
        --output_dir ../output/ \
        --init_model \
        --use_amp \
        --ita_type ${ita_type} \
        --tau_init 0.01 \
        --sogclr_gamma ${gamma} --rho_init ${rho} \
        --eta_init 0.03 --sched cosine \
        --no-distributed \
        --epochs ${epochs} \
        --evaluate --checkpoint ../checkpoint_30.pth
    ```

## References
[1] Yuan, Z., Wu, Y., Qiu, Z., Du, X., Zhang, L., Zhou, D., & Yang, T. (2022). Provable Stochastic Optimization for Global Contrastive Learning: Small Batch Does Not Harm Performance. ArXiv, abs/2202.12387.

[2] Qiu, Z., Guo, S., Xu, M., Zhao, T., Zhang, L., & Yang, T. (2024). To Cool or not to Cool? Temperature Network Meets Large Foundation Models via DRO. ArXiv, abs/2404.04575.

[3] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. International Conference on Machine Learning.

[4] Bardes, A., Ponce, J., & LeCun, Y. (2021). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. ArXiv, abs/2105.04906.

[5] Parulekar, A., Collins, L., Shanmugam, K., Mokhtari, A., & Shakkottai, S. (2023). InfoNCE Loss Provably Learns Cluster-Preserving Representations. Annual Conference Computational Learning Theory.

[6] Wang, B., Lei, Y., Ying, Y., & Yang, T. (2024). On Discriminative Probabilistic Modeling for Self-Supervised Representation Learning. ArXiv, abs/2410.09156.

[7] Yuan, Z., Zhu, D., Qiu, Z., Li, G., Wang, X., & Yang, T. (2023). LibAUC: A Deep Learning Library for X-Risk Optimization. Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining.

[8] Qiu, Z., Hu, Q., Yuan, Z., Zhou, D., Zhang, L., & Yang, T. (2023). Not All Semantics are Created Equal: Contrastive Self-supervised Learning with Automatic Temperature Individualization. ArXiv, abs/2305.11965.

