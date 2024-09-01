# Introduction
Prompt learning in pretrained visual-language models has shown remarkable flexibility across various downstream tasks. Leveraging its inherent lightweight nature, recent research attempted to integrate the powerful pretrained models into federated learning frameworks to simultaneously reduce communication costs and promote local training on insufficient data. Despite these efforts, current federated prompt learning methods lack specialized designs to systematically address severe data heterogeneities, e.g., data distribution with both label and feature shifts involved. 
To address this challenge, we present Federated Prompts Cooperation via Optimal Transport (FedOTP), which introduces efficient collaborative prompt learning strategies to capture diverse category traits on a per-client basis. Specifically, for each client, we learn a global prompt to extract consensus knowledge among clients, and a local prompt to capture client-specific category characteristics. Unbalanced Optimal Transport is then employed to align local visual features with these prompts, striking a balance between global consensus and local personalization. 
Extensive experiments on datasets with various types of heterogeneities have demonstrated that our FedOTP outperforms the state-of-the-art methods.


## How to Run

You can run `federated_main.py` with some specified arguments.

### Training

`--root` takes as input a path to dataset, like `caltech101` or `oxford_flowers`.

`--config-file` means which config file to use, such as `rn50` or `vit_b16`.

You can select variables like shots, users by changing `cfg` or you can change every arguments you like in `plot_few_shot.sh`.

### For example
**FedOTP (M=16, end)**:
If you want to train caltech101 with 2 shots, backbone rn50 and total independent non-iid setting.
You can specify that:
`MODEL=FedOTP`
`TRAINER=PLOT`
`OT=COT`
`DATA=caltech101`
`SHOTS=2`
and run `bash plot_few_shot.sh`

After the experiments, all the results are finished and save to `output/`.
We build and modify the code based on Dassl and CoOp.
We will release the full-version and detailed description later.


