# Adversarial Embedding

This is the code release for our IEEE Symposium on Security and Privacy 2020 submitted paper entitled Adversarial Embedding: A robust and elusive Steganography and Watermarking technique.
A pre-print can be requested on our repository: https://orbilu.uni.lu/handle/10993/40970 


## Abstract of the paper
We propose adversarial embedding, a new steganography and watermarking technique that embeds secret information within images. The key idea of our method is to use deep neural networks for image classification and adversarial attacks to embed secret information within images. Thus, we use the attacks to embed an encoding of the message within images and the related deep neural network  outputs to extract it. The key properties of adversarial attacks (invisible perturbations, nontransferability, resilience to tampering) offer guarantees regarding the confidentiality and the integrity of the hidden messages.
We empirically evaluate adversarial embedding using more than 100 models and 1,000 messages. Our results confirm that our
embedding passes unnoticed by both humans and steganalysis methods, while at the same time impedes illicit retrieval of the
message (less than 13% recovery rate when the interceptor has some knowledge about our model), and is resilient to soft and (to
some extent) aggressive image tampering (up to 100% recovery rate under jpeg compression). We further develop our method
by proposing a new type of adversarial attack which improves the embedding density (amount of hidden information) of our
method to up to 10 bits per pixel.


*The SATA attack and all the experiments are tested on Ubuntu 16.04 (64-bit) and Windows 10 (64-bits), but should be compatible with other major Linux/MacOS distros and versions. Feel free to contact [Salah Ghamizi](https://wwwen.uni.lu/snt/people/salah_ghamizi) if you run into any problem running or building our tools or running the experiments.*

The experiments of the paper are available in the dedicated folder */experiments/*

The SATA attack we propose in the papers is available in the file */utils/sorted_attack.py*


## Prerequisite
### Python
The code should be run using python 3.5, Keras & Tensorflow frameworks. We recommend using conda for a dedicated environment

### Requirements installation

You can use either conda or pip to install the project requirements.

```bash
sudo pip install -r ./requirements.txt
```

This project uses a customized version of [Aletheia Library](aletheia/README.md), [Perceptual Similarity Librarie](lpips-tensorflow/README.md) and [SSIM Similarity Librarie](pyssim/README.md). You should refer their installation instructions (in the previous links) for guidance on how to set them up (for instance Aletheia requires Octave).

if your machine does not support gpu, replace *tensorflow-gpu* in the requirements file by the CPU version.

Not however that the experiments that have been run on 100 pre-sampled models require to sample and generate these models using [FeatureNet](https://github.com/yamizi/FeatureNet) 

FeatureNet is a tool that allows to generate a DNN neural networks under the constraint of diversity.