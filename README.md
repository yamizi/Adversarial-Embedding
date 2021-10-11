# Adversarial Embedding


This is the code release for our ppaper "Evasion Attack STeganography: Turning Vulnerability Of Machine Learning To Adversarial Attacks Into A Real-world Application"
Published in ICCV - AROW, 2021

## Abstract
Evasion Attacks have been commonly seen as a weakness of Deep Neural Networks. In this paper, we flip the paradigm and envision this vulnerability as a useful application. We propose EAST, a new steganography and watermarking technique based on multi-label targeted evasion attacks. The key idea of EAST is to encode data as the labels of the image that the evasion attacks produce. Our results confirm that our embedding is elusive; it not only passes unnoticed by humans, steganalysis methods , and machine-learning detectors. In addition, our embedding is resilient to soft and aggressive image tampering (87% recovery rate under jpeg compression). EAST out-performs existing deep-learning-based steganography approaches with images that are 70% denser and 73% more robust and supports multiple datasets and architectures.

Recommended citation: Ghamizi, S., Cordy, M., Papadakis, M., & Traon, Y.L. (2021). Evasion Attack STeganography: Turning Vulnerability Of Machine Learning To Adversarial Attacks Into A Real-world Application. Proceedings / IEEE International Conference on Computer Vision. IEEE International Conference on Computer Vision. 

This project is now a Pytorch project. If you want the previous version (with SATA algorithm), please check the branch "master"
The "EAST" branch contains the exact version of the code of the ICCV - AROW paper
The "main" branch will contain the new code and updates of our Steganography package.


## Prerequisite
### Python
The code should be run using python 3.8, torch==1.7.1. We recommend using conda for a dedicated environment

### Requirements installation

You can use either conda or pip to install the project requirements.

```bash
sudo pip install -r ./requirements.txt
```
