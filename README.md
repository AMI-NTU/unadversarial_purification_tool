## Unadversarial Purification：A Defense Tool for Pre-trained Classifiers

`unadversarial-purification` is a PyTorch library for improving adversarial robustness of pre-trained classifiers.
**NOTE: this library is in early development and we plan to make it public before 2022-4-30.**

Deep neural networks are known to be vulnerable to adversarial attacks, where a small perturbation leads to the misclassification of a given input. Here, we aim to propose a defense tool called Unadversarial Purification to improve adversarial robustness of pre-trained classifiers. This method allows public vision API providers and users to seamlessly convert non-robust pre-trained models into robust ones and it can be applied to both the white-box and the black-box settings of the pre-trained classifier. Most of the existing defenses require that the classifier be trained (from scratch) specifically to optimize the robust performance criterion, which makes the process of building robust classifiers computationally expensive. Therefore, we consider providing a defense tool that robustifies the given pre-trained model without any additional training or fine-tuning of the pre-trained model. In this tool, we design to prepend a custom-trained denoiser before the pre-trained classifier, termed as purification layer. The purification layer is agnostic to the architecture and training algorithm of the pre-trained model, hence can be easily incorporated into well-known public APIs.  

![](https://github.com/AMI-NTU/unadversarial_purification_tool/blob/main/tools.png?raw=true)
