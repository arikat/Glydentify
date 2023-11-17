# Glydentify: A Deep Learning Tool for Glycosyltransferase Classification

## Overview

Glydentify is an open-source tool leveraging the power of protein language models for the classification of glycosyltransferases (GTs). This application, grounded in the cutting-edge ESM2 protein language model, offers a novel approach to classifying GTs into fold A families. Glydentify not only achieves high accuracy in prediction but also provides insights into donor binding preferences and key contributing residues, adding an explainable component to its functionality.

## Features

- **High Accuracy Classification**: Classifies GTs into fold A families with 92% accuracy and predicts GT-A donor binding preferences with 88% accuracy.
- **Explainable AI**: Identifies key residues contributing to predictions, offering insights into the underlying mechanisms of enzyme function.
- **User-Friendly Interface**: Built with Gradio, Glydentify offers an intuitive interface requiring no programming experience, making it accessible for a wide range of users.
- **Web-Based Access**: Available directly through any web browser via [Hugging Face Spaces](https://huggingface.co/spaces/arikat/Glydentify).

## Components

- **Training Script**: The heart of Glydentify lies in the deep learning model training, contained in `family_prediction.ipynb`. This Jupyter Notebook includes detailed steps for model training, validation, and evaluation, utilizing PyTorch and the ESM2 model.
- **Application Folder**: The `App` folder hosts a Python script responsible for loading the trained model and labels, and constructing the Gradio-based web application for easy interaction and function prediction.
- **Accessible Deployment**: Hosted on Hugging Face, the application offers a seamless experience for end-users wanting to explore GT classification without any setup requirements.

## How to Use

1. **Model Training**: To train your own model or understand the training process, explore `family_prediction.ipynb`. This notebook requires a fundamental understanding of Python and deep learning.
2. **Running the Application**: For end-users, simply visit the Glydentify space on Hugging Face. The web interface allows you to input protein sequences and receive function predictions without any coding requirement.
3. **Local Deployment**: Advanced users can set up Glydentify locally by cloning the repository, setting up the environment, and running the application script within the `App` folder.

## Requirements

- Python 3.9+
- PyTorch
- Transformers Library
- Gradio

## Installation

Clone the repository:

```bash
git clone https://github.com/arikat/Glydentify.git
cd Glydentify
```

Install the required packages from the environment file (requires conda/mamba):

```bash
conda env create -f glydentify.yml
```

## Contributing

Contributions to Glydentify are welcome! Whether it's feature enhancements, bug fixes, or documentation improvements, feel free to fork the repository and submit a pull request.

## Citation

If you use Glydentify in your research, please cite our accompanying paper:

```bibtex
@article{
  title={Glydentify, a deep learning tool for classifying glycosyltransferase function},
  author={Venkat, A., Zhou, Z., Gill, S. and Kannan, N.},
  journal={manscript in preparation},
  year={2024}}
```

## License

Glydentify is released under the GNU General Public License v3.0 License. See the LICENSE file in the repository for details.
