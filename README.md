# Facial Impression Bias in Language-Vision AI Repository
This is the code repository for the AIES'24 paper ***Dataset Scale and Societal Consistency Mediate Facial Impression Bias in Vision-Language AI***, available at [https://arxiv.org/pdf/2408.01959](https://arxiv.org/pdf/2408.01959).

### 1. Structure

This project required two primary workflows: one to obtain data from AI models, and a second to analyze that data. The first workflow utilizes the files prefixed with "embedding_", which embed the OMI images and associated prompts using various CLIP models, and the files in the "text_to_image" directory, which generate images from three popular Stable Diffusion models, and then embed them for analysis using a Vision Transformer. Note that the "float_to_tensor" notebook converts tensor objects to floats to make the data easier to work with in Pandas.

The analysis workflow uses the files suffixed with "_analysis" to study the presence and structure of facial impression bias in CLIP models and the impact of scale-related variables on that bias. The "tti_f1_bias" notebook analyzes facial impression biases in text-to-image ("TTI") models, as described in the paper.

Note that the files beginning with "validation" employ the data collection and analysis described in our appendix to validate the methods we used with CLIP models.

### 2. Requirements

The requirements file includes the libraries needed to run the analyses. We recommend creating a unique environment for running the project (for example, with conda, `conda create -n "facialImpressionBias" python=3.11`) and then installing the requirements (`pip install -r requirements.txt`). Note that some of the notebooks are intended to be run on Google Colab, especially those using a GPU to run CLIP and Stable Diffusion models; we've left in the code to do that.

### 3. Paper & Citation

Below follows the information to cite our paper:

```bibtex
@article{wolfe2024dataset,
  title={Dataset Scale and Societal Consistency Mediate Facial Impression Bias in Vision-Language AI},
  author={Wolfe, Robert and Dangol, Aayushi and Hiniker, Alexis and Howe, Bill},
  journal={arXiv preprint arXiv:2408.01959},
  year={2024}
}
```

### 4. Other Resources

This work draws on prior research in vision-language AI, facial impression bias, and AI bias more broadly. Below are a few essential resources for understanding the context of this research:

- [https://github.com/jcpeterson/omi](**https://github.com/jcpeterson/omi**): The One Million Impressions (OMI) dataset of Peterson et al., a set of images and human ratings which serves as the primary source of data for this work.
- [https://www.pnas.org/doi/full/10.1073/pnas.2115228119](**https://www.pnas.org/doi/full/10.1073/pnas.2115228119**): The paper describing Peterson et al.'s findings with the OMI dataset.
- [https://github.com/openai/CLIP](**https://github.com/openai/CLIP**): The github repository for OpenAI's landmark CLIP model, which enabled the study of associations between text and images in an unsupervised model.
- [https://github.com/LAION-AI/scaling-laws-openclip](**https://github.com/LAION-AI/scaling-laws-openclip**): The repository of OpenCLIP models trained explicitly to study scaling laws (relating, for example, parameter count or dataset scale to model performance) in vision-language models.
- [https://github.com/FacePerceiver/FaRL](**https://github.com/FacePerceiver/FaRL**): The repository of FaRL/FaceCLIP models post-trained for facial analysis tasks.
- [https://github.com/SepehrDehdashtian/the-dark-side-of-dataset-scaling](**https://github.com/SepehrDehdashtian/the-dark-side-of-dataset-scaling**): Essential work from ACM FAccT'24 on biases emerging from increasing dataset scale.
- [https://dl.acm.org/doi/10.1145/3442188.3445922](**https://dl.acm.org/doi/10.1145/3442188.3445922**): The ahead-of-its-time Stochastic Parrots paper, which asked important questions about scale in AI that continue to inform analyses of bias in these models today.