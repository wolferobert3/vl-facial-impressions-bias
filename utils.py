import torch
import pandas as pd
import numpy as np

from os import listdir
from scipy.stats import spearmanr, ttest_rel
from typing import Tuple, Iterable
from tqdm import tqdm

def create_attribute_dict(file_path: str) -> dict:
    """
    Take a file of attributes and return a dictionary mapping each attribute to its prompts.
    """

    # Read in the file and split it into a list of lines each associated with a single attribute
    attribute_list = [i.strip() for i in open(file_path, 'r').read().split('\n') if i not in ['', ' ', '\n']]
    
    # Return a dictionary mapping each attribute to its prompt
    return {i.split(':')[0]: i.split(':')[1] for i in attribute_list}

def create_model_association_df(model_similarity_df: pd.DataFrame, attribute_dict: dict, opposite_dict: dict=None, baseline: str='difference') -> pd.DataFrame:
    """
    Creates a dataframe of the difference between the cosine similarity of each image to the positive prompt and the negative prompt for each attribute in a model.
    """
    if baseline == 'difference':
        return pd.DataFrame({attribute: model_similarity_df[attribute_dict[attribute]] - model_similarity_df[opposite_dict[attribute]] for attribute in attribute_dict.keys()})
    elif baseline == 'positive':
        return pd.DataFrame({attribute: model_similarity_df[attribute_dict[attribute]] for attribute in attribute_dict.keys()})
    elif baseline == 'someone':
        return pd.DataFrame({attribute: model_similarity_df[attribute_dict[attribute]] - model_similarity_df['a photo of someone'] for attribute in attribute_dict.keys()})

def create_model_human_similarity_dict(model_association_df: pd.DataFrame, attribute_dict: dict, omi_ratings: dict) -> Tuple[dict, dict]:
    """
    Creates a dictionaries of the Spearman's r correlation coefficient and significances between the model association and the OMI rating for each attribute in a model.
    """
    attribute_correlations, attribute_sigs = {}, {}
    
    # Iterate through each attribute
    for attribute in attribute_dict.keys():

        # Obtain Spearman's r between the model association and the OMI rating for each attribute
        corr = spearmanr(omi_ratings[attribute], model_association_df[attribute])

        # Store the correlation coefficient in a dictionary
        attribute_correlations[attribute] = corr[0]

        # If the correlation is significant and positive, set the attribute to 1, otherwise set it to 0
        if corr[1] < 0.05 and corr[0] > 0:
            attribute_sigs[attribute] = 1
        else:
            attribute_sigs[attribute] = 0
    
    return attribute_correlations, attribute_sigs

def get_model_names(path: str, family: str='scaling') -> list:
    """
    Returns a list of model names from a directory of model similarity files.
    """
    return [i.split('_first_impression_similarities.pkl')[0] for i in listdir(path) if i.split('.')[-1] == 'pkl' and i.split('_')[0] == family]

def compute_cosine_similarity(tensor_1: torch.tensor, tensor_2: torch.tensor):
  """
  Computes the cosine similarity between two tensors.
  """

  return torch.dot(tensor_1.squeeze(), tensor_2.squeeze()) / (torch.norm(tensor_1) * torch.norm(tensor_2)).item()

def compute_normalized_frobenius_product(A: np.ndarray, B: np.ndarray) -> float:
    """
    Computes the normalized Frobenius inner product between two matrices.
    """

    return np.inner(A.flatten(), B.flatten()) / (np.linalg.norm(A, ord='fro') * np.linalg.norm(B, ord='fro'))

def cohens_d(x: Iterable, y: Iterable) -> float:
    """Compute Cohen's d for two samples."""

    return (np.mean(x) - np.mean(y)) / np.std(np.concatenate([x, y]), ddof=1)

def paired_ttest(x: Iterable, y: Iterable) -> float:
    """Compute paired t-test for two samples."""

    return ttest_rel(x, y)

def add_significance_stars(d: float, p: float) -> str:
    """Add significance stars to Cohen's d."""
    if p < 0.05:
        return f"{d:.2f}*"
    else:
        return f"{d:.2f}"

def round_and_format(x: float) -> str:
    """Round and format a float, adding gray cell shading."""

    rounded = f"{x:.2f}"

    if rounded.startswith("-"):
        if rounded[1] == "0":
            return '\\gray{0}' + '-' + rounded[2:]
        else:
            return '\\gray{0}' + rounded

    elif rounded.startswith("0"):
        return '\\gray{' + rounded[-2:] + '}' + rounded[1:]
    
    return '\\gray{' + rounded[-2:] + '}' + rounded

def round_and_format_cohens_d(d: float, p: float) -> str:
    """Round and format Cohen's d, adding gray cell shading."""

    tex_string = '\\gray{0}'

    if d > 0:
        tex_string = '\\gray{' + f'{min(d*100, 100):.0f}' + '}'
    
    rounded = f'{d:.2f}'

    if rounded.startswith("-") and rounded[1] == "0":
        tex_string = tex_string + '-' + rounded[2:]

    elif rounded.startswith("0"):
        tex_string = tex_string + rounded[1:]

    else:
        tex_string = tex_string + rounded

    if p < 0.05:
        tex_string = tex_string + '*'
    
    return tex_string