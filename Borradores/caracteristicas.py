import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from   skimage.measure import label
from   pyxvis.processing.segmentation import seg_bimodal
from   pyxvis.features.extraction import extract_features
from   pyxvis.io.plots import plot_ellipses_image
from   sklearn.metrics import confusion_matrix, accuracy_score