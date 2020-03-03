# Audio Models Stattus4
Audio Models and statistical tools used in Devs for Stattus4 Start-Up of Water Wasting Detection

# Data

## Children

The children data up-to-now (04-12-2019), comprises of the LANNA (AI Labs from pals in Northeast Europe) data for children with specific speech impairement and my own recordings from children in circus stages having fun. I plan to do all the type of bit science and reverberation with it, for further explanations contact me.

## Birds

Birds down inside the flowers is a beautiful scenery. #TODO

# Data Historical Disclamer

This API is suited for any audio modelling. It was produced under the hood of Stattus4 and FAPESP enterprise for water tubulation sounds, and will be turned into Beta version as soon as possible for the author and possible contribuitors. 

Note: this is a Alpha Version.

## Audio Deep Learning

Deep Neural Networks for Tagging Audio based on binary tagged database using Audio Features Models.
This Neural Network is being implemented altogether with Neural Network API, Keras like, but focused on Input Signal Framing and Semantization and Input Based Neural Architecture Search. The two projects will hold it's own repository in the future.

0. Milestones
  * 02/09/2019
    * Classifier for any input image size (i.e. flattening last Features from Hidden Layers). **Done**
    * Model for diverse FRAME size as Input. **Done**
    * Model for Hyperparam Tuning (parent class for hyperparameter search of any specific architecure). **Done**
    
  * 12/09/2019    
    * Hyperparameter para GCNN + MaxPooling de acordo com o tamanho do FRAME de entrada. **Done**
    * Corrected Bugs from previous commit, pyramidal Hyperparameter automatic has a beta (17/09/2019) **Done**
    * Builder for final model. **Doing**
    
  * 08/12/2019
    * Builder for stacked layers (i.e. deepness for same design arch).

  * Future 
    
    * Implementation of Bi-LSTM(RNN) module, GAN modules such Sequentially Framed GANs. Other architecture modules
    * Implementation of another Neural Architecture Search Algorithms such Monte Carlo Search Trees and GAs Metaheuristic class of algorithms.
    
  * Previous Milestones
  
    * Implementing GCNN Maxpooling Architecture Units. **Done**
    * Hand choosing parameters (Convolution filter size, layer numbers, batch size). **Done**
    * Model Overfit for small amout of data. **Done**

## Installation

Accepts Python 3.6.9

See the requiriment file to do pip install, we advise to install the dependencies from conda channel before running pip install.

```bash
git clone https://github.com/Uiuran/stattus4-audio-models.git
cd ~/path/to/dir/stattus4-audio-models/
conda create --name audio python=3.6
conda activate audio
pip install .
``` 

## Usage 
```python
import audiomodels.model as audio
``` 

## TODO

* Port future Beta version to TF 2.0. 

## Acknowledged backlogs (or archived, for the moment priority, TODOs)

These were developments predicted but dumped for Neural Network development.

* Enrich Statistical Analysis.
* Filtering
  * Spectral
  * Inverse Filtering
* Big-Data and cluster processing of models (such using Spark and serving TF services).
