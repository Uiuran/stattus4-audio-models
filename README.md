# Audio Models Stattus4
Audio Models and statistical tools used in Devs for Stattus4 Start-Up of Water Wasting Detection


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
    * Builder for final model. **Doing**
    
  * What is Next ?
    * Cross-Validation with Spectrogram Data, ~ 400x600, of the order of ~ 100000 units.
    * Calculus of accuracy and fn parameters.
    
  * Previous Milestones
  
    * Implementing GCNN Maxpooling Architecture Units. **Done**
    * Hand choosing parameters (Convolution filter size, layer numbers, batch size). **Done**
    * Model Overfit for small amout of data. **Done**

## Installation

Nowadays requires python 2.7

```bash
git clone https://github.com/Uiuran/stattus4-audio-models.git
cd ~/path/to/dir/stattus4-audio-models/
conda create --name audio python=2.7
conda activate audio
pip install .
``` 

Bitbucket repository to be synchronized ...

## Usage 
```python
import audiomodels.model as audio
``` 

## Must TODO

* Port to python 3

## Possible TODOs 

These were developments predicted but dumped for Neural Network development.

* Statistical Analysis
* Filtering
  * Spectral
  * Inverse Filtering
