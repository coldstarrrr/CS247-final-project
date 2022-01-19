# CS247_final_project
Final Project For CS247 - Neural Collaborative Filtering

Contributors:

WEI ZHOU       weiz@g.ucla.edu  
KEVAN LOO      kevanloo@ucla.edu  
YUNTIAN WANG   ytwlu180@g.ucla.edu  
TONG WU        wutong0218@g.ucla.edu  

The dataset is too big to upload, but it can be downloaded here: https://www.kaggle.com/rdoume/beerreviews


#### Models folder:
This folder contains the models (ipynb and py files) of the project. 
- EDA_Preprocessing.ipynb
this file contains the data EDA and preprocessing, this file must be run firstly in order to split train/val/test data, and store them in **DATA** folder
- MLP Embedding model V2
this file contains the MLP model, running this file will generate the model and store it into local **checkpoints** folder
- MatrixFactorization.ipynb
this file contains both MF and GMF model, running this file will generate the models and store them into local **checkpoints** folder
- NeuMF V2.ipynb
this file contains NeuMF model with α=0.5, running this file will generate the model and store it into local **checkpoints** folder
- NeuMF - RMSProp.ipynb
this file contains NeuMF model with RMSProp optimizor. This model is the best NeuMF model and the model is stored in **checkpoints** folder
#### Checkpoints
This folder contains the best model of each recommendation system (MF, GMF, MLP, Neu_MF)
