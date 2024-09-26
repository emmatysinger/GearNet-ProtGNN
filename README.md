# GearNet-ProtGNN: Protein function prediction model from protein structure and biomedical knowledge

This repository hosts the official implementation of GearNet-ProtGNN, a model for predicting protein function. 

### **Path**
- Must change line 13 in `os.environ['WAND_EXECUTABLE']` to local path to python in `training_downstream.py`


### **Training**

`training_gearnetprotgnn.py` is the main script to train GearNet-ProtGNN. Parameters to change:
- `num_epoch`: Number of epochs to train model for
- `hyperparameter`: True or False (run a hyperparameter sweep or not)
- `embed_file`: path to pickle file of the protein embeddings from ProtGNN
The script can be run from the command line or in a bash script like this: 
```
python training_gearnetprotgnn.py
```

### **Downstream Prediction Tasks**

`training_downstream.py` is the main script to run downstream prediction tasks. Parameters to change:
- `model_path`: Change to path where your model is stored
- `dataset_type`: 'GO' or 'EC'
- `branch`: 'MF', 'BP' or 'CC' if dataset_type is 'GO'
- `freeze`: True or False (freezing GCN layer weights)

The script can be run from the command line or in a bash script like this: 
```
python training_downstream.py
```

### **Embedding Space Visualization**

There are two notebooks that demonstrate how to visualize the predicted embedding space. 
- `visualize_predicted.ipynb`: Colors predicted embedding space by molecular function and biological process
- `visualize_by_structure.ipynb`: Colors predicted embedding space by CATH superfamilies



