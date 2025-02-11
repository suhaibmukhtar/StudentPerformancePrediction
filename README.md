## End to End Machine learning Project
#### -e . (This automatically triggers the setup.py file and builds the package.)[When we run pip install -r requirements.txt]
#### requirements list in the setup.py contains the list of libraries, it should not contain the -e ., so we have to remove it when installing libraries.

##### create a folder inside the src named components which contains files such as data_injestion.py, data_transformation.py, model_trainer.py.
This folder will contain all the modules that we are going to use for building a project. like data_injestion.py, data_transformation.py etc.
#### Pipeline will contain all the code related to the pipeline: training and prediction pipeline
<li>Training_pipeline will contain code related to training the models</li>
<li>In this training_pipeline,we will import all the components needed for training a model(all components will be bought here)</li>
<li>predict_pipeline, in this file we will import all the components needed for performing prediction</li>
<li>logger.py: will be responsible for creating and storing logs.</li>
<li>exception.py: will be responsible for tracking and handling exceptions.</li>
<li>utils.py: will contain functionalities that are common among all the components/modules</li>