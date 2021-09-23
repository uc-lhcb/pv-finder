## Overview of what is in the "baileyds_stuff" folder

- **baileyds**: symbolic link to /share/lazy/baileyds, where I've been storing all of my runs

- **dataAA**: symbolic link to /share/lazy/sokoloff/ML-data_AA, to get training data

- **DOC_code**: directory of some code that I used over the summer studying optimizers, and I tried 
             to include the notebooks responsible for all figures shown in the documentation.
             
    - **AllCNN**: full of notebooks using an AllCNN model, the most important of which is 
                AllCNN_runs_compared.ipynb, which compares various previous runs
                
    - **DDplus**: full of notebooks using DDplus, which are honestly not very conveniently labeled.
                For an overview, look at various_DDplus_runs_compared.ipynb to see a bunch of runs
                compared, and Sep14_DDplus_runs_compared.ipynb, which compares runs all done from
                the same initial state and with the same dataset.
    
    - **functions**: directory of python function scripts, which are incredibly messy, but I am
                   keeping it so you can look at what functions were run to get old results.
                   
    - **lin_reg**: full of notebooks of a model that solves a linear regression problem, as well as
                 some functions to generate data. 
                 
    - **MNIST**: full of notebooks of a model classifying the MNIST image dataset.
    
    - **model**: symbolic link to the pv-finder/model directory.
    
- **functions**: directory of edited functions I have made over the summer.
    
    - **common_optimizers**: file of common optimizers I wrote as pytorch optimizers, not that useful 
                           except for learning how pytorch optimizers are structured. Also contains 
                           some learning rate schedulers.
                           
    - **new_optimizers**: file of the new optimizers I created, which are modified versions of Adam and 
                        epoch optimizers which function each epoch, instead of each minibatch.
                        
    - **new_training_kde**: file of modified training utility functions, to use the new optimizers
                          and learning rate algorithms developed.
                          
    - **plotting_DDplus_hists**: file of functions to plot the histograms predicted by a DDplus model
                               state dictionary.
                               
    - **utils**: utility functions for pytorch models.
    
- **model**: symbolic link to pv-finder/model directory.
 
- **AllCNN_example.ipynb**: notebook of an example of training an AllCNN model using the new methods
                          developed, with detailed explanations of each part of the notebook.
                          
- **DDplus_example.ipynb**: notebook of an example of training the DDplus model using the new methods
                          developed, with detailed explanations of each part of the notebook.
                       
- **documentation.docx**: a word doc that is the detailed documentation of my methods and results.

Feel free to contact me at baileyds@oregonstate.edu if you have any questions.                    