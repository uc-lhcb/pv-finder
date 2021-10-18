# 1. **PV-Finder Manual**

**Michael Peters**  
*University of Cincinnati*  

**Last Updated:** October 2021  

## 1.1. **Table of Contents**

- [1. **PV-Finder Manual**](#1-pv-finder-manual)
  - [1.1. **Table of Contents**](#11-table-of-contents)
  - [1.2. **Introduction**](#12-introduction)
  - [1.3. **PV-Finder**](#13-pv-finder)
    - [1.3.1. **Purpose**](#131-purpose)
    - [1.3.2. **Models**](#132-models)
      - [1.3.2.1. **SimpleCNN**](#1321-simplecnn)
      - [1.3.2.2. **AllCNN**](#1322-allcnn)
  - [1.4. **Generating Histograms**](#14-generating-histograms)
    - [1.4.1. **Overview**](#141-overview)
    - [1.4.2. **What is a KDE?**](#142-what-is-a-kde)
    - [1.4.3. **Different KDEs**](#143-different-kdes)
  - [1.5. **Git & GitHub**](#15-git--github)
    - [1.5.1. **Basic Commands**](#151-basic-commands)
    - [1.5.2. **Resources**](#152-resources)
  - [1.6. **Linux**](#16-linux)
    - [1.6.1. **Setup**](#161-setup)
    - [1.6.2. **Basic Commands**](#162-basic-commands)
    - [1.6.3. **Connecting to Server**](#163-connecting-to-server)
  - [1.7. **Model**](#17-model)
    - [1.7.1. **Architectures**](#171-architectures)
      - [1.7.1.1. **AllCNN**](#1711-allcnn)
      - [1.7.1.2. **Perturbative AllCNN**](#1712-perturbative-allcnn)
      - [1.7.1.3. **UNet**](#1713-unet)
      - [1.7.1.4. **Other Architectures**](#1714-other-architectures)
    - [1.7.2. **Using a Model**](#172-using-a-model)
    - [1.7.3. **Creating a Model**](#173-creating-a-model)
  - [1.8. **MLFlow**](#18-mlflow)
    - [1.8.1. **Setup**](#181-setup)
    - [1.8.2. **Navigating MLFlow**](#182-navigating-mlflow)
  - [1.9. **References**](#19-references)

## 1.2. **Introduction**

The purpose of this manual is to give a simple and complex overview of pv-finder. This manual will cover many things. For starters, it will explain what pv-finder is, how the process goes from LHCb collision data to learned primary vertices (PVs), some of the physics behind the process, and why this is all important. It will also explain more general concepts, like some of the basics of Linux/UNIX commands, how to use MLFlow, and how to use Git and GitHub. Finally, a sizable will occur of the machine learning that occurs in pv-finder. This will include a thorough walk-through of how to create and implement a neural network architecture into pv-finder, as well as an analysis of existing architectures.

## 1.3. **PV-Finder**

This section will discuss the purpose, use, and current model architecture of PV-Finder.

### 1.3.1. **Purpose**

The LHCb detector is facing major upgrades in Run 3 in 2021, which includes the hardware level-0 trigger being removed in favor of a purely software trigger. PV-finder is intended to be a method for finding vertices from tracks or hits using machine learning techniques, which would function as this purely software trigger. Currently, the algorithm starts with tracks containing location, direction, and covariance matrix information. The tracks in 3D space are then projected in a 1D binned "kernel", which most closely relates to the z-axis projections of the primary vertices. A 1D machine learning algorithm is then used to predict the PV location from the generated kernel.

This process of creating a kernel estimate of the z-axis projections is known as the Kernel Density Estimator (KDE). There are many different KDEs that have been created, which is beyond the scope of this manual to thoroughly list and explain. It is suffice to say that only one or two KDEs are ever used at once as part of the input for a single machine learning run. More information can be found in the previous papers written on the topic from 2019 and 2021.

### 1.3.2. **Models**

There are a few models being used to find PVs. The default model architecture used as a comparison to all others in pv-finder is the AllCNN architecture, which is a modified version of the SimpleCNN architecture. Alternative models that have been tested include the UNet and modified AllCNN architectures. For now, it is best to focus on the AllCNN architecture.

#### 1.3.2.1. **SimpleCNN**

To explain AllCNN, it is important to understand what the SimpleCNN architecture is. For the purposes of discussion, it will be assumed that this is a 6 layer neural network. In this case, SimpleCNN is a series of 5 convolutional layers with LeakyReLU activation functions, as well as a fully connected layer at the end, where it passes through a Softplus activation function. Dropout is also employed in each layer. Furthermore, the input dimension is preserved at each layer (1 x 4000), which is accomplished with a stride of one and padding in the convolutional layers. This architecture can also be seen in Figure 1.

| ![alt text](/figs/simplecnn-arch.bmp "Figure 1") |
|:--:|
| *Figure 1. This is a sample architecture of SimpleCNN. Different channel and kernel sizes are used throughout the model, which have been decided largely out of trial and error, as well as some (currently) unprincipled estimates.* |

#### 1.3.2.2. **AllCNN**

AllCNN is similar to SimpleCNN. As the name suggests, the difference is that AllCNNdoes not use a fully-connected layer at the end. Instead, there are only convolutional layersused throughout the network. This typically does not work on its own, however, and musthave trained weights loaded into all but the last, newly added, convolutional layer.  It isnot currently understood why this is the case, but the network seems incapable of learningand locating PVs without pre-training weights. This is accomplished by first training usingSimpleCNN, then loading all but its last layer’s weights into an AllCNN model.  Experi-mentally, this works and has consistently produced better results than by just running theSimpleCNN model. Below is a diagram of the AllCNN architecture, seen in Figure 2.

| ![alt text](/figs/allcnn-arch.bmp "Figure 2") |
|:--:|
| *Figure 2. This is a sample architecture of AllCNN. As in Figure 1, different channel and kernel sizes are used throughout the model, likewise chosen in the same fashion.* |

## 1.4. **Generating Histograms**

***NOTE: This section is still a work in progress and may not be entirely accurate yet.***

The process of locating PVs after an experiment has been run begins with preparing the data collected. In order to perform machine learning, data needs to be labeled and formatted such that it can be interpreted in a meaningful way. In the case of PV-Finder, it involves:

1. Hits are formed into linear tracks.
2. Determining the point of closest approach of each track with respect to the beamline in x, y, z space.
3. Calculate the variance of each point of closest approach in x, y, z space.
4. Plot point of closest approach in x, y space and z space separately, including the variances in each axis.
5. Sum z tracks together (sum of point of closest approach curves from each track).
6. Generate a Kernel Density Estimator from the point of closest approach sums along the z-axis.

A more in-depth discussion of this process will follow.

### 1.4.1. **Overview**

| ![alt text](/figs/lhcb_diagram.bmp "Figure 3") |
|:--:|
| *Figure 3. This is a diagram of the LHCb detector located at CERN. The z direction used in the following discussion is defined as horizontal with respect to this image.* |

Figure 3 above provides a high-level idea of what is happening at the LHCb collision site. In one direction past the vertex location (collision site), a series of plates and sensors detect what is produced from the collision and logs it as a series of x, y, and z coordinates. This can be better thought of as "voxels" for reasons that will be explained shortly.

All voxels are normalized with respect to the beam's position, which can be thought of as pointing in the direction (0, 1, 1).

If a series of readings are collected that are nearly linear (as in Figure 4 below), the markers are substituted with a straight line parameterized by a few values; **this** is a track.

| ![alt text](/figs/tracks.bmp "Figure 4") |
|:--:|
| *Figure 4. Here are a series of readings along the z-axis. This comes from a [presentation](https://indico.cern.ch/event/759388/contributions/3303404/attachments/1814784/2965597/2019_HOW_PvFinder.pdf) on pv-finder given in 2019.* |

The point of closest approach (POCA) is then measured between each track line and the beamline. These points are not perfectly accurate; thus, they are generated using gaussian distributions in x, y, and z space. In other words, each coordinate determined by the sensors is, in reality, more like a small space in x, y, and z. This is why it is referred to as a voxel rather that a point.

Once the POCAs are determined, their uncertainty is included; this forms ellipses around the POCAs in x, y, z space. If, for some reason, we were completely certain about the x, y, and z coordinates of the POCAs, the resultant KDE would be zero almost everywhere with only a few spikes where PVs are located.

Instead, the most variance in position occurs in the z direction, which can range to several meters. The x/y direction, on the other hand, has much less variance -- usually +/- 20 mm.

This is typically why the data is divided into two representations -- the z uncertainty and the x, y uncertainty.

| ![alt text](/figs/kde_plots.bmp "Figure 5") |
|:--:|
| *Figure 5. This is a set of plots that represent the data collected from LHCb.* |

In Figure 5, the bottom plot represents the x,y POCA ellipsoids, which form the uncertainty in x, y space. The z POCAs and their uncertainties for each track are shown in the second plot down (with the purple curves). Each curve represents one track and are stacked on top of one anohter. The top plot, then, is simply the sum of each of these tracks along the z-axis. This is the POCA sum.

### 1.4.2. **What is a KDE?**

### 1.4.3. **Different KDEs**

Each KDE is represented and formed in different ways. There are certainly many ways a KDE could be formed, but included here are the ones currently in-use today.

**KDE-A**: KDE-A uses the purple plot from Figure 5; that is, it takes the distributions along the z dimension stacked up as its input for generating the KDE.

**KDE-B**: KDE-B uses the poca-sum along the z dimension (shown in the top plot of Figure 5) as its input for generating the KDE.

**POCA-KDE**: The POCA-KDE uses KDE-A, KDE-B, and the x,y poca ellipsoids as its input for generating the KDE.

## 1.5. **Git & GitHub**

There are many videos, tutorials, discussions, and blogs about what Git and GitHub are, as well as their uses. Rather than reinvent the wheel, this section will present some useful/common commands, as well as some useful resources to read/watch to help understand how to use Git/GitHub. These commands can be used when connected to a Linux server (in a terminal, usually).

### 1.5.1. **Basic Commands**

There are a few basic commands needed to operate Git in a terminal, shell, or Linux machine. These include:

- `git clone <remote-repo-url>`
  - Clones a repository, which can be found on GitHub (see Figure 6). The `<remote-repo-url>` argument should be the link to the repository from GitHub.
- `gitclone -b <branch-name> <remote-repo-url>`
  - Clones a specific branch from a repository.

  | ![alt text](/figs/github-clone.bmp "Figure 6") |
  |:--:|
  | *Figure 6. This is the GitHub page for pv-finder. In the top right corner, the Code drop menu can be selected and the repository link copied to the clipboard.* |
- `git add -A`
  - Generally speaking, when you want to push changes, this is what to use.  This stages the changes made to be committed.
- `git commit -m "Your message"`
  - Commits changes to Git with associated message.
  - The message is essentially the documentation for changes made in the GitHub repository, so ensure it is meaningful and descriptive.
- `git push`
  - Pushes changes to GitHub repository, making them a part of the repository now, which others can pull and clone.
  - This can be thought of as uploading the changes made in the cloned repository to the GitHub, official repository.
- `git status`
  - Displays whether there are any modified files, stages files, and/or committed files.
- `git pull`
  - "Pulls" (i.e. copies, removes, adds, or updates) any changes from the repository into your clone of the repository.
- `git reset`
  - Resets all the files in the project, removing any staged changes before they are committed/pushed to the repository.

These commands will function well enough to push/pull code to/from the repository, but not much else. In order to gain a more comprehensive understanding, see the next section on Resources to learn Git/GitHub.

### 1.5.2. **Resources**

- In-depth [tutorial](https://www.youtube.com/watch?v=RGOj5yH7evk) on almost everything needed to know to use Git and GitHub.
- General overview of how GitHub flow works. The rest of the [website](https://guides.github.com/introduction/flow/) website offers some useful guides, too.

## 1.6. **Linux**

Linux is an operating system that is used to connect and work with the computers that run the jobs for pv-finder. A great deal of discussion could be had about what, precisely, Linux is and what it is used for. For the purpose of operating pv-finder, however, this is not necessary. Rather, a discussion will be had about how to connect to the servers used for pv-finder (goofy, sleepy), some basic commands to navigate these servers, and how to set up a PC to connect to these servers.

### 1.6.1. **Setup**

To connect to a server that runs pv-finder jobs, a config file must be created that contains the necessary commands to execute this operation. The process described below assumes that this is being done on a Windows PC, though a similar process could likely be applied for Mac users.

First, you need to navigate to the .ssh directory, if you have one. To do this, open File Explorer and go to `C:\Users\<Name>\.ssh`. If a .ssh directory does not exist, create a folder named `.ssh` and place it in this directory.

Once inside the .ssh directory, create a file named `config`. This can be done by creating a text document `config.txt`, then deleting the `.txt` extension at the end of it. Once this is done, right click the file, select *Open With*, then choose Notepad (or a similar text editor). Then, enter the following text:

    Host goofy-XXXX
      HostName goofy.geop.uc.edu
      User <username>
      ProxyCommand C:\Windows\System32\OpenSSH\ssh.exe -q -x <username>@earth.phy.uc.edu -W %h:%p
      LocalForward XXXX localhost:XXXX
      LocalForward 6009 localhost:6009
    
    Host sleepy-YYYY
        HostName sleepy.geop.uc.edu
        User <username>
        ProxyCommand C:\Windows\System32\OpenSSH\ssh.exe -q -x <username>@earth.phy.uc.edu -W %h:%p
        LocalForward YYYY localhost:YYYY
        LocalForward 6007 localhost:6007
        LocalForward 6009 localhost:6009

`XXXX` and `YYYY` are unique ports that should be assigned to you. They will typically start with an 8 or 9 (e.g. 8890). `<username>` should be the username assigned to you during account creation by an administrator.\\

### 1.6.2. **Basic Commands**

Compiled here is a list of commands to use when connected to the Linux machine, especially when working with pv-finder. This is not intended to be a comprehensive list of commands (as this can be found in the Linux manual (see [here](https://www.man7.org/linux/man-pages/index.html)), but rather a cheat sheet to navigate pv-finder.

Note: do not type any <> that are present in the commands. These are merely intended to denote when a command unique to the user must be inputted.

- `ssh <hostname>`
  - Connects to Linux server
  - If you followed the setup above, `<hostname>` should be `goofy-XXXX` or `sleepy-YYYY` (or equiv.).
- `conda activate [env]`
  - Sets Python interpreter and environments (pytorch, matplotlib, pandas, etc.).
  - The `<env>` should be the environment of choice.
    - Currently, this is `goofit-june2020` in goofy and `june2020-gpu` in sleepy.
- `ls`
  - List information about the files in the current directory.
- `cd <path>`
  - Enters a directory. If the `ls` command were typed after this, it would show all of the contents/sub-directories inside of the directory you stepped into.
- `ps -aef | grep jupyter`
  - Shows jupyter notebooks/labs currently running.
  - Useful for finding lingering jobs that should be canceled, other users running notebooks/labs, and more.
  - Contains a PID key used for other commands.
- `jupyter notebook --no-browser --port <port #> &`
  - Starts a jupyter notebook and returns a series of url's and paths. Copy one and paste into browser of choice.
  - The `<port #>` should be the port number assigned to you. These were specified in the Setup section above.
- `jupyter lab --no-browser --port <port #> &`
  - Starts a jupyter labs and returns a series of url's and paths. Copy one and paste into browser of choice.
  - The `<port #>` should be the port number assigned to you. These were specified in the Setup section above.
- `mlflow ui --port <port #> --backend-store-uri file:///share/lazy/pv-finder_model_repo \&`
  - Launches MLFlow, using the path given for mlflow's stored runs repository
  - Type `localhost:<port #>` into browser to view MLFlow.
  - The `<port #>` should be the port number entered in your config file (see [Setup](#setup) above).
  - Generally, I've found this does not need to be run each time you connect to the server.
- `screen -S <name>`
  - Creates a screen.
  - The `<name>` is the name of the screen, which can be anything you want.
  - Create a screen after activating the python environment (conda) when you want to run a job that you want to close out of. Otherwise, closing out of your browser or disconnecting from the server will end the job.
- `screen -r <name>`
  - Enter `<name>` screen session.
  - If you only have one screen running, entering the session name is optional.
- `screen -X -S <name> quit`
  - Quit `<name>` session.
  - This will cancel any jobs running on the screen.
- `lsof -i:<port #>`
  - Command used to view PIDs in a given port.
  - Mostly used as a intermediate command for the kill command.
- `kill -9 PID`
  - Kills a program.
  - `PID` should be the `PID` number for a given program.
  - Usually used to end a jupyter notebook/lab.
- `mkdir <name>`
  - Creates a sub-directory inside of the current directory.
  - The `<name>` should be the name of the sub-directory you want to create.
- `nvidia-smi`
  - Shows the server's GPU's and their capacity (and other information).
  - Usually used to view the memory usage of a GPU and change the device used for your job, if necessary.
- `&`
  - Attach to the end of any command to run in the background.
  - Useful for running other commands while a command executes.

### 1.6.3. **Connecting to Server**

When connecting to a server, the typical process that will occur is this:

Open the terminal of choice (command prompt, vsc terminal, etc.)

1. Type `ssh <hostname>`
2. Enter the password(s)
3. Type `conda activate <env>`
4. Type `cd <directory to work on>`
5. `jupyter lab --no-browser --port <port #> &`
6. Copy the url and paste into browser

These instructions assume that the config file was setup and created correctly as explained in the Setup section above. If you are working within a screen that is already created, the process is simplified even more:

1. Open the terminal of choice (command prompt, vsc terminal, etc.)
2. Type `ssh <hostname>`
3. Enter the password(s)
4. Type `screen -r`

Sometimes, the jupyter notebook/lab will say that it needs a token/password for you to connect to an already-running instance. To fix this, simply type `jupyter notebook list`, find your notebook, copy the characters after `token=`, and paste into the token box in the notebook/lab. This will grant you access to your work.

## 1.7. **Model**

A model usually refers to a particular neural network architecture, which is the framework for how a computer is able to learn. Here, the current process for creating models from scratch will be presented, as well as how to import a model into a notebook for use.

### 1.7.1. **Architectures**

It was already shown in the introduction that there are two models used by default in pv-finder. These are not the only the models used in testing, nor are they the full representations of our best-performing model architectures.

#### 1.7.1.1. **AllCNN**

In an earlier section of this manual, the AllCNN in its simplest form was presented. Further work has since been done to add complexity to the model in order to increase its performance. These features include skip connections, batch normalization, hyperparameter tuning (i.e. experimenting to find the best learning rate), parameter tuning (i.e. experimenting to find the best kernel sizes, channel sizes, etc.), and more.

#### 1.7.1.2. **Perturbative AllCNN**

The AllCNN with perturbation include the x and y components of data from the tracks, which can help improve the learning process for a machine learning algorithm. ~~These perturbation features, however, cannot be added immediately. In order to implement them, AllCNN must first have well-trained weights for the z-projection data (often called the input, or X, data), which in itself requires SimpleCNN to be well-trained. This makes training a perturbative AllCNN network a multi-step process:~~

~~1. Acquire well-trained SimpleCNN weights.~~
~~2. Replace the last fully-connected layer with a convolutional layer and well-train this AllCNN network.~~
~~3. Freeze weights of AllCNN, add the extra (x,y) feature set, and train the set on perturbative AllCNN network.~~
~~4. Unfreeze all weights and train the perturbative AllCNN network.~~

***NOTE**: As of September 30th, 2021, an AllCNN model with the x,y feature set (a.k.a. perturbation features) can successfully generate a well-trained model in "one shot" using the poca_kde and DenseNet architecture.*

#### 1.7.1.3. **UNet**

UNet is a complete different architecture implemented by Will Tepe in 2020. In it, he took an architecture originally created for biomedical image segmentation and applied it to the pv-finder scenario, with the KDE histogram serving as the input, like normal.

UNet, in layman's terms, works by having layers with different levels of resolution for the data learn features and patterns which, when combined, creates a robust and well-trained network capable of picking out primary vertices quite well. Its results rival that of the AllCNN networks using many metrics of performance.

For a more technical definition, UNet uses a series of convolutional layers to downsample the dimension of the input features (as well as MaxPooling, if needed), then uses upsampling techniques to return to the original input dimension. This is what gives the model its name, as the structure creates a sort of "U" shape, where the input features start large, shrine, then become larger again. To complexify the model further, skip connections are also used from one side of the "U" to another, which helps with the dimensions and preserves earlier, less complex, features that the network picks out. An example of the UNet architecture can be found below.

| ![alt text](/figs/unet.bmp "Figure 7") |
  |:--:|
  | *Figure 7. This is the UNet diagram shown in the original paper it was presented in (see paper [here](https://arxiv.org/abs/1505.04597v1)).* |

Another represents that matches the earlier format of SimpleCNN and AllCNN can be seen below.

| ![alt text](/figs/unet-arch.bmp "Figure 8") |
  |:--:|
  | *Figure 8. This is the UNet diagram shown in the original paper it was presented in (see paper [here](https://arxiv.org/abs/1505.04597v1)).* |

#### 1.7.1.4. **Other Architectures**

There are many other architectures that are untested, but could work. Some architectures (as of writing this manual) are in the process of being tested for their viability. One notable example is the **DenseNet**, which is a sort of extreme version of the AllCNN network where every layer has the maximum number of skip connections to previous layers possible. Another model architecture of interest that is currently being explored is the graph neural network (GNN), which could radically alter the way we find primary vertices. It is possible that GNNs could render the whole process of generating KDEs redundant. This remains to be seen and tested, though it is interesting to think about.

A final, more recent, model implemented into pv-finder is **UNet++**. This architecture adds a series of densely connected layers between each skip connection. A diagram for the architecture can be seen below.

| ![alt text](/figs/unet++.bmp "Figure 9") |
  |:--:|
  | *Figure 9. This is the UNet++ diagram shown in the original paper it was presented in (see paper [here](https://arxiv.org/abs/1807.10165)).* |

### 1.7.2. **Using a Model**

Using a model for pv-finder is a relatively simple process, as it is largely already completed. Navigate to the notebooks subdirectory in pv-finder and select a .ipynb file that you wish to run. Make sure to use an existing model.

To add a model to a model notebook for testing, first locate which file the model you want to use is in (usually .py file type) in the `model` subdirectory. Once this is done, locate the exact name of the model you want to use. This should be the name of a class. An example of a model in one of these files can be seen in Figure 10 below.

| ![alt text](/figs/model-file-ex.bmp "Figure 10") |
  |:--:|
  | *Figure 10. Model file example. Here, models_mjp_07June21.py is the model file and ACN_1i4_10L_4S is the model name.* |

The two pieces of information should then be copied into the jupyter notebook at the appropriate place using the example format below.

    from model.models_mjp_19Nov20 import SimpleCNN5Layer_Ca_A as ModelA
    from model.models_mjp_19Nov20 import SimpleCNN5Layer_Ca_E as ModelE
    from model.models_mjp_19Nov20 import SimpleCNN7Layer_Ca_W as ModelW
    from model.models_mjp_26Dec20 import SimpleCNN9Layer_Ca_X as ModelX
    from model.models_mjp_30Jan21 import ACN_1_10L_4S as ModelXX
    from model.models_mjp_07June21 import ACN_1i4_8L_DenseNet as ModelDN8
    from model.models_mjp_07June21 import ACN_1i4_10L_DenseNet_BN as ModelDN10

### 1.7.3. **Creating a Model**

This section will discuss where to find existing models and suggest ideas for how to format these models to stay consistent and organized with the rest of the existing model files. This section will not, however, discuss what to do in order to write your own model. This is a subject matter better suited to your own personal study and research. Exploring the existing model files and the way they are written is another way to familiarize yourself and provide ideas for how to structure and write your own model architectures.

Existing models for pv-finder can be found in the `pv-finder` directory, under the `model` subdirectory. It is best to sort these by the last date they were modified, as there are swaths of old models files that are out of date and/or no longer used$^1$. Another way to find recent models is to look at the file names. Oftentimes (though not always), the name of the file indicates the date it was created. This can be a starting point for finding more recent model files to look at.

It is the current practice to create new files every few months for any new models that you implement. This is not a rule, per say, but a general, historical guideline to keep in mind. Following in this idea prevents model files from being bloated with too many different models to the point where the file is no longer navigable.

Model names should also be descriptive enough to get a quick, at-a-glance look at what the model does. Each model should also include a series of comments above the class declaration which defines the model's parameters and features in more detail. An example of these practices can be seen below.

    '''
    Modified network architecture of benchmark with the following attributes:
    NOTE: All attributes shared with benchmark are omitted
    1. Three feature set using X, x, y.
    2. 10 layer convolutional architecture for X feature set.
    3. 4 layer conovlutional architecture for x and y feature set.
    4. Takes element-wise product of the two feature sets for final layer.
    5. Channel count follows the format:    01-20-10-10-10-10-07-05-01-01 (X), 20-10-10-01 (x, y),  20-01 (X, x, y)
    6. Kernel size follows the format:      25-15-15-15-15-15-09-05-91 (X),    25-15-15-91 (x, y),  25-91 (X, x, y)
    7. 4 skip connections, located at layers 3,5,7,9
    '''
    class ACN_1i4_10L_4S(nn.Module):

$^1$*Work is currently being done to segregate outdated, redundant files from new, useful ones. This may take some time and is not yet complete.*

## 1.8. **MLFlow**

Taken from the MLflow website:

>MLflow is an open source platform to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry.

 MLflow lets you save model "runs", which are training phases that end with a output file containing the results of said run. These results can be viewed and loaded into future runs. Parameters, efficiency/false positive measurements, and other user-defined metrics may be included for future viewing, reference, comparison, or  use in presentations. MLflow provides almost all of the basic framework for accomplishing these tasks and more. Mlflow also stores the path to the model's run data, which lets you load the weights for a future model. In the most recent

You can find more about MLflow on their website, found [here](https://mlflow.org/).

### 1.8.1. **Setup**

The setup for MLflow can be found in the [Basic Commands](#basic-commands-1) section, as well as here. To set up MLflow in your session, first add the MLflow file setup (found in the [Setup](#setup) section). Then, type the following command once logged into goofy or sleepy:

    mlflow ui --port <port #> --backend-store-uri file:///share/lazy/pv-finder_model_repo/ML/mlflow &

You can then type `localhost:<port #>` into browser to view MLFlow. The `<port #>` should be the port number entered in your config file. Generally, I've found this does not need to be run each time you connect to the server.

### 1.8.2. **Navigating MLFlow**

| ![alt text](/figs/mlflow-ui.bmp "Figure 11") |
  |:--:|
  | *Figure 11. This is a preview of what mlflow might look like when you first launch it using the command given above (using the same directory).* |

Once you are able to launch MLflow and reach a page similar to the one found in Figure 8 above, you can spend some time exploring all the options the software has to offer. It is actually quite intuitive and easy to navigate. On the left sidebar, you can choose which "experiment" you want to view, which is essentially just a series of folders containing runs of your choice. Each experiment is assigned a number that is a part of every save file's path. In the top right corner -- the "Columns" button -- you can select what columns you want to view. In the model notebook, you can set up parameters that MLflow tracks and saves in the run's save file, which can be viewed here. You can add learning rate, false positive rate, efficiency, and any custom tags that you desire, which each take up their own column.

## 1.9. **References**

[1]: https://arxiv.org/pdf/1906.08306.pdf "A hybrid deep learning approach to vertexing"
[2]: https://arxiv.org/abs/2103.04962 "Progress in developing a hybrid deep learning algorithm for identifying and locating primary vertices"
[3]: https://arxiv.org/abs/1505.04597v1 "U-Net: Convolutional Networks for Biomedical Image Segmentation"
