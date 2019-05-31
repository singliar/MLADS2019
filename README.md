# Practical forecasting with AutoML

In this tutorial, we will first learn how to set up forecasting problems with AutoML in the benign, clean-data case.
Then we'll look at some of the more complex cases that come up in real world datasets and how to shape the data so 
that AutoML forecasting will eat it.

## Where and when

Currently scheduled for Kalaloch (35/2615) on Jun 6, 2019 from 1pm to 3pm. 
Times and places sometimes change, verify with the [official MLADS agenda](https://aka.ms/mlads-agenda).

## Lesson plan

_"Planning is useful though the plans are normally useless."_

The plan for the tutorial (90 minutes) is:
* 15 minutes – set up with AutoML (clone this repo, AzureML notebooks repo, solve hiccups).
* 30 minutes – go through the standard forecasting notebooks
* 15 minutes - [slides] common data problems, how to diagnose and solve them
* 30 minutes – Demonstrations of data wrestling
* 30 minutes - schedule slip, questions, coffee breaks


## Prerequisites and setup

You should already have these:
* Bring a laptop with [Python 3.6](https://www.anaconda.com/distribution/) and [git client](https://git-scm.com/downloads) installed
* Have an Azure subscription (see below if you don't have one)

### Setting up AutoML
* Clone the [Machine Learning Notebooks repo](https://github.com/Azure/MachineLearningNotebooks).
* Open a shell or command prompt window, go to `/how-to-use-azureml/automated-machine-learning` 
  and execute the `automl_setup` script appropriate for your platform (Win, Linux, Mac). 
  Many packages will be installed (5 minutes on good network).
* A browser window with Jupyter will open. Ctrl+C the ipykernel in the terminal.
* `conda activate azure_automl`
* `pip install binpacking`
* Re-start jupyter in the root directory of the repo (two folders up) with `jupyter notebook`
* Open the setup notebook [configuration.ipynb](https://github.com/Azure/MachineLearningNotebooks/blob/master/configuration.ipynb).
* Make sure to use the `azure_automl` kernel.
* Transfer your subscription id,  resource group name and workspace you want to use into the second code cell of the notebook.
* Run cells according to instructions. 

### Setting up this repo

One day, these tools, and much more, will become part of AutoML. Until they do, they live here.

* Clone this repo
* Copy config.json from your AutoML repo into this directory
* Start a new `jupyter notebook` here and run the notebook.

### How to get a free Azure subscription

1. Create a Microsoft account at https://outlook.com (skip if you already have an account @outlook.com, @Hotmail.com, or @live.com)
2. Use your Microsoft account to get the free Azure subscription https://azure.microsoft.com/en-us/free/
  * You get 200$ Azure credits expiring in 30 days
  * Credit card needed for identification purpose. You will not be charged even after 30 days or the 200$ credits are used
