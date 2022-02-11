# GraphGeo
![](https://img.shields.io/badge/python-3.8.12-green)![](https://img.shields.io/badge/pytorch-1.8.2-green)![](https://img.shields.io/badge/cudatoolkit-10.2.89-green)![](https://img.shields.io/badge/cudnn-7.6.5-green)

This folder provides a reference implementation of **GraphGeo** and the **datasets** collected from *New York*, *Los Angeles*, and *Shanghai* as described in the paper:

## Basic Usage

### Requirements

The code was tested with `python 3.8.12`, `pytorch 1.8.2`, `cudatookkit 10.2.89`, and `cudnn 7.6.5`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name GraphGeo python=3.8.12

# activate environment
conda activate GraphGeo

# install pytorch & cudatoolkit
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts

# install other requirements
conda install numpy pandas
pip install scikit-learn
```

### Run the code
```shell
### we run our model on the New York dataset as an example.
# Open the "GraphGeo" folder
cd GraphGeo

# unzip data file in "New_York"
unzip ./datasets/New_York/data.zip

# loda data from the region and execute IP clustering. 
python load_and_cluster.py --dataset "New_York" 

# run the model GraphGeo
python run.py --dataset "New_York" --dim_in 30
```
## Folder Structure

```tex
└── GraphGeo
	├── datasets # contains datasets from three regions for street-level IP geolocation
	│	|── Los_Angeles
	│	|── New_York
	│	|── Shanghai
	├── load_and_cluster.py # Load dataset and execute IP clustering the for model running
	├── model # contains model implemention files
	│	|── cnf.py # Implement the continuous-form normaling flow
	│	|── layers.py # Implement the attibute similarity module
	│	|── model.py # The core source code of proposed GraphGeo
	│	|── ode # The support files for "cnf.py"
	│	|── sublayers.py # The support file for "layer.py"
	├── run.py # Run model for training and test
	├── utils.py # Defination of auxiliary functions for running
	└── README.md # This document
```

## Dataset Information

The "datasets" folder contains three subfolders corresponding to the datasets collected from different regions. There are three files in each subfoler:

- data.csv    *# features and labels for street-level IP geolocation* 
- ip.csv    *# IP addresses*
- last_traceroute.csv    # last four routers and coressponding delays for efficient IP host clustering

The detailed **columes and description** of each dataset are as follows:

#### New York & Los Angeles

| Column Name                     | Data Description                                             |
| ------------------------------- | ------------------------------------------------------------ |
| ip                              | The IPv4 address                                             |
| as_mult_info                    | The ID of the autonomous system where IP locates             |
| country                         | The country where the IP locates                             |
| prov_cn_name                    | The state/province where the IP locates                      |
| city                            | The city where the IP locates                                |
| isp                             | The Internet Service Provider of the IP                      |
| vp900/901/..._ping_delay_time   | The ping delay from probing hosts "vp900/901/..." to the IP host |
| vp900/901/..._trace             | The traceroute list from probing hosts "vp900/901/..." to the IP host |
| vp900/901/..._tr_steps          | #steps of the traceroute from probing hosts "vp900/901/..." to the IP host |
| vp900/901/..._last_router_delay | The delay from the last router to the IP host in the traceroute list from probing hosts "vp900/901/..." |
| vp900/901/..._total_delay       | The total delay from probing hosts "vp900/901/..." to the IP host |
| longitude                       | The longitude of the IP (as label)                           |
| latitude                        | The latitude of the IP host (as label)                       |

#### Shanghai

| Column Name                       | Data                                                         |
| --------------------------------- | ------------------------------------------------------------ |
| numip                             | The IPv4 address                                             |
| asnumber                          | The ID of the autonomous system where IP locates             |
| scene                             | The IP usage scenario                                        |
| isp                               | The Internet Service Provider of the IP                      |
| asname                            | The name of the autonomous system where the IP locates       |
| orgname                           | The ISP organization of the IP host                          |
| address                           | The address of the ISP                                       |
| port_80/443/..._alive             | The opening status of the IP host's port 80/443/...          |
| aiwen/vp813/..._ping_delay_time   | The ping delay from probing hosts "aiwen/vp813/..." to the IP host |
| aiwen/vp813/..._trace             | The traceroute list from probing hosts "aiwen/vp813/..." to the IP host |
| aiwen/vp813/..._tr_steps          | #steps of the traceroute from probing hosts "aiwen/vp813/..." to the IP host |
| aiwen/vp813/..._last_router_delay | The delay from the last router to the IP host in the traceroute liist from probing hosts "aiwen/vp813/..." |
| aiwen/vp813/..._total_delay       | The total delay from probing hosts "aiwen/vp813/..." to the IP host |
| longitude                         | The longitude of the IP (as label)                           |
| latitude                          | The latitude of the IP host (as label)                       |
