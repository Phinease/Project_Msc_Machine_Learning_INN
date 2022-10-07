# HVS_INN_MC_Project
Hypervision surgical INN project with MC simulation dataset

> Hyperspectral imaging is a safe, non-contact and contrast-agent-free optical imaging modality perfectly suited to provide intraoperative tissue characterisation for more precise and safer surgery. This project will focus on developing invertible neural network approaches to extract physiological tissue properties and associated uncertainty, such as blood perfusion, from sparse hyperspectral imaging data.

- Host: HyperVision Surgical Ltd
- Student: Shuangrui CHEN (Msc Machine Learning of UCL)
- Internal supervisor: 
  - Dr Simon Arridge
  - Dr Tom Vercauteren
- Industrial supervisor:
  - Dr Michael Ebner
  - Dr Conor Horgan
  - Dr Mirek Janatka
  
## Requirement
```
TODO
```

## File Structure
```
.
├── pdf                     # key milestone of jupyternote book
├── data                    # dataset folder
├── logs                    # log of training process of each trial
├── runs                    # run env to push temp cache
├── tensors                 # export of tensors with longtime calculation (cuda output)
├── models                  # models and correspond config python file
├── src                     # lib of model, dataset and config
│   ├── config_*.py         # configuration of pipeline
│   ├── inn_model.py        # model and train functions
│   ├── loss_funs.py        # loss functions
│   └── dataset.py          # dataset class file
│   └── cuda_run.py         # single process of posterior estimation
│   └── cuda_main.py        # multiprocess of posterior estimation
│   └── camera_*.py         # camera simulation upon MC simulation data
│   └── utils.py            # Reuben's SuperResulution Demosiac data generation utils copy
```

## Run
## Pre-processing
### Data reformat
- Full band spectrum to 16-band spectrum
- CSV file cleanup and standardization

```
  data_reformat.ipynb
```

### Camera simulation pipeline
- HVS virtual camera simulation
```
  python camera_hvs_with_l1.py
```

- HVS camera / light source simulation
```
  python camera_simulate_pipeline.py
```

## Model
- Train and MAP validate

```
  model.ipynb
```

- Model visualisation
```
 model_visualisation.ipynb
 tensorboard --logdir=runs
```

- Benchmark
```
  benchmark.ipynb
```

## Validation
- Posterior estimation
```
  python cuda_main.py
```

- Uncertainty estimation
```
  uncertainty_estimation.ipynb
```

## HSI image application
- Sto2 estimation on HSI
```
  sto2_estimation.ipynb
```

- Sto2 distribution visualisation
```
  sto2_distribution.ipynb
```