# Installation Python Environment Setup

1. install miniconda or anaconda
2. install conda packages

   ```bash
   # create new environment
   conda env remove -n ENV_NAME -y
   conda create -n ENV_NAME -y python=3.11

   # install conda packages
   conda activate ENV_NAME
   conda update --all -y
   conda clean --all -y

   # install other conda packages ...
   conda install numpy h5py cython imageio scikit-image matplotlib yarn imaris-ims-file-reader multiprocess

   # install our zimg package
   conda config --append channels conda-forge
   conda install zimg -c fenglab

   # install opencv
   pip install --upgrade --no-cache-dir opencv-contrib-python 


   ```
