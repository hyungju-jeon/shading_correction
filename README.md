# Installation Python Environment Setup

1. install miniconda or anaconda
2. install conda packages

   ```bash
   # create new environment
   conda env remove -n ENV_NAME -y
   conda create -n ENV_NAME -y python=3.8

   # install conda packages
   conda activate ENV_NAME
   conda update --all -y
   conda clean --all -y

   # install other conda packages ...
   conda install numpy h5py cython imageio scikit-image matplotlib yarn imaris-ims-file-reader

   # install pytorch
   # Linux or Windows with cuda:
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
   # Linux or Windows without cuda:
   conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly
   # macOS:
   conda install pytorch-nightly::pytorch torchvision torchaudio -c pytorch-nightly

   # install our zimg package
   conda config --append channels conda-forge
   conda install zimg -c fenglab

   # install opencv
   pip install --upgrade --no-cache-dir opencv-contrib-python 



   03. update conda packages
   # if has cuda:
   conda update --all -y -c fenglab -c pytorch-nightly cudatoolkit=11.0
   # otherwise
   conda update --all -y -c fenglab -c pytorch-nightly

   04. local packages related to pytorch, need to run if pytorch is updated
   zsh ./utils/update_libs_depends_on_pytorch.sh


   # install opencv
   pip install --upgrade --no-cache-dir opencv-contrib-python # or pip install opencv-python
   ```
