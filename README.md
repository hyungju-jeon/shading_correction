# jinnyLab Python setup

## Python Environment Setup
01. install miniconda or anaconda
02. install conda packages
    ```bash
    # create new environment
    conda env remove -n ENV_NAME -y
    conda create -n ENV_NAME -y python=3.8
    # install conda packages
    conda activate ENV_NAME
    conda update --all -y
    conda clean --all -y

    # install other conda packages ...
    conda install mkl numpy tbb scikit-learn scipy h5py cython ipykernel imageio protobuf future mock shapely pandas seaborn joblib anaconda-client conda-build ninja qt markdown scikit-image matplotlib mkl-service mkl_fft mkl_random

    # install pytorch
    # if has cuda:
    conda install pytorch torchvision cudatoolkit=11.0 -c pytorch-nightly
    # otherwise:
    conda install pytorch torchvision -c pytorch-nightly

    # install our zimg and opencv package
    conda install zimg conda-opencv -c fenglab
    ```
03. update conda packages
    ```bash
    conda activate ENV_NAME
    # if has cuda:
    conda update --all -y -c fenglab -c pytorch-nightly cudatoolkit=11.0
    # otherwise
    conda update --all -y -c fenglab -c pytorch-nightly
    ```
04. local packages related to pytorch, need to run if pytorch is updated
    ```bash
    zsh ./utils/update_libs_depends_on_pytorch.sh
    ```
05. install or update pip packages
    ```bash
    pip install yacs anytree termcolor tabulate grpcio tensorboard catboost lightgbm natsort lap pycocotools itk itk-elastix antspyx tensorstore networkit --upgrade --no-cache-dir
    ```

## Git Setup
Check https://rogerdudler.github.io/git-guide/index.ko.html
01. Make a folder and initialize git
    ```bash
    # Initialize git
    git init
    ```
02. Git pull and and remote repository
    ```bash
    # Pull from remote
    git pull https://github.com/hyungju-jeon/jinnyLab.git
    # Add remote repository as origin
    git remote add origin https://github.com/hyungju-jeon/jinnyLab.git
    ```
03. Update the progess
    ```bash
    # check which files have been updated
    git status
    # add files you want to update
    # add . will update all files you have changed
    git add .
    # commit with note
    git commit -m "Test message"
    # push to remote repository
    git push origin master
