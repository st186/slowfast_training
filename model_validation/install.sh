#!/bin/bash
echo "--------------- Installation is Initiated ---------------"

# install git
apt install -y git

# clone slow fast
git clone https://github.com/facebookresearch/slowfast.git
ls
cd /app/slowfast || exit


apt-get update -y

# slow fast setup
python3 setup.py build develop

# slow fast dependency installation
pip3 install packaging
pip3 install cycler
pip3 install kiwisolver
pip3 install wheel
pip3 install -U iopath
pip3 install 'git+https://github.com/facebookresearch/fvcore'
pip3 install simplejson
pip3 install psutil
pip3 install tensorboard
pip3 install 'git+https://github.com/facebookresearch/fairscale'
pip3 install pytorch==1.9.0
pip3 install pyyaml==5.1
pip3 install av
pip3 install pytz
pip3 install sklearn
pip3 install torchvision==0.10.0

# detectron installation
TORCH_VERSION=$(python3 -c"import torch;print('.'.join(torch.__version__.split('.')[:2]))")
CUDA_VERSION=$(python3 -c"import torch;print(torch.__version__.split('+')[-1])")
echo "torch: $TORCH_VERSION  cuda: $CUDA_VERSION "
pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_VERSION/torch$TORCH_VERSION/index.html

cd ..