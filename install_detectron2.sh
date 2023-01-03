DETECTRON2_VERSION=$1
FORCE_CUDA="1"
TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

if   [ ${DETECTRON2_VERSION} = "0.6" ]; then
    TORCH_VERSION="1.10"
    TORCHVISION_VERSION="0.11.0"
elif [ ${DETECTRON2_VERSION} = "0.5" ]; then
    TORCH_VERSION="1.9"
    TORCHVISION_VERSION="0.10.0"
fi

pip install torch==${TORCH_VERSION}+cu111 \
            torchvision==${TORCHVISION_VERSION}+cu111 \
            -f "https://download.pytorch.org/whl/torch_stable.html"
pip install detectron2==${DETECTRON2_VERSION} \
    -f "https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch${TORCH_VERSION}/index.html"