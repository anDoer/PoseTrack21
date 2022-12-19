if test "$#" -lt 1; then
  echo "No GPU specified! Please pass gpu ids, i.e. 0,1,2"
  echo "I will grant the docker container access to all gpus now"
  gpus='all'
else
  gpus='"device='"$1"'"'
fi

echo $gpus


./build.sh

USERNAME=user

SRC_DIR="$PWD/.."
DATASET_DIR="/media/datasets/"
CEPHS="/home/group-cvg/"
DATA_DIR="/media/data/"
XSERVER="/tmp/.X11-unix/"

docker run\
    --gpus $gpus\
    --shm-size="60g"\
    -v "$SRC_DIR":/home/$USERNAME/SeqNet\
    -v "$DATASET_DIR":/media/datasets\
    -v "$DATA_DIR":/media/data\
    -v "$CEPHS":/home/group-cvg\
    -v "$XSERVER":/tmp/.X11-unix:rw\
    -e PYTHONPATH=/home/$USERNAME/SeqNet\
    -e DISPLAY=unix$DISPLAY\
    --rm -it\
    seqnet \
    /bin/bash


