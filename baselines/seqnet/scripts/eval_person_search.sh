#block(name=eval_person_search, threads=5, memory=18000, subtasks=1, hours=24, gpus=1)

# set gpus...
gpus='"device='"$CUDA_VISIBLE_DEVICES"'"'
echo $gpus

# move to docker directory
cd $PWD/docker/

# run build script
./build.sh
USERNAME=user

# set up docker volumnes
SRC_DIR="$PWD/../"
DATASET_DIR="/media/datasets/"
CEPHS="/home/group-cvg/"
DATA_DIR="/media/data/"
XSERVER="/tmp/.X11-unix/"


docker run\
    --gpus $gpus\
    --shm-size="50g"\
    -v "$SRC_DIR":/home/$USERNAME/SeqNet\
    -v "$DATASET_DIR":/media/datasets\
    -v "$DATA_DIR":/media/data\
    -v "$CEPHS":/home/group-cvg\
    -e PYTHONPATH=/home/$USERNAME/SeqNet\
    --rm\
    -it\
    seqnet \
    python /home/$USERNAME/SeqNet/training/train.py --eval --cfg exps/exp_posetrack/config.yaml --ckpt exps/exp_posetrack/epoch_5.pth  EVAL_USE_CBGM True
