#block(name=SeqNet, threads=16, memory=18000, subtasks=1, hours=64, gpus=1)

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
CEPHFS="/home/group-cvg/"

docker run\
    --gpus $gpus\
    --shm-size="50g"\
    -v "$SRC_DIR":/home/$USERNAME/SeqNet\
    -v "$DATASET_DIR":/media/datasets\
    -v "$CEPHFS":/home/group_cvg/\
    -e PYTHONPATH=/home/$USERNAME/SeqNet\
    --rm\
    seqnet \
    python /home/$USERNAME/SeqNet/training/train.py --cfg configs/posetrack_psearch.yaml OUTPUT_DIR exps/seqnet_test/
