#!/usr/bin/env bash
USERNAME=user

cp Dockerfile Dockerfile.bkp
echo "RUN adduser --disabled-password --gecos \"\" -u $UID $USERNAME"  >> Dockerfile
echo "USER $USERNAME" >> Dockerfile
echo "WORKDIR /home/$USERNAME" >> Dockerfile

echo "WORKDIR /home/$USERNAME/SeqNet" >> Dockerfile

docker build --tag seqnet .

rm Dockerfile
mv Dockerfile.bkp Dockerfile
