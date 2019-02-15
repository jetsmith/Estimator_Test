arena submit tf --name=tf-estimator              \
    --gpus=1              \
    --workers=1             \
    --chief                  \
    --logdir=/data/mnist/jj_model                 \
    --data=tfdata:/data/mnist                  \
    --workerImage=tensorflow/tensorflow:1.9.0-devel-gpu  \
    --syncMode=git \
    --syncSource=https://github.com/corerain/distributed_training_job.git \
    --ps=1              \
    --psImage=tensorflow/tensorflow:1.9.0-devel   \
    --tensorboard \
    "python code/distributed_training_job/estimator.py"
