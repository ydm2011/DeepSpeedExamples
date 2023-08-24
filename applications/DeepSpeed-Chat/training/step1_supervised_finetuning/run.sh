#cd ~/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning &&  sudo sh training_scripts/single_gpu/run_1.3b.sh |  sudo tea /data/log/info.log
.  /home/ubuntu/software/anaconda3/etc/profile.d/conda.sh 
export PATH="/home/ubuntu/software/anaconda3/bin:$PATH" 
sudo mkdir -p /data/log 
sudo chown -R ubuntu:root /data/log 
ZERO_STAGE=$1
MODEL_NAME=$2
if [ "$ZERO_STAGE" = "" ]; then
    ZERO_STAGE=0
fi
if [ "$MODEL_NAME" = ""]; then
  MODEL_NAME=facebook/opt-1.3b
fi

sh training_scripts/opt/single_gpu/run_test_1.3b.sh $ZERO_STAGE $MODEL_NAME
