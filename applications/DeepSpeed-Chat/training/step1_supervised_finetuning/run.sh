#cd ~/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning &&  sudo sh training_scripts/single_gpu/run_1.3b.sh |  sudo tea /data/log/info.log
export PATH="/home/ubuntu/software/anaconda3/bin:$PATH" 
sudo mkdir -p /data/log 
sudo chown -R ubuntu:root /data/log 
.  /home/ubuntu/software/anaconda3/etc/profile.d/conda.sh 
mkdir workspace && cd workspace 
cd applications/DeepSpeed-Chat/training/step1_supervised_finetuning
sh training_scripts/opt/single_gpu/run_test_1.3b.sh
