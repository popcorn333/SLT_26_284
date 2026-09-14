source /store/store4/software/bin/anaconda3/etc/profile.d/conda.sh
conda activate CUDA122_Grad-TTS_eval

job_name='GPM_Hydra_Blur_X0_v3'    #Hydra job name
task_prefix='model.Masking.d:'       #Hyper parameter name 

echo "$(date)"


#for value in 10 20 30 40; do   #name is the Hyper Paremeter name
#    ./tools/MCD/MCD.sh "$job_name" "model.Masking.a:0.04__model.Masking.b:0.01__model.Masking.c:${value}__model.Masking.d:0.2"
#done

./tools/MCD/MCD.sh "$job_name" 'model.Masking.c:0__model.Masking.d:0.1__training.learning_rate:0.0001__training.load_decoder:True__training.load_encoder:True__training.n_epochs:700__training.start_checkpoint:490__training.train_encoder:True'

echo "$(date)"

