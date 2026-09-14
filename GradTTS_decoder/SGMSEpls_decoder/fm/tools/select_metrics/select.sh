N=10
job_name="GPM_Hydra_Blur_X0_v3"
root_dir="/exp/exp4/acp23xt/Hydra_${job_name}_eval"
config_dir_suffix="model.Masking.c:0__model.Masking.d:0.1__training.learning_rate:0.0001__training.load_decoder:True__training.load_encoder:True__training.n_epochs:700__training.start_checkpoint:490__training.train_encoder:True/dev/metrics"

MCD="${root_dir}/${config_dir_suffix}/mean_mcd.txt"
F0="${root_dir}/${config_dir_suffix}/mean_f0.txt"
stdMCD="${root_dir}/${config_dir_suffix}/std_mcd.txt"
stdF0="${root_dir}/${config_dir_suffix}/std_f0.txt"

echo "Processing $MCD,  $F0, $stdMCD, $stdF0"
python3 tools/select_metrics/select.py "$MCD" "$F0" "$stdMCD"  "$stdF0" "$N" &> "Metrics.txt"
echo "--------------------------------------"

