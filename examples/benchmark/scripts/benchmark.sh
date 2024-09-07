export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# for name_model in "gpt2-4b" "gpt2-10b" "gpt2-15b" "opt-30b"; do
#     for num_gpu in 4 8; do
#         for batch_size in 4 8; do
#             echo "****************** Begin ***************************"
#             T_MODEL=${name_model} N_GPU=${num_gpu} N_BS=${batch_size} bash ./deepspeed.sh
#             T_MODEL=${name_model} N_GPU=${num_gpu} N_BS=${batch_size} bash ./elixir.sh
#             echo "****************** Finished ***************************"
#             echo ""
#             echo ""
#         done
#     done
# done

# T_MODEL="gpt2-4b" N_GPU=8 N_BS=4 bash ./deepspeed.sh
# T_MODEL="opt-350m" N_GPU=8 N_BS=4 bash ./elixir.sh
T_MODEL="gpt2-4b" N_GPU=8 N_BS=4 bash ./elixir.sh
