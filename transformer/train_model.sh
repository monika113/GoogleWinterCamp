input_path='../data/simpson_pkl/Homer_Simpson_Q'
output_path='../data/simpson_pkl/Homer_Simpson_A'
save_model_path='model/Homer_1.h5'
pre_train_model_path='model/step02_420_80.h5'

python train.py --epoch 1 --pre_train 1 --load_tokenizer 1 --load_tokenizer_path 'tokenizer/token01' --pre_train_model_path ${pre_train_model_path} --save_model_path ${save_model_path} --input_path ${input_path} --output_path ${output_path}

