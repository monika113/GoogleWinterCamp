input_path='../persona_data/train_enc.pk'
output_path='../persona_data/train_dec.pk'
save_model_path='model/step02_320_1.h5'
pre_train_model_path='model/step01_320.h5'

python train.py --epoch 1 --pre_train 1 --load_tokenizer 1 --load_tokenizer_path 'tokenizer/token01' --pre_train_model_path ${pre_train_model_path} --save_model_path ${save_model_path} --input_path ${input_path} --output_path ${output_path}

