save_model_path=model/step01_420.h5
epoch=0
step=20
thred=100
input_path='../persona_data/train_enc.pk'
output_path='../persona_data/train_dec.pk'
while [ ${epoch} -lt ${thred} ]
do
  pre_train_model_path=${save_model_path}
  epoch=`expr ${epoch} + ${step}`
  save_model_path=model/step02_420_${epoch}.h5
  python train.py --epoch ${step} --pre_train 1 --load_tokenizer 1 --load_tokenizer_path 'tokenizer/token01' --pre_train_model_path ${pre_train_model_path} --save_model_path ${save_model_path} --input_path ${input_path} --output_path ${output_path}
  echo "=============train success, epoch ${epoch}==========="
done

