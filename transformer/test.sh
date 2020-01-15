save_model_path=model/step02_420_80.h5
epoch=0
step=20
thred=100
person='Homer'
input_path='../data/simpson_pkl/'${person}'_Simpson_Q'
output_path='../data/simpson_pkl/'${person}'_Simpson_A'
while [ ${epoch} -lt ${thred} ]
do
  pre_train_model_path=${save_model_path}
  epoch=`expr ${epoch} + ${step}`
  save_model_path=model/${person}_${epoch}.h5
  #python train.py --epoch ${step} --pre_train 1 --load_tokenizer 1 --load_tokenizer_path 'tokenizer/token01' --pre_train_model_path ${pre_train_model_path} --save_model_path ${save_model_path} --input_path ${input_path} --output_path ${output_path}
  echo "=============train success, epoch ${epoch}==========="
done

