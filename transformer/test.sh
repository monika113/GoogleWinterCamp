save_model_path=model/step_new.h5
epoch=20
step=50
while ((${epoch} < 1000))
do
  pre_train_model_path=save_model_path
  epoch=`expr ${epoch} + ${step}`
  save_model_path=model/step01_${epoch}.h5
  python train.py --epoch 50 --pre_train 1 --load_tokenizer 1 --load_tokenizer_path 'tokenizer/token01' --pre_train_model_path pre_train_model_path --save_model_path save_model_path
  echo "=============train success, epoch ${epoch}==========="
done