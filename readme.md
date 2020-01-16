# Talking with Simpsons: Chatbots with Different Personalities

Project repo for Google AI Winter Camp 2020

Chatbots with personalities of main characters of *The Simpsons*

## Dataset
Our project is based on following datasets:
1. [Dialogue Lines of The Simpsons](https://www.kaggle.com/pierremegret/dialogue-lines-of-the-simpsons) in `
2. [Cornell Movie--Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
3. [PersonaChat ConvAI2](http://convai.io/#personachat-convai2-dataset)
4. Emojify-Data EN

## Model
We tried two different methods for dialogue generation. And a classification model is trained to get the answer that best suits the character's personality.
We managed to make the chatbots response with emojis, in order to make it more like a real conversation online.

### Method 1: Transformer
train model:
```
sh train.sh
```
pretrained model
### Method 2: GPT2



## Guide
To try the Chatbot:
```
cd chatbot_website/
python manage.py makemigrations
python manage.py migrate
```
Then, to Launch the Server Locally:
```
redis-server &  # Launch Redis in Background
python manage.py runserver 127.0.0.1:8000
```
Then, Go to the Browser: http://127.0.0.1:8000/


## Code
1. ```/gpt2```contains all codes of run

## Reference
1. [https://github.com/yangjianxin1/GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)
2. [https://github.com/huggingface/transfer-learning-conv-ai](https://github.com/huggingface/transfer-learning-conv-ai)
3. [https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html](https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html)
4. Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017.
5. https://github.com/shbhmbhrgv/Personality-Chatbot
