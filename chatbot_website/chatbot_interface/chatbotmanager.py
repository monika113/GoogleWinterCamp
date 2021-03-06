from django.conf import settings
import logging
import sys

from django.apps import AppConfig
import sys
import os
import numpy as np

#data_folder = Path("source_data/text_files/")

#file_to_open = data_folder / "raw_data.txt"

chatbotPath = "/".join(settings.BASE_DIR.split('/')[:-1])
sys.path.append(chatbotPath)
from gpt2 import interact
from emoji_prediction import inference
from classsifier.character_classification_train import AttentionLSTM
from classsifier.character_classification_train import load_model_classifier
from classsifier.character_classification_train import predict_class
import random
# from chatbot import chatbot


logger = logging.getLogger(__name__)
use_classifier = True


class ChatbotManager(AppConfig):
    """ Manage a single instance of the chatbot shared over the website
    """
    name = 'chatbot_interface'
    verbose_name = 'Chatbot Interface'

    bot = None
    emoji_bot = None
    # def __init__(self):
    #     self.bot_list = []

    #TO DO: add model path!!
    model_prefix = '../gpt2/models/'
    person_name = ['homer', 'marge', 'bart', 'lisa']

    def ready(self):
        """ Called by Django only once during startup
        """
        # Initialize the chatbot daemon (should be launched only once)
        if (os.environ.get('RUN_MAIN') == 'true' and  # HACK: Avoid the autoreloader executing the startup code twice (could also use: python manage.py runserver --noreload) (see http://stackoverflow.com/questions/28489863/why-is-run-called-twice-in-the-django-dev-server)
            not any(x in sys.argv for x in ['makemigrations', 'migrate'])):  # HACK: Avoid initialisation while migrate
            ChatbotManager.initBot()

    @staticmethod
    def initBot():
        """ Instantiate the chatbot for later use
        Should be called only once
        """
        if not ChatbotManager.bot:
            logger.info('Initializing bot...')
            ChatbotManager.bot_list = []
            for i in range(4):
                # load diff model
                ChatbotManager.bot_list.append(interact.Chatbot())

            # ChatbotManager.bot = evaluate.Chatbot()
                ChatbotManager.bot_list[i].load_model(ChatbotManager.model_prefix+ChatbotManager.person_name[i])
        else:
            logger.info('Bot already initialized.')

        if not ChatbotManager.emoji_bot:
            logger.info('Initializing emoji...')
            ChatbotManager.emoji_bot = inference.Emojibot()
            ChatbotManager.emoji_bot.load_model()
        else:
            logger.info('Emoji Bot already initialized.')

        if use_classifier:
            ChatbotManager.classifier = AttentionLSTM()
            ChatbotManager.classifier = load_model_classifier(ChatbotManager.classifier)
            print("load classifier !!!!!!!!!!!!!")

    @staticmethod
    def callBot(sentence, p=0, port=0):
        """ Use the previously instantiated bot to predict a response to the given sentence
        Args:
            sentence (str): the question to answer
        Return:
            str: the answer
        """
        if ChatbotManager.bot_list and ChatbotManager.emoji_bot:
            # score the candidates
            num = 4
            Answers = []
            Scores = np.zeros(num)
            for i  in range(num):
                answer = ChatbotManager.bot_list[p].predict(sentence, port)
                Answers.append(answer)
                score = predict_class(ChatbotManager.classifier, answer, person=p)
                Scores[i] = score
            best = np.argmax(Scores)
            final_answer = Answers[best]
            print("final answer:", final_answer)
            # print("score", score)

            prob = np.random.rand()
            if prob < 0.2:
                return final_answer + ChatbotManager.emoji_bot.predict_class(sentence)
            elif prob > 0.8:
                return ChatbotManager.emoji_bot.predict_class(sentence) + final_answer
            else:
                return final_answer
        else:
            logger.error('Error: Bot not initialized!')
