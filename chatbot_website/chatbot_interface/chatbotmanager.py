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
from transformer import evaluate
from emoji_prediction import inference
import random
# from chatbot import chatbot


logger = logging.getLogger(__name__)


class ChatbotManager(AppConfig):
    """ Manage a single instance of the chatbot shared over the website
    """
    name = 'chatbot_interface'
    verbose_name = 'Chatbot Interface'

    bot = None
    emoji_bot = None

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

            ChatbotManager.bot = evaluate.Chatbot()
            ChatbotManager.bot.load_model()
        else:
            logger.info('Bot already initialized.')

        if not ChatbotManager.emoji_bot:
            logger.info('Initializing emoji...')
            ChatbotManager.emoji_bot = inference.Emojibot()
            ChatbotManager.emoji_bot.load_model()
        else:
            logger.info('Emoji Bot already initialized.')

    @staticmethod
    def callBot(sentence):
        """ Use the previously instantiated bot to predict a response to the given sentence
        Args:
            sentence (str): the question to answer
        Return:
            str: the answer
        """
        if ChatbotManager.bot and ChatbotManager.emoji_bot:
            p = np.random.rand()
            if p < 0.2:
                return ChatbotManager.bot.predict(sentence) + ChatbotManager.emoji_bot.predict_class(sentence)
            elif p > 0.8:
                return ChatbotManager.emoji_bot.predict_class(sentence) + ChatbotManager.bot.predict(sentence)
            else:
                return ChatbotManager.bot.predict(sentence)
        else:
            logger.error('Error: Bot not initialized!')
