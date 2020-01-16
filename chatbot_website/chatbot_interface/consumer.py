from channels import Group
from channels.sessions import channel_session
import logging
import sys
import json

from .chatbotmanager import ChatbotManager


logger = logging.getLogger(__name__)


def _getClientName(client):
    """ Return the unique id for the client
    Args:
        client list<>: the client which send the message of the from [ip (str), port (int)]
    Return:
        str: the id associated with the client
    """
    return 'room-' + client[0] + '-' + str(client[1])


@channel_session
def ws_connect(message):
    """ Called when a client try to open a WebSocket
    Args:
        message (Obj): object containing the client query
    """
    if message['path'] == '/chat':  # Check we are on the right channel
        clientName = _getClientName(message['client'])
        logger.info('New client connected: {}'.format(clientName))
        Group(clientName).add(message.reply_channel)  # Answer back to the client
        message.channel_session['room'] = clientName
        message.reply_channel.send({'accept': True})

Name_dict = {"Homer":0, "Marge":1, "Bart":2, "Lisa": 3}
@channel_session
def ws_receive(message):
    """ Called when a client send a message
    Args:
        message (Obj): object containing the client query
    """
    # Get client info
    clientName = message.channel_session['room']
    port = int(clientName.split('-')[-1])
    # print("port", port, "!!!!!!!!!!")
    # print("clientname", message.channel_session, "!!!!!!!")
    data = json.loads(message['text'])
    # print("data", data)
    # print("message", message, "!!!!!!!!!!!!!!!!")
    # Compute the prediction
    question = data['message']
    # print("url", data['url'])
    name = data['url'].split('/')[-2]
    person = Name_dict[name]
    try:
        answer = ChatbotManager.callBot(question, p=person, port=port)
    except:  # Catching all possible mistakes
        logger.error('{}: Error with this question {}'.format(clientName, question))
        logger.error("Unexpected error:", sys.exc_info()[0])
        answer = 'Error: Internal problem'

    # Check eventual error
    if not answer:
        answer = 'Error: Try a shorter sentence'
        # answer = u'\U0001F60D'+'test'
    logger.info('{}: {} -> {}'.format(clientName, question, answer))

    # Send the prediction back
    Group(clientName).send({'text': json.dumps({'message': answer})})

@channel_session
def ws_disconnect(message):
    """ Called when a client disconnect
    Args:
        message (Obj): object containing the client query
    """
    clientName = message.channel_session['room']
    logger.info('Client disconnected: {}'.format(clientName))
    Group(clientName).discard(message.reply_channel)
