from django.shortcuts import render

def mainView(request):
    """ Main view which launch and handle the chatbot view
    Args:
        request (Obj): django request object
    """
    return render(request, 'main.html', {})
    # return render(request, 'index.html', {})

def homer(request):
    return render(request, 'homer.html', {})

def marge(request):
    return render(request, 'marge.html', {})

def bart(request):
    return render(request, 'bart.html', {})

def lisa(request):
    return render(request, 'lisa.html', {})

