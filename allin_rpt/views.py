from django.http import HttpResponse

def hello(request):
    return HttpResponse("Hello world！ This is my first trial. [vector的笔记]")


from django.http import HttpResponse
import time

def current_time(request):
    return HttpResponse("Current time is: "+time.strftime('%Y-%m-%d %H:%M:%S'))




from django.shortcuts import render

# Create your views here.
# 添加index函数，用于返回index.html页面
def index(request):
    return render(request, 'index.html')
