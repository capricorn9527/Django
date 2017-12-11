#coding:utf-8
from __future__ import unicode_literals # compatible with python3 unicode

from django.http import HttpResponse
from django.shortcuts import render
from deepnlp import segmenter
from deepnlp import pos_tagger
import argparse
import logging
import pandas
import chardet
import unicodecsv
import json
tagger = pos_tagger.load_model(lang = 'zh')

def hello(request):
    # return HttpResponse("Hello world ! ")
    text=request.GET.get("text")
    context = parse_data(text)
    #返回页面
    # return render(request, 'hello.html', context)
    #返回内容
    return HttpResponse(json.dumps(context, encoding="UTF-8", ensure_ascii=False))


def parse_data(text):
    words = segmenter.seg(text);
    words_p = "";
    context = {}
    # POS Tagging
    tagging = tagger.predict(words);
    for (w, t) in tagging:
        context[w] = t
    #tmp = text.encode("utf-8") + " : ".encode("utf-8") + words_p.encode("utf-8")
    return context;