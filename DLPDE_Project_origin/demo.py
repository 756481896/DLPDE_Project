#!/usr/bin/env python
# coding=utf-8
import tornado.web
import tornado.httpserver
import tornado.ioloop
import tornado.options
import os.path
import json
from models.model_demo import *
from tornado.options import define, options
define("port", default=8887, help="run on the given port", type=int)
class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", MainHandler),
            (r"/parse", ParserHandler),
        ]
        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            debug=True,
        )
        tornado.web.Application.__init__(self, handlers, **settings)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html', result='')

class ParserHandler(tornado.web.RequestHandler):
    def post(self):
        args = {k:self.get_argument(k) for k in self.request.arguments}
        print('args:', args)
        input_sentence = args.get('input_sentence', '')
        try:
            result = model.correction(input_sentence)
        except:
            result = input_sentence
        # result = input_sentence
        self.render('index.html', result=result)

if __name__ == "__main__":
    print('loading')
    model = correction_model()
    print('loaded')
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
