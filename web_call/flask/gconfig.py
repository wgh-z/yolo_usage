import os
from gevent import monkey
monkey.patch_all()
 
import multiprocessing
 
# debug
debug = True
loglevel = 'debug'
bind = '0.0.0.0:2053'
if not os.path.exists("./log"):
    os.makedirs("./log")
pidfile = "log/gunicorn.pid"
accesslog = "log/access.log"
errorlog = "log/debug.log"
daemon = True
 
# 启动进程数
workers = multiprocessing.cpu_count()
worker_class = "gevent"
x_forwarded_for_header = "X-FORWARDED-FOR"