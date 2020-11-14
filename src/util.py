# -*- coding: utf-8 -*-
from datetime import datetime


def debugprint(func):
    def wrapper(*args, **kwargs):
        print(f'\nSTARTING FUNCTION: {func.__name__} {datetime.now().strftime("%H:%M:%S")} \n')
        func(*args, **kwargs)
        print(f'\nFINISHED FUNCTION: {func.__name__} {datetime.now().strftime("%H:%M:%S")}\n')
    return wrapper
