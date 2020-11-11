# -*- coding: utf-8 -*-
def debugprint(func):
    def wrapper(*args, **kwargs):
        print(f'\nSTARTING FUNCTION: {func.__name__}\n')
        func(*args, **kwargs)
        print(f'\nFINISHED FUNCTION: {func.__name__}\n')
    return wrapper
