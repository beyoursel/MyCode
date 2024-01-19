import functools


def print_log(print_str, log, same_line=False, display=True):

    """
    parameters:
        print_str: a string to print
        log:       a opened file to save the log
        same_line: True if we want to print the string without a new next line
        display:   False if we want to disable to print the string
    """
    if display:
        if same_line: print('{}'.format(print_str), end='')
        else: print('{}'.format(print_str))

    if same_line: log.write('{}'.format(print))
    else: log.write('{}\n'.format(print_str))
    log.flush() # save into the memory immediately without cache


def log(text, log, array=None):
    """
    Print a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20} min: {:10.5f} max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print_log(text, log=log)

