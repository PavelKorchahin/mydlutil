from pathlib import Path
from typing import Sequence, Callable

def debug(
        *args,
        end: str = '\n',
        stop: bool = True,
        col_space: int = 4,
        enter: int = 5,
        func : Callable = lambda x: x,
        out : Callable = print,
        **kwargs
) -> None:
    """
    Debug tools and pretty print.
    Args:
        *args: Any arguments to be printed.
        end (str, optional): The end of print each atomic element.
            Default is ``'\n'``.
        stop (bool, optional): Whether to exit the program after printing.
            It is very helpful to debug.
            Default is ``True`` .
        col_space (int, optional): The number of spaces between each atomic element,or an indentation between a parent object and its child object
            Default is 4 .
        enter (int, optional): The number of atomic elements to be printed on a line.
            Default is 5 .
        func (Callable, optional): A function to convert each atomic element to needed format before printing.
            Default is ``lambda x: x``, namely a identity function.
        out (Callable, optional): Output function.
            Default is ``print``.
        **kwargs: Any keyword arguments to be printed.

    """
    if out != print :
        _ = out
        def _out(*args, **kwargs):
            if len(args) == 0:
                print()
            else:
                _(*args)
                print(**kwargs)
        out = _out

    def print_seq(seq, col_space=col_space, enter=enter, func=lambda x: x, indent=0):
        def print_bound(seq, mode, spaces=0, is_enter=True):
            if is_enter:
                out()
                out(' ' * spaces, end='')
            if type(seq) == list:
                if mode == 'start':
                    out('[', end="")
                elif mode == 'end':
                    out(']', end="")
            elif type(seq) == tuple:
                if mode == 'start':
                    out('(', end="")
                elif mode == 'end':
                    out(')', end="")
            elif type(seq) == dict:
                if mode == 'start':
                    out('{', end="")
                elif mode == 'end':
                    out('}', end="")
        print_bound(seq, 'start', spaces=col_space * indent, is_enter=True if len(seq) else False)

        i = 1
        if len(seq) == 0:
            print_bound(seq, mode='end', spaces=0, is_enter=False)
            return
        if type(seq) in (list, tuple):

            for x in seq:
                if type(x) not in (list, tuple, dict):
                    if i == 1:
                        out()
                        out(' ' * col_space * (indent + 1), end='')
                    out(func(x), end=' ' * col_space)
                else:
                    print_seq(x, col_space=col_space, enter=enter, func=func, indent=indent+1)

                if i % enter == 0:
                    out()
                    out(' ' * col_space * (indent + 1), end='')
                i +=1
        else:
            l = len(seq)
            for k, v in seq.items():

                if i==1:
                    out()
                out(' ' * col_space * (indent + 1), end='')

                out(f'{func(k)}: ', end='')
                if type(v) not in (list, tuple, dict):
                    out(func(v), end=', ' if i != l else '')
                else:
                    indent +=1
                    print_seq(v, col_space=col_space, enter=enter, func=func, indent=indent+1)
                    indent -=1
                    if i != l:
                        out(', ', end='')
                if i !=l:
                    out()
                i += 1
        print_bound(seq, 'end', spaces=col_space * indent)
    for arg in args:
        if type(arg) not in (list, tuple, dict):
            out(func(arg), end=end)
        else:
            print_seq(arg, col_space=col_space, enter=enter, func=func)
    for k, v in kwargs.items():
        if type(v) not in (list, tuple):
            out(func(k), ':' , func(v), end=end)
        else:
            out(func(k), ':')
            out(end='\t')
            print_seq(v, col_space=col_space, enter=enter, func=func)
    if stop:
        exit()

def printf(file: str | Path, content: str | list[str], mode: str = 'a+') -> None:
    """
    Write content to file
    Args:
        file (str | pathlib.Path): Path of the file to write.
        content (str | list[str]): The content to be written.
            If specified by a sequence of strs, each str in the squence will be as a line in the file.
        mode (str): The mode of opening the file.
            Default is 'a+'.

    """
    with open(file, mode) as f:
         if not isinstance(content, Sequence) or isinstance(content, str):
             f.writelines(content)
         else:
             for x_elem in content:
                 f.writelines(x_elem)


if __name__ == '__main__':
   ...
