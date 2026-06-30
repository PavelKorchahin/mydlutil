# show charts
from pathlib import Path
import  matplotlib.pyplot as plt
from typing import Sequence, Literal


def plot_in_one_chart(
        x: Sequence,
        y: Sequence | Sequence[Sequence],
        *line_name,
        title: str = None,
        y_lim: Sequence | Literal['default', 'no_limit'] = 'default',
        marker:str | None | Sequence[str | None] = 'o',
        save: str | Path = None,
        show: bool = True,
        ax:plt.Axes = None
) -> plt.Axes:
    """
    Draw multiple lines in one chart.
    Args:
        x (Sequence) : The x-axis data.
        y (Sequence | Sequence[Sequence]): The y-axis data.
            It can be a sequence or a sequence of sequences.

            - a sequece : the squence of y value corresponding to x.
              in this case, the length of the squence should be the same as the length of ``x``.

            - a sequence of sequences : the squence of y sequences,
              each of which is a squence of y value corresponding to x that represents a line.
              In this case, the length of each squence should be the same as the length of ``x``.

        *line_name: The name of each line.
            It can be a list of strings, or strs in the form of postional parameters, namely ``line1,line2,line3...``.
            Whatever the format of line names, the number of names should be equal to the number of lines
            (namely the number of sequence given by ``y`` ).
            and the titles should be matched to lines in order.
        title (str, optional): The title of the chart.
        y_lim (Sequence | 'default' | 'no_limit', optional): The limit of the y-axis.

            - ``'default'`` (default) : the y-axis will be set to [0, 1].

            - ``'no_limit'`` : the y-axis will not be limited.

            - a sequence with two elements: the first element is the lower limit, the second element is the upper limit.

        marker (str | Sequence[str | None], optional): The marker of each line.
            It can be a str or a sequence containing str or None
            Default is ``'o'`` .

            - None: No marker will be used.

            - str: The marker of all lines.

            - a sequence: The markers of each line where each element is a str or None, reparenting the marker of each line.
              the length of the sequence should be equal to the number of lines (namely the number of sequence given by ``y`` ).

        save(str | pathlib.Path, optional): The path of file to save the chart.
            If not specified, the chart will not be saved.
        show(bool, optional): Whether to show the chart when calling the fuction.
            Default is ``True`` .
        ax (plt.Axes, optional): The axis to draw on.
            If not specified, a new ax object will be created
    Returns:
        an object of ``plt.Axes`` , on which the lines are drawn.
    """
    if y_lim == 'default':
        y_lim = [0, 1]
    if ax is None:
        ax = plt.subplot()
    if title:
        ax.set_title(title)
    if not hasattr(y[0], '__len__'):
        y = [y]
    if isinstance(marker, str):
        marker = [marker] * len(y)
    if len(line_name) == 0:
        line_name = [None] * len(y)
    if marker is None:
        marker = [None] * len(y)
    if line_name and isinstance(line_name[0], Sequence) and not isinstance(line_name[0], str):
        line_name = line_name[0]

    assert not line_name or len(line_name) == len(y), f"The number of line names({len(line_name)}) does not match the number of y values({len(y)})."
    assert not marker or len(marker) == len(y), f"The number of markers({len(marker)}) does not match the number of y values({len(y)})."


    for i, y_data in enumerate(y):
        assert len(y_data) == len(x), (f"x and y data length does not match, "
                                       f"the number of x data is {len(x)}，the number of y data is {len(y_data)} in line {i} (start from 0) .")
        ax.plot(x, y_data, marker=marker[i], label=line_name[i])
    if not isinstance(y_lim, str):
        ax.set_ylim(y_lim)
    if len(line_name) > 0:
        ax.legend()
    if show:
        plt.show()
    if save:
        plt.savefig(save)
    return ax
