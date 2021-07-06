"""Some global constants."""
import os


article_folder = 'article/'
local_graphics_folder = 'graphics/'
remote_graphics_folder = os.path.join(article_folder, local_graphics_folder)


def _subfolder(graphics_folder, name):
    subfolder = os.path.join(graphics_folder, name)
    os.makedirs(subfolder, exist_ok=True)
    return subfolder


get_tab_folder = lambda folder: _subfolder(folder, 'tab')
get_fig_folder = lambda folder: _subfolder(folder, 'fig')
