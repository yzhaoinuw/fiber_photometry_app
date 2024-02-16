# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:05:21 2024

@author: yzhao
"""

import os
from dash_iconify import DashIconify
import dash_mantine_components as dmc


class FileTree:
    def __init__(self, filepath: os.PathLike):
        """
        Usage: component = FileTree('Path/to/my/File').render()
        """
        self.filepath = filepath

    def render(self) -> dmc.Accordion:
        return dmc.Accordion(
            FileTree.build_tree(self.filepath, isRoot=True), id="file-explorer"
        )

    @staticmethod
    def flatten(l):
        return [item for sublist in l for item in sublist]

    @staticmethod
    def make_file(file_name):
        return dmc.Text(
            [DashIconify(icon="akar-icons:file"), " ", file_name],
            style={"paddingTop": "5px"},
        )

    @staticmethod
    def make_folder(folder_name):
        return [DashIconify(icon="akar-icons:folder"), " ", folder_name]

    @staticmethod
    def build_tree(path, isRoot=False):
        d = []
        if os.path.isdir(path):  # if it is a folder
            children = [
                FileTree.build_tree(os.path.join(path, x)) for x in os.listdir(path)
            ]
            if isRoot:
                return FileTree.flatten(children)
            item = dmc.AccordionItem(
                [
                    dmc.AccordionControl(FileTree.make_folder(os.path.basename(path))),
                    dmc.AccordionPanel(children=FileTree.flatten(children)),
                ],
                value=path,
            )
            d.append(item)

        else:
            d.append(FileTree.make_file(os.path.basename(path)))
        return d
