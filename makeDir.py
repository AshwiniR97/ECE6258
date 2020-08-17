"""
                        ECE 6258
                Digital Image Processing
                    Final Project
        Stephanie Sagun, Ashwini Rajasekhar, Dominick Freeman

    This file has functions that generate required directories for the processed
                    images to be stored
"""
import os

# Pokemons to be used
pokemons = ["charmander", "pikachu", "bulbasaur", "meowth", "squirtle"]

# Sub-directories to be made (level 1)
subDir = ["hpf_data", "binary_data", "edge_data", "blur_data", "cropped_data"]

# Sub-directories within level 1
secondDir = ["training", "testing"]


def makePokemonDir():
    for pokemon in pokemons:
        if not os.path.isdir(pokemon):
            os.mkdir(pokemon)


def makeSecondDir():
    for sec in secondDir:
        if not os.path.isdir(sec):
            os.mkdir(sec)
        os.chdir(sec)
        makePokemonDir()
        os.chdir('..')


def makeDir():
    for sub in subDir:
        if not os.path.isdir(sub):
            os.mkdir(sub)
        os.chdir(sub)
        makeSecondDir()
        os.chdir('..')
