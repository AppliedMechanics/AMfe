# -*- coding: utf-8 -*-
"""
Script for installation in developer mode
"""
import sys
import subprocess
import configparser


config = configparser.ConfigParser()
config.read('./meta.cfg')

list_of_requirements = config['condadata']['dependencies'].replace(' ', '').splitlines()[1:]
version = config['metadata']['version']

# Install dependencies
try:
    env_name = sys.argv[1]
    install_command = ['conda', 'install', '-n', '{}'.format(env_name), '-c', 'conda-forge'] + list_of_requirements
except Exception:
    env_passed = False
    install_command = ['conda', 'install', '-c', 'conda-forge'] + list_of_requirements

subprocess.check_call(install_command)
