import os
import sys

# Adiciona o diretório raiz do projeto ao sys.path
sys.path.insert(0, os.path.abspath('../..'))

# Informações do projeto
project = 'FeynmS'
copyright = '2025, Seu Nome'
author = 'Seu Nome'
release = '0.1.2'  # Versão atual do projeto

# Extensões do Sphinx
extensions = [
    'sphinx.ext.autodoc',          # Documentação automática de docstrings
    'sphinx.ext.napoleon',         # Suporte a docstrings estilo Google/NumPy
    'sphinx.ext.viewcode',         # Links para o código-fonte
    'sphinx_autodoc_typehints',    # Suporte a anotações de tipo
    'myst_parser',                 # Suporte a Markdown
]

# Configurações do tema
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Configurações do autodoc
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'special-members': '__init__',
}

# Suporte a Markdown
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
