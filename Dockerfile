FROM jupyter/scipy-notebook:lab-3.2.1

RUN pip3 install pandas numpy jupyter_contrib_nbextensions jupyterthemes
COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt
RUN jupyter contrib nbextension install --user
RUN mkdir -p $(jupyter --data-dir)/nbextensions && cd $(jupyter --data-dir)/nbextensions && git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding && jupyter nbextension enable vim_binding/vim_binding
RUN jt -t oceans16 -vim -cursc g -nfs 9 -fs 9 -cellw 90%

USER root

RUN apt-get update && apt-get install -y inetutils-ping vim

USER jovyan
