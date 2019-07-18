FROM continuumio/miniconda3:4.6.14

COPY environment.yml /tmp/environment.yml

RUN conda env create -qf /tmp/environment.yml

RUN echo "source activate pvfinder" > ~/.bashrc
