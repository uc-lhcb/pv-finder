FROM mambaorg/micromamba:0.13.1
COPY environment.yml /root/environment.yml
RUN micromamba install -y -n base -f /root/environment.yml && \
    micromamba clean --all --yes
