FROM ruby:3.3.4-bookworm

# Install base, Ruby, Headers, Jekyll, Bundler, github-pages Export Path
RUN apt update
RUN apt upgrade -y
RUN apt install -y curl wget bash cmake

RUN apt install -y ruby-full libvips-tools libwebp7 libpng-dev

RUN export PATH="/root/.rbenv/bin:$PATH"
RUN rm -rf /var/cache/apt/*
# Install Jekyll and required gems
# RUN gem install bundler -v 2.4.22
# RUN gem install github-pages
# RUN mkdir /home/mypage

ARG USERNAME=devcontainer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

# Set Working directory
WORKDIR /workspace
