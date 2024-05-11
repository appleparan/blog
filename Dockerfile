FROM ruby:3.1.4-bullseye

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
RUN mkdir /home/mypage
