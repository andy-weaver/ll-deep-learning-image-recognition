FROM alpine:latest

WORKDIR ../app

RUN apk update \
    && apk add --no-cache python3 py3-pip curl git zsh \
    && pip3 install --upgrade pip \
    && python3 -m venv .venv \
    && source .venv/bin/activate \
    && ln -s /app/.venv/bin/python /usr/bin/py \
    && ln -s /app/.venv/bin/pip /usr/bin/pip \
    && git clone http://github.com/aaweaver-actuary/dotfiles.git --quiet \
    && cd dotfiles \
    && mv ./install_dotfiles /usr/bin/install_dotfiles \
    && chmod +x /usr/bin/install_dotfiles \
    && cd /app \
    && rm -rf /var/cache/apk/* \
    && rm -rf dotfiles \ 
    && install_dotfiles /app .ruff.toml .mypy.ini .prettierrc \
    && install_dotfiles $HOME .zshrc .zsh_aliases \
    && install_dotfiles /usr/bin install_devcontainer \
    && chmod +x /usr/bin/install_devcontainer \
    && install_devcontainer /app python

SHELL ["/bin/zsh", "-c"]

CMD ["/bin/zsh"]
