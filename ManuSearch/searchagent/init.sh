#!/bin/bash

cd /opt/aps/workdir/SearchAgent/vpn || exit
nohup ./clash -d . > clash.log 2>&1 &

# 等待clash启动
sleep 3

IP=127.0.0.1
PROXY_HTTP="http://${IP}:7895"
PROXY_HTTPS="http://${IP}:7895"
PROXY_SOCKS5="${IP}:7895"

# 添加代理设置到bashrc
{
    echo "export http_proxy=\"${PROXY_HTTP}\""
    echo "export https_proxy=\"${PROXY_HTTPS}\""
    echo "export ALL_PROXY=\"socks5://${PROXY_SOCKS5}\""
    echo "alias pinggg='curl -I https://www.google.com'"
} >> ~/.bashrc

# 立即应用代理设置
export http_proxy="${PROXY_HTTP}"
export https_proxy="${PROXY_HTTPS}"
export ALL_PROXY="socks5://${PROXY_SOCKS5}"

curl -I https://www.google.com