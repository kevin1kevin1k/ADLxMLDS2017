#!/usr/bin/env bash

if [ ! -d ./adl_hw1_models/ ]; then
    wget -r --no-parent --no-host-directories --cut-dirs=1 --reject="index.html*" -e robots=off https://www.csie.ntu.edu.tw/~b03902086/models/
fi

