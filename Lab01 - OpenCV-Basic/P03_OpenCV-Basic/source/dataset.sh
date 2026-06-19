#!/bin/bash

gdown 18-AUodrP2NDTvPWC2NsDbhrZt9XO21OR -O ./data/ChessPieces.v1-416x416auto-orient.yolov4pytorch.zip
unzip ./data/ChessPieces.v1-416x416auto-orient.yolov4pytorch.zip -d data/
rm ./data/ChessPieces.v1-416x416auto-orient.yolov4pytorch.zip