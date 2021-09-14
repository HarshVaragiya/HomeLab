#!/bin/bash
docker run --name gotify -d -p 51443:51443 --restart always -v $PWD/conf:/etc/gotify/ -v $PWD/data:/app/data gotify/server
