#!/bin/sh



base=/mnt/projects/USS_MEP/MEP 

find "$base" -type f -print0 |xargs -0 -- chmod ug+rw
find "$base" -type d -print0 |xargs -0 -- chmod ug+rwx
