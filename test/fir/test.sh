#!/usr/bin/env bash

d_root=../..

$d_root/dot2smv *_optimized.dot
$d_root/nuXmv-2.0.0-Linux/bin/nuXmv -source prove.cmd
