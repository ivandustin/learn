#!/usr/bin/env bash
set -ex
black bin/* .
flake8 bin/* .
