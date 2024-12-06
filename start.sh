#!/bin/bash
exec gunicorn -b 0.0.0.0:5000 search_engine_api:appbash