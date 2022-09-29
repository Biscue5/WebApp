import os 
import httpx 
import streamlit as st
import pandas as pd


BACKEND_HOST = os.environ.get('BACKEND_HOST', '127.0.0.1:80')
