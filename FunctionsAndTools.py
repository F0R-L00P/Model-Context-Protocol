# FunctionsAndTools.py

import logging
import os
import re
import time
from collections import deque
from typing import List, Set, TypedDict
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# only import your generic wrapper
from llm_provider import LLMClient
from langgraph.graph import StateGraph, END

# ─── configure logger ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s • %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── initialize a “generic” LLM client ───────────────────────────────────────────
llm = LLMClient()
logger.info("✅ LLMClient wrapper initialized.")