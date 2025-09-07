# langgraph-advanced-ai-agent
A more advanced AI agent written in python that uses langgraph and bright data. 
Bright data APIs and web scrapers are being utilized to fetch results from google, bing, yandex, reddit.
This is done by taking advantage of the langgraph library in order represent each data search as a separate state graph node and combine the various nodes to compute a directed state graph with a final end result node. 

You will need the following keys configured in an .env configuration file placed under the main project's folder.

##### BRIGHT_DATA_API_URL
##### BRIGHT_DATA_API_KEY
##### ANTHROPIC_API_KEY
