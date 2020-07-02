from nltk.parse.corenlp import CoreNLPServer
import os
# The server needs to know the location of the following files:
#   - stanford-corenlp-X.X.X.jar
#   - stanford-corenlp-X.X.X-models.jar
STANFORD = "../../stanford-corenlp-4.0.0"

# Create the server
server = CoreNLPServer(
   os.path.join(STANFORD, "stanford-corenlp-4.0.0.jar"),
   os.path.join(STANFORD, "stanford-corenlp-4.0.0-models.jar"),    
)

# Start the server in the background
server.start()
