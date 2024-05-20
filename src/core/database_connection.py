import os, sys, dotenv
from .neo4j_schema import SCHEMA
from neo4j import GraphDatabase
from semantic_kernel.functions.kernel_function_decorator import kernel_function
if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    from typing_extensions import Annotated
dotenv.load_dotenv(dotenv_path='../example.env') 

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
CUR_DIR = os.path.dirname(os.path.realpath(__file__))

class Neo4jPlugin:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    @kernel_function(name="Neo4jGraphCalling", 
                     description= "Call an API to send CYPHER query and retrieve the result from a Neo4j Graph database.",
                     )
    def get_kg_content(self, 
                       cypher: Annotated[str, "The CYPHER query"],):
        data = self.driver.execute_query(query_=cypher)
        return data
    

