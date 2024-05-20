import os, json, dotenv
if '__file__' in globals():
    parent_directory = os.path.dirname(os.path.realpath(__file__))
else:
    parent_directory = os.getcwd()

plugin_path = os.path.join(parent_directory, "plugins/")

from semantic_kernel import Kernel
from semantic_kernel.planners import SequentialPlanner
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from .llm_prompts import SEQUENTIAL_PLANNER_PROMPT
from .core.database_connection import Neo4jPlugin
dotenv.load_dotenv(dotenv_path='../example.env') 

class Planner:
    def __init__(self):
        self.service_id = "planner"
        self.kernel = Kernel()
        self.kernel.add_service(
            OpenAIChatCompletion(
                service_id=self.service_id,
                ai_model_id="gpt-3.5-turbo-1106",
                api_key=os.environ.get("OPENAI_API_KEY")))
        self.kernel.add_plugin(parent_directory=plugin_path, plugin_name="ChatScripts")
        self.kernel.add_plugin(parent_directory=plugin_path, plugin_name="CypherProcessing")
        self.kernel.add_plugins({"Neo4jCalling": Neo4jPlugin()})
        
        #-Sequential Planner-------------------------------------------
        self.planner = SequentialPlanner(self.kernel, self.service_id, prompt=SEQUENTIAL_PLANNER_PROMPT)
    
    async def run(self, question: str):
        sequential_plan = await self.planner.create_plan(goal=question)
        print("The plan's steps are:")
        for step in sequential_plan._steps:
            print(
                f"- {step.description.replace('.', '') if step.description else 'No description'} using {step.metadata.fully_qualified_name} with parameters: {step.parameters}"
            )

        result = await sequential_plan.invoke(self.kernel)
        print("\nStep results are:")
        for res in result.metadata['results']:
            if str(type(res.value)) == "<class 'neo4j._work.eager_result.EagerResult'>":
                try:
                    print(res.value[0][0])
                except Exception as e:
                    print(e)
            if type(res.value) == list:
                print(res.value[0].inner_content.choices[0].message.content)
        try:
            return json.loads(str(result))
        except Exception as e:
            print(e)
            return result
