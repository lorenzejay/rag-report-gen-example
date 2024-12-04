from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from rag_report_gen.tools.weaviate_tool import WeaviateTool
# Uncomment the following line to use an example of a custom tool
# from rag_report_gen.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool


@CrewBase
class RagReportGen:
    """RagReportGen crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # @before_kickoff  # Optional hook to be executed before the crew starts
    # def pull_data_example(self, inputs):
    #     # Example of pulling data from an external API, dynamically changing the inputs
    #     inputs["extra_data"] = "This is extra data"
    #     return inputs

    # @after_kickoff  # Optional hook to be executed after the crew has finished
    # def log_results(self, output):
    #     # Example of logging results, dynamically changing the output
    #     print(f"Results: {output}")
    #     return output

    @agent
    def rag_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["rag_agent"],
            verbose=True,
            tools=[WeaviateTool()],
            allow_delegation=False,
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(config=self.agents_config["reporting_analyst"], verbose=True)

    @agent
    def report_for_stakeholders_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["report_for_stakeholders_agent"],
            verbose=True,
            llm="gpt-4o-mini",
        )

    @task
    def rag_task(self) -> Task:
        return Task(
            config=self.tasks_config["rag_task"],
        )

    def task_formatter_callback(self, task):
        print("task.raw", task.raw)
        if task.raw:
            if "```" in task.raw:
                updated_raw = task.raw.replace("```", "")
                print("updated_raw", updated_raw)
                with open("outputs/report.ipynb", "w") as f:
                    f.write(updated_raw)

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config["reporting_task"],
            callback=self.task_formatter_callback,
        )

    @task
    def report_for_stakeholders_task(self) -> Task:
        return Task(config=self.tasks_config["report_for_stakeholders_task"])

    @crew
    def crew(self) -> Crew:
        """Creates the RagReportGen crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
