from agent.agent_controller import Agent
from core.config import Config, DebugConfig, ExecutionConfig, SandboxConfig

config = Config(
    sandbox=SandboxConfig(
        enabled=False
    ),
    debug=DebugConfig(
        debug_mode=True
    )
)

with Agent(config=config) as agent:
    agent.execute_mission("go to google", starting_url="https://google.com")