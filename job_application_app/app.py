from agent import AgentEvent
from agent.agent_controller import Agent
from agent.events import EventDefinition
from core.config import Config, DebugConfig, ModelConfig, SandboxConfig

config = Config(
    sandbox=SandboxConfig(
        enabled=False
    ),
    debug=DebugConfig(
        debug_mode=True,
        suppress_policy_debug_logs=True,
        suppress_live_telemetry_terminal_logs=True
    ),
    model=ModelConfig(
        image_detail="low"
    )
)

events = [
    EventDefinition(
        name="google_visited",
        description="Google was visited",
        schema={}
    )
]

def on_event(event: AgentEvent):
    print(event)

with Agent(
    config=config,
    event_definitions=events,
    event_callback=on_event
) as agent:
    agent.execute_mission("go to google. emit google_visited when you reach there", starting_url="https://google.com")