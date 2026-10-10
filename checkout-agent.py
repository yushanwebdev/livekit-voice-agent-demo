import logging

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, inference, llm, room_io
from livekit.agents.beta.workflows import TaskGroup
from livekit.agents import AgentTask, function_tool, RunContext
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from dataclasses import dataclass

logger = logging.getLogger(__name__)

load_dotenv()

# Define result types for each task


@dataclass
class EmailResult:
    email_address: str


@dataclass
class AddressResult:
    address: str

# Task to collect email


class GetEmailTask(AgentTask[EmailResult]):
    def __init__(self) -> None:
        super().__init__(
            instructions="Collect the user's email address."
        )

    @function_tool()
    async def record_email(self, context: RunContext, email: str) -> None:
        """Record the user's email address"""
        self.complete(EmailResult(email_address=email))

# Task to collect shipping address


class GetAddressTask(AgentTask[AddressResult]):
    def __init__(self) -> None:
        super().__init__(
            instructions="Collect the user's shipping address."
        )

    @function_tool()
    async def record_address(self, context: RunContext, address: str) -> None:
        """Record the user's shipping address"""
        self.complete(AddressResult(address=address))

# Agent that uses the task group


class CheckoutAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are a friendly checkout assistant. Guide the customer through
            the checkout process by collecting the information you need, one
            step at a time. Be concise and confirm details as you go.
            """
        )

    async def on_enter(self) -> None:
        task_group = TaskGroup()

        # Each task wrapped in lambda so it can be reinitialized if user goes back
        task_group.add(
            lambda: GetEmailTask(),
            id="email",
            description="Collect email address"
        )
        task_group.add(
            lambda: GetAddressTask(),
            id="address",
            description="Collect shipping address"
        )

        # Run tasks in sequence
        results = await task_group

        # Access results by task ID
        email = results.task_results["email"].email_address
        address = results.task_results["address"].address

        await self.session.generate_reply(
            instructions=f"Confirm the order will be sent to {email} at {address}."
        )


server = AgentServer()


# The entrypoint function runs when a participant joins the room
@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Configure the voice pipeline with STT, LLM, TTS, and VAD providers
    session = AgentSession(
        # Speech-to-text, with server-side fallback to Deepgram if AssemblyAI fails
        stt=inference.STT(
            model="assemblyai/universal-streaming",
            language="en",
            fallback=["deepgram/nova-3"],
        ),
        # Language model, with client-side fallback to Gemini if OpenAI fails
        llm=llm.FallbackAdapter([
            inference.LLM(model="openai/gpt-4.1-mini"),
            inference.LLM(model="google/gemini-2.5-flash"),
        ]),
        # Text-to-speech, with server-side fallback to Inworld if Cartesia fails
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
            fallback=[{"model": "inworld/inworld-tts-1", "voice": "Ashley"}],
        ),
        vad=silero.VAD.load(),                    # Voice activity detection
        turn_detection=MultilingualModel(),  # Semantic turn detection
        preemptive_generation=True,
    )

    # Start the session with noise cancellation enabled
    await session.start(
        agent=CheckoutAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),  # Background voice cancellation
            ),
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(server)
