import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from matplotlib import pyplot as plt
import io

load_dotenv()

# Set up the search tool and python repl tool
search_tool = TavilySearchResults(max_results=1, api_key=os.getenv("TAVILY_API_KEY"))
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="Executes Python code and returns the result.",
    func=python_repl.run,
)

# Initialize the LLM agent
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=1024,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY"),
)

# Create the chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can use tools."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Define the available tools
tools = [search_tool, repl_tool]

# Create a unique session ID if it doesn't exist
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{id(st.session_state)}"

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt_template)

# Initialize memory and agent executor if not in session state
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = AgentExecutor(
        agent=agent, tools=tools, checkpointer=st.session_state.memory
    )

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Add figure history to store plots
if "figure_history" not in st.session_state:
    st.session_state.figure_history = []


##############################################
# HELPER FUNCTIONS FOR FIGURE PERSISTENCE
##############################################
def save_figure_to_session(fig, message_index):
    """Save a matplotlib figure to session state with reference to message index"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.session_state.figure_history.append(
        {"message_index": message_index, "image_data": buf.getvalue()}
    )


def display_figure_from_data(fig_data):
    """Display a figure from its saved binary data"""
    if "image_data" in fig_data:
        st.image(fig_data["image_data"])


##############################################
# RESET FUNCTION
##############################################
def reset_conversation():
    """Clear all chat history and figures"""
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.figure_history = []
    st.session_state.session_id = f"session_{id(st.session_state)}"
    st.session_state.memory = MemorySaver()
    st.session_state.agent_executor = AgentExecutor(
        agent=agent, tools=tools, checkpointer=st.session_state.memory
    )


##############################################
# STREAMLIT CHAT APP SETUP
##############################################

# Create a header with title and reset button
col1, col2 = st.columns([7, 1])
with col1:
    st.title("💫 Arnab Web Search")
    st.markdown("Llama 3.3 🔗 Web Search 🔗 Python Execution")
with col2:
    if st.button("🧹", help="Clear chat history"):
        reset_conversation()
        st.success("Conversation has been reset!")

# Display previous conversation history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for fig_data in st.session_state.figure_history:
            if fig_data["message_index"] == i:
                display_figure_from_data(fig_data)

# Accept new user input
user_input = st.chat_input("Ask your question (you can trigger tool calls):")

if user_input:
    # Append user input to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append(("human", user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    # Process the query with streaming response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        figure_container = st.container()

        full_response = ""  # Store full response
        message_text = ""   # Store gradually appearing text

        max_attempts = 2
        attempts = 0
        success = False

        while attempts < max_attempts and not success:
            try:
                # Streaming response from agent
                for step in st.session_state.agent_executor.stream(
                    {
                        "input": user_input,
                        "chat_history": st.session_state.chat_history,
                    },
                    {"configurable": {"thread_id": st.session_state.session_id}},
                ):
                    if "output" in step:
                        full_response += step["output"]

                        # Word-by-word display effect
                        words = step["output"].split()
                        for word in words:
                            message_text += word + " "
                            message_placeholder.markdown(message_text)
                            time.sleep(0.1)  # Adjust speed if needed

                        # Check if any matplotlib figures were generated
                        if plt.get_fignums():
                            with figure_container:
                                fig = plt.gcf()
                                st.pyplot(fig)
                                save_figure_to_session(fig, len(st.session_state.messages))
                                plt.close(fig)

                success = True
            except Exception as error:
                if "Failed to call a function" in str(error):
                    attempts += 1
                    full_response = ""
                    continue
                else:
                    st.error(f"An unexpected error occurred: {error}")
                    break

        if not success:
            st.error("Error persists after retries. Please adjust your prompt and try again.")

        # Append the final response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.chat_history.append(("ai", full_response))


# Footer with documentation link
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: rgba(240, 242, 246, 0.9);
    color: #262730;
    text-align: center;
    padding: 3px;
    font-size: 12px;
    z-index: 999;
    border-top: 1px solid #e6e9ef;
}
.footer a {
    color: #0068c9;
    text-decoration: none;
}
.footer a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)
