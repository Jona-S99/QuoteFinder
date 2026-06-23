# ---------------------------------------------------------------------------
# Grafo de Langgraph
# - Para construir el cerebro de la app, usare Langgraph
# - Con un flujo de nodo sencillo, pero que es flexible
#   para agregar o quitar nodos segun sea necesario
# ---------------------------------------------------------------------------


from typing import TypedDict

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from app.rag.rerank import rerank_chunks
from app.rag.retrieval import retrieve_chunks

try:
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register
except Exception:
    register = None
    LangChainInstrumentor = None


# ---------------------------------------------------------------------------
# Configuraciones iniciales
# ---------------------------------------------------------------------------

LLM_MODEL = "llama3.2:3b"  # modelo de LLM a utilizar



# ---------------------------------------------------------------------------
# Observabilidad con phoenix
# ---------------------------------------------------------------------------
if register and LangChainInstrumentor:
    try:
        tracer_provider = register(
            project_name="QuoteFinder",
            endpoint="http://localhost:6006/v1/traces",
        )
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    except Exception:
        # Si observabilidad no está disponible, el workflow sigue funcionando.
        tracer_provider = None
else:
    tracer_provider = None


# ---------------------------------------------------------------------------
# Estado global del grafo
# ---------------------------------------------------------------------------
# Defino el estado global
class State(TypedDict):
    query: str
    embedding_model: str
    top_k: int
    retrieved_chunks: list[Document]
    top_ranked: int
    reranked_chunks: list[Document]
    llm_model: str
    final_answer: str


# Inicializo el grafo con el estado global
graph = StateGraph(State)


# ---------------------------------------------------------------------------
# Nodos del grafo
# ---------------------------------------------------------------------------


def retrieve_node(state: State) -> State:
    state["retrieved_chunks"] = retrieve_chunks(
        query=state["query"],
        embedding_model=state["embedding_model"],
        top_k=state["top_k"]
    )
    return state


def rerank_node(state: State) -> State:
    state["reranked_chunks"] = rerank_chunks(
        query=state["query"],
        docs=state["retrieved_chunks"],
        top_ranked=state["top_ranked"]
    )
    return state


def response_node(state: State) -> State:
    query = state["query"]
    top_ranked=state["top_ranked"]
    llm_model = state["llm_model"]

    chunks_for_prompt = [
        {
            "source_file": doc.metadata.get("source_file"),
            "participant": doc.metadata.get("participant"),
            "page_content": doc.page_content,
        }
        for doc in state["reranked_chunks"]
    ]

    prompt = f"""
    <rol>Eres un asistente de investigación que ayuda a formatear citas textuales de entrevistas.</rol>
    <tarea>A partir de los siguientes chunks de texto, extrae las citas {top_ranked} textuales y presentalas en formato Markdown, indicando la fuente de la cita</tarea>
    <chunks>{chunks_for_prompt}</chunks>
    <query_usuario>{query}</query_usuario>
    <reglas>Debes responder solo con las citas textuales encontradas en los chunks, sin agregar información adicional. Cada cita debe ir acompañada de su fuente, que se encuentra en la metadata de cada chunk bajo la clave "fuente". Si no hay citas que sirvan para responder al usuario, indica que no se encontraron citas.</reglas>
    <ejemplo_formato>
    > CITA TEXTUAL
    Fuente: fuente de la cita
    </ejemplo_formato>
    """

    # Instancio ollama
    llm = ChatOllama(model=llm_model, temperature=0)
    llm_response = llm.invoke(prompt)
    state["final_answer"] = llm_response.content
    return state


# ---------------------------------------------------------------------------
# Agregar nodos del grafo
# ---------------------------------------------------------------------------
graph.add_node("Retrieve", retrieve_node)
graph.add_node("Rerank", rerank_node)
graph.add_node("Response", response_node)


# ---------------------------------------------------------------------------
# Agregar edges al grafo
# ---------------------------------------------------------------------------
graph.add_edge(START, "Retrieve")
graph.add_edge("Retrieve", "Rerank")
graph.add_edge("Rerank", "Response")
graph.add_edge("Response", END)


# ---------------------------------------------------------------------------
# Compilar el grafo para su ejecución
# ---------------------------------------------------------------------------
graph_compiled = graph.compile()
